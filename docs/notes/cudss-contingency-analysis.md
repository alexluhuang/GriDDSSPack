# cuDSS contingency-analysis investigation

This note tracks the investigation and design for accelerating contingency
analysis with NVIDIA cuDSS while preserving the existing invocation, inputs,
and outputs. Source locations refer to commit
`bb411473c48d0e17aecfe52703546f72bb5bd00d`.

## Current-state architecture

### Build and process lifetime

`src/CMakeLists.txt` always includes
`applications/contingency_analysis`. Its CMake target builds `ca.x` from
`ca_main.cpp` and `ca_driver.cpp`, links the power-flow, component, math,
analysis, Global Arrays, PETSc, and MPI libraries, and installs the executable
to `bin`.

`ca_main.cpp` owns the process-wide library lifetime:

```text
MPI_Init
  GA_Initialize
    MA_init(C_DBL)
      gridpack::math::Initialize  -> PetscInitialize
        CADriver::execute
      GA_Terminate
    gridpack::math::Finalize      -> PetscFinalize
MPI_Finalize
```

Objects created by `CADriver::execute()` are destroyed before
`GA_Terminate()`. A replacement executor must retain that ordering.

`CADriver::execute()` reads `argv[1]` and defaults to `input.xml`. This is the
user-facing behavior that the GPU implementation must preserve.

### Execution topology

The driver creates one world communicator and divides it by `groupSize`, which
defaults to one:

```text
MPI world communicator
  |
  +-- world-level GA TaskManager (one atomic contingency counter)
  |
  +-- task communicator 0
  |     PFNetwork 0
  |     PFAppModule 0
  |     base-case solve
  |     dynamically assigned contingency solves
  |
  +-- task communicator 1
  |     PFNetwork 1
  |     PFAppModule 1
  |     base-case solve
  |     dynamically assigned contingency solves
  |
  `-- ...
```

Task-communicator rank zero atomically claims a world-level task ID, then
shares it with the other ranks in that task communicator. A group claims its
next contingency as soon as its current one finishes.

With `mpirun -n K` and `groupSize=1`, there are `K` independent size-one task
communicators, `K` complete power-flow networks, and `K` replicated base-case
solves. Consequently:

> A size-one matrix or task communicator does not establish exclusive
> ownership of the physical GPU.

GPU ownership must be decided from the world topology. The default topology
would otherwise allow every rank to construct a cuDSS context for the same
GPU.

The current implementation has correctness assumptions that make
`groupSize>1` unsuitable as an immediate GPU path. Auto-generated event lists,
contingency lookup, island detection, and slack transfer use task-root or
rank-local state without consistently reducing it across a multi-rank task
communicator. All existing contingency-analysis tests use `groupSize=1`.

### Power-flow and contingency lifecycle

Each task communicator creates one `PFNetwork` and one reusable
`PFAppModule`. `PFAppModule::readNetwork()` parses and partitions the network,
constructs task-communicator serial I/O objects, and retains the configuration
pointer. `initialize()` loads the factory and creates exchange buffers.

Every task group then:

1. Solves the same base case.
2. Optionally repeats the solve after Q-limit conversion.
3. Marks pre-existing voltage violations to ignore.
4. Reuses its network and application for dynamically assigned contingencies.

For each assigned contingency, `ca_driver.cpp` performs:

```text
reset bus voltages and angles
update ghost buses
apply outage and save original element status
transfer/check slack and detect islands
solve nonlinear power flow, unless rejected or islanded
check slack capacity and operational violations
collect structured and/or StatBlock results
restore contingency state
clear Q-limit state and process-static warnings
close optional per-contingency output
```

Restoration order is significant. `unSetContingency()` clears island state,
restores slack selection and controller state, and restores the saved
generator or branch statuses. Q-limit state is cleared afterward because PV
restoration depends on the restored online-generator status.

`resetVoltages()` restores the bus's stored initial voltage and angle, not the
last converged base-case solution. Area-interchange and remote-regulation
mutations do not have an equally explicit rollback path, so these cases need
golden-master coverage before a multi-session executor is introduced.

### Newton and controller loops

`PFAppModule::solve()` uses real matrices and vectors through a file-local
`USE_REAL_VALUES` definition. Its nesting and ordering are:

```text
area-interchange loop
  controller loop
    set YBus and SBus component state
    create RHS mapper and mismatch vector
    create Jacobian mapper and matrix
    create RealLinearSolver
    first linear solve
    while mismatch exceeds tolerance
      apply prior correction to buses
      exchange ghost-bus state
      remap mismatch vector
      zero and remap Jacobian
      solve next linear system
      update convergence/stagnation state
    end
    if no controller change, apply final pending correction
    IREG controller
    Q-limit controller
    switched-shunt controller
    LTC controller
  end
  area-interchange controller
end
```

The first linear solve occurs before the Newton `while` loop, and the reported
Newton counter increments only after later solves. The number of linear solves
is therefore normally one more than the reported Newton-iteration count. The
last pending correction is applied only after the controller decision.
Preserving these semantics is a requirement for any stepped solve interface.

Each controller pass creates new mappers, a new Jacobian, and a new linear
solver. Within that pass, `FullMatrixMap::mapToRealMatrix(J)` calls
`MatZeroEntries()` and refills the existing PETSc allocation. Ordinary branch
outages normally retain allocated branch blocks and alter values. PV/PQ
conversion, reference-bus changes, and isolation can change dimensions or
structure. Pattern reuse must therefore be based on the exact CSR identity,
not contingency type.

The convergence summary is reset at the start of each controller pass and
describes only the final pass. Controller ordering is IREG, Q limits, switched
shunts, then LTC. A PV-to-PQ change invalidates mapper dimensions and forces a
new controller pass with new mappers.

### Linear-solver and PETSc representation

`LinearSolverT<T, I>` is a pimpl interface, but
`src/math/petsc/petsc_linear_solver.cpp` directly constructs
`PETScLinearSolverImplementation<T, I>`. It explicitly instantiates both the
real and complex constructors; those symbols must remain available.

The PETSc implementation:

- Builds a KSP and applies nested `LinearSolver/PETScOptions` configuration.
- Calls `KSPSetOperators()` and `KSPSolve()`.
- Throws when PETSc reports a negative convergence reason.
- Obtains the native objects through the existing `PETScMatrix()` and
  `PETScVector()` visitor helpers.

GridPACK creates `MATSEQAIJ` and `VECSEQ` on a one-rank communicator and
`MATMPIAIJ` and `VECMPI` on a larger communicator. Runtime PETSc options can
override these types, so a cuDSS path must verify the actual type, assembly
state, scalar type, and index width. A `MATMPIAIJ` rank owns only a local row
partition and does not expose a complete global CSR system.

The initial cuDSS milestone can therefore accept only an assembled, real,
double-precision sequential AIJ matrix with matching sequential vectors.
PETSc-borrowed CSR and vector arrays must always receive their matching
restore calls.

Changing the generic `LinearSolverT` constructor without an eligibility gate
would affect power flow, dynamic simulation, state estimation, nonlinear
solvers, and other applications. A scalar backend must be selected only for
the real power-flow use case and must retain a lazily constructed PETSc
fallback with the same configuration.

The synchronous `solver.solve()` interface can support scalar validation but
cannot hide true cross-contingency batching. Batching requires a power-flow
seam at the assemble/solve/apply boundary while retaining the scalar
`PFAppModule::solve()` wrapper.

### Result and output pipeline

The current driver records:

- Optional per-contingency text output.
- Convergence and failure classification.
- Voltage and branch violations.
- Slack-capacity and islanding status.
- JSON/CSV structured results.
- Voltage, generator, and branch `StatBlock` columns and masks.
- Q-limit warnings.
- `success.txt` and aggregate statistics files.

`StatBlock` columns use `task_id + 1`, so their logical identity is
deterministic despite dynamic completion order. JSON/CSV fragments are
currently concatenated by world rank and are not sorted by task ID.

Two pre-existing issues were resolved before using these outputs as a golden
master:

1. `writeStats` is now an optional runtime setting, defaulting to the legacy
   `true` behavior. The supplied `false` value skips StatBlock allocation,
   per-contingency collection, and final files.
2. After `ca_success.getData()` fills `contingency_success`, the vector is
   no longer cleared before `success.txt` indexes it.

The second Q-limit solve's return value is now included in contingency
classification rather than ignored.

### Process-wide state and concurrency

`PFBus` stores initialization mode, Q-limit configuration, and Q-limit warning
collection in process-static data. PETSc option insertion/removal is also
process-global. Multiple live `PFAppModule` sessions in one process are
therefore not currently independent or proven thread-safe.

No threaded assembly or one-process/many-session architecture should be
selected until this state is isolated or measured under deliberate
serialization. An MPI worker/GPU-broker prototype preserves process isolation
but must account for CSR packing, transfer, synchronization, and broker load.

### Multiple-state feasibility result

The source audit provides a hard negative result for concurrent sessions in
the current revision. `PFBus::p_qlimWarnings` is one process-static
`std::vector<std::string>`; bus calculations append to it, while the
contingency driver clears and reads it between cases. Two threads solving
different sessions would concurrently mutate and clear the same vector
without synchronization, which is a C++ data race. `PFBus::p_qlim` and
`p_initStartMode` are likewise process-static configuration, and PETSc solver
objects insert and later clear values in the process-global options database.

Sequentially constructing two networks would avoid the vector data race, but
it cannot assemble independent nonlinear states concurrently for a GPU
wavefront and still shares warning/configuration identity. It also duplicates
the complete network and Global Arrays resources. A runtime concurrency probe
cannot make an architecture with a source-level data race safe, so no
threaded multiple-state experiment was run. The minimum prerequisites for
reconsideration are session-owned Q-limit state/warnings, scoped PETSc
options, explicit controller rollback, and a bounded two-session memory and
interleaving test.

## Target platform and supplied case

The measured host is a native Arm64 DGX Spark with 20 physical Arm cores, one
NVIDIA GB10 Blackwell GPU, 121 GiB of visible shared memory, driver
580.159.03, and CUDA toolkit 13.0. Docker GPU passthrough is operational.

cuDSS was not preinstalled. NVIDIA's Arm64-SBSA cuDSS 0.8.0.10 CUDA 13 archive
was checksum-verified against its redistribution manifest, and its supplied
real-double example passed on the GB10. The host does not provide the CPU
GridPACK dependency stack directly; the existing native Arm64 GridPACK image
contains Open MPI, Global Arrays, PETSc 3.24.2, and SuiteSparse/KLU.

That existing image was built with `CMAKE_BUILD_TYPE=Debug`, so it is useful
for compatibility checks but not sufficient for final CPU/GPU performance
claims. Optimized Release baselines are required.

The supplied Texas case contains 6,717 buses and 8,646 GridPACK branch
objects. It auto-generates 8,160 branch and 731 generator outages, for 8,891
contingencies. Its `groupSize` is one, controllers other than Q limits are
disabled, and PETSc is configured for `preonly` LU with KLU.

## Instrumentation and optimized CPU baseline

The baseline build uses the exact checkout in native Arm64 Release mode
(`-O3 -DNDEBUG`) with the dependency stack from `pnnl/gridpack:latest`.
Instrumentation retains the existing timers and adds:

- `Contingency: Base Case`
- `Contingency: Execute Tasks`
- `Contingency: Output Pipeline`
- `Powerflow: Controller Checks`

An opt-in `profile` setting (or `GRIDPACK_CA_PROFILE=1`) writes one
deterministically indexed CSV row per contingency. It records the outcome,
actual linear-solve calls, applied Newton corrections, legacy reported Newton
iterations, controller and area passes, and final tolerance. Normal execution
does not create this file.

The no-optional-output runs use the supplied `input.xml` unchanged:
`printCalcFiles=false`, `writeStats=false`, and text output. External wall time
includes native Arm64 container startup. Throughput is `8891 / external wall`.
The KLU fraction uses the average accumulated per-rank
`Powerflow: Solve Linear Equation` timer divided by the average complete
application timer.

| Invocation / placement | External wall (s) | Contingencies/s | Base case (s) | Matrix map (s) | RHS map (s) | KLU solve (s) | Controller checks (s) | KLU fraction | Ideal solver-only ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mpirun -n 5`, five X925 CPUs | 218.66 | 40.66 | 0.079 | 49.728 | 16.565 | 29.495 | 4.125 | 13.52% | 1.156x |
| `mpirun -n 10`, all ten X925 CPUs | 118.66 | 74.93 | 0.082 | 25.258 | 8.328 | 14.756 | 2.340 | 12.50% | 1.143x |
| `mpirun -n 15`, ten X925 + five A725 CPUs | 94.65 | 93.94 | 0.092 | 19.623 | 6.425 | 12.415 | 1.833 | 13.21% | 1.152x |
| production `mpirun -n 20`, all CPUs | 83.05 | 107.06 | 0.104 | 16.349 | 5.616 | 10.860 | 1.606 | 13.18% | 1.152x |

Dynamic task assignment compensates for the heterogeneous processor: in the
20-rank run, each A725 rank completed 365--425 tasks while each X925 rank
completed 459--509. All measured runs produced the same 8,639 successful and
252 unsuccessful entries in `success.txt`.

The production run's remaining average per-rank categories were 22.893 s
creating mappers, 3.740 s in factory operations, 3.029 s updating buses,
0.844 s mapping corrections back to buses, and 0.056 s creating linear
solvers. The world-level output pipeline was 0.101 s. StatBlock work was
disabled by the supplied `writeStats=false`; the seven-case golden run
exercised structured JSON output instead. Per-case Newton/controller metrics
were also collected on that bounded representative set. No additional
full-case output/profile sweep was run after the user requested that full
contingency runs stop.

The 10-rank run and the first 36 seconds of the 15-rank run overlapped a few
sub-second 4-by-4 solver-harness checks. They establish the scaling trend but
are not used as the final best-baseline claim. The 5-rank and production
20-rank samples were isolated from those checks.

For the production configuration:

```text
f_linear = 10.8596 / 82.3721 = 0.131836
S_max,linear-only = 1 / (1 - f_linear) = 1.1519x
```

This rejects a solver-only replacement as the dominant end-to-end
optimization. Matrix/mismatch assembly, mapper creation, state updates, and
task orchestration must remain part of any performance architecture. Scalar
cuDSS is still useful for validation and for measuring the actual device
boundary.

## Jacobian export and pattern study

`GRIDPACK_PF_CSR_EXPORT_DIR` enables a guarded diagnostic at the two
pre-linear-solve sites in `PFAppModule::solve()`. Capture also requires a
positive per-process limit and/or a case-name filter. Normal runs therefore
do no CSR extraction or file I/O. The exporter accepts only the actual
milestone-one representation: assembled real-double `MATSEQAIJ`, 32-bit
`PetscInt`, and a matching sequential vector. All PETSc array borrows have
matching restore calls.

The binary `.gpcsr` v1 format is explicitly little-endian and contains a
40-byte versioned header followed by zero-based 32-bit CSR, binary64 values,
and binary64 RHS data. A per-process CSV manifest records the case, nonlinear
and controller positions, dimensions, exact pattern and numeric hashes, and
enabled controller features.

To avoid another full sweep, the study used a seven-contingency Texas subset:
a typical branch, a difficult convergent branch, a divergent branch, an
islanding branch, a slack-overload branch, a typical generator, and a
slack-overload generator. The one-rank run completed in 1.44 seconds and
captured all 42 systems that reached a linear solve.

| Case | Systems | Controller passes | Exact patterns | Dimensions / nonzeros |
|---|---:|---:|---:|---|
| base case | 2 | 1 | 1 | 12,955 / 91,241 |
| typical branch | 3 | 1 | 1 | 12,955 / 91,241 |
| difficult convergent branch | 10 | 2 | 2 | 12,955 / 91,241; 12,956 / 91,248 |
| divergent branch | 14 | 1 | 1 | 12,955 / 91,241 |
| slack-overload branch | 4 | 1 | 1 | 12,953 / 91,233 |
| typical generator | 5 | 2 | 2 | 12,956 / 91,248; 12,958 / 91,262 |
| slack-overload generator | 4 | 1 | 1 | 12,955 / 91,241 |

There were five exact patterns across four dimension/nonzero classes.
Ordinary Newton iterations retained their pattern while numeric hashes
changed. Q-limit/controller transitions changed both dimension and structure.
The difficult-branch and typical-generator cases each produced a
12,956-by-12,956 matrix with 91,248 entries, but their exact patterns differed.
Consequently, dimensions and nonzero counts are insufficient cache keys.
Analysis reuse and any future uniform batches must be bucketed by exact CSR
identity.

The branch-outage cases normally retained explicitly allocated entries and
shared the base pattern, so a universal superpattern is unnecessary for the
common class. The bounded sample does not exercise switched-shunt, LTC,
remote-regulation, or area-interchange transitions because those controllers
are disabled in the supplied input; those states remain outside the measured
claim.

## Standalone solver result

The independent harness reads the exported format without linking GridPACK
and compares KLU with cuDSS device and hybrid modes. It measures cold
analysis/factor/solve, analysis-reuse factor/solve, and repeated-RHS solve
lifecycles separately, synchronizing the CUDA stream around every measured
cuDSS phase.

For the representative base system (12,955 rows, 91,241 entries), all
backends passed with scaled residual at most `1.15e-18`. The largest cuDSS
versus KLU difference was `2.44e-13` relative L2 and `1.55e-17` absolute.

| Median lifecycle time | KLU | cuDSS device | cuDSS hybrid |
|---|---:|---:|---:|
| cold analysis + factor + solve | 8.351 ms | 27.909 ms | 19.755 ms |
| analysis reused, factor + solve | 6.044 ms | 1.380 ms | 1.119 ms |
| factor reused, repeated solve | 0.158 ms | 0.186 ms | 0.177 ms |

The one-time cuDSS analysis cost was approximately 25--27 ms in device mode
and 17 ms in hybrid mode, in addition to a 308 ms process-level CUDA/cuDSS
startup. KLU analysis was approximately 2.04 ms.

Three-repeat checks of four structurally representative captures confirmed
the lifecycle result. Across the base, late typical-branch, post-Q-limit
difficult-branch, and post-Q-limit generator systems, median cold times were
7.42--8.20 ms for KLU, 28.19--28.97 ms for cuDSS device, and
18.88--19.44 ms for cuDSS hybrid. With analysis reuse, KLU factor-plus-solve
was 5.24--5.70 ms versus 0.95--1.37 ms for device and 1.06--1.32 ms for
hybrid. All 36 solver/scenario checks passed; the worst scaled residual was
`1.351e-18`, and the worst relative KLU/cuDSS difference was `2.826e-13`.

A bounded 16-process cold-KLU pool over 16 distinct captures completed 64
lifecycle executions at 735/s, versus 72.9/s serially (10.09x wall-clock
speedup). Its per-process median solve lifecycle rose from 7.63 to 9.21 ms
under contention. This is not a full-application throughput prediction, but
it confirms that the GPU must compete with concurrent Arm CPU solves rather
than a single KLU call.

Scalar cuDSS is therefore slower when every short-lived GridPACK solver
object performs a cold analysis, and it does not improve a solve-only
repeated-RHS lifecycle. It is substantially faster only when the exact
pattern analysis survives across refactorizations. This result selects an
exact-pattern cache as the scalar validation design and rejects automatic
GPU selection. It does not establish that a GPU batch can beat the existing
107-contingency/s CPU-MPI execution.

## GPU ownership alternatives

### One process with multiple power-flow states

This is the shortest data path: one process would assemble several independent
systems and submit them directly to one cuDSS context. It also has the largest
unproven GridPACK state-isolation surface:

- `PFBus` Q-limit mode and warning storage are process-static.
- Configuration and PETSc options have process-wide state.
- Every live network duplicates a Texas-sized component/data graph.
- Controller rollback is incomplete for remote-regulation and
  area-interchange mutations.
- No existing test advances two `PFAppModule` instances concurrently.

Sequentially interleaved states may be usable after a feasibility probe, but
threaded assembly is not a safe phase-one assumption. This architecture
requires isolation of static state and a memory measurement before it can be
selected.

### MPI workers with one GPU broker

This retains one process-isolated power-flow state per worker. Exactly one
world rank owns the CUDA stream and cuDSS handles; workers send versioned CSR
descriptors, values, and RHS vectors and receive corrections. The ownership
invariant is explicit and independently testable.

The costs are CSR packing, transfer, synchronization at every nonlinear
wavefront, broker queueing, and a more complex failure protocol. A broker is
credible only if batched device throughput exceeds the best 20-rank CPU farm
after those measured costs. A scalar cuDSS result from every MPI rank is
explicitly prohibited: `groupSize=1` creates 20 independent rank-local
solvers, not 20 GPU owners.

The current preliminary preference for a future batched path is the MPI
broker because it preserves process isolation. No batched executor will be
implemented until exported-system benchmarks and a transfer-cost spike
support that choice.

## Solver integration decision

The selected scalar-validation boundary is the existing GridPACK
`LinearSolverT<double, int>` pimpl seam. It confines selection to real
systems, leaves the complex constructor and every unsupported representation
on PETSc, and can pass the unchanged `LinearSolver` configuration to a lazy
PETSc fallback. Integrating inside PETSc would preserve KSP configuration but
the pinned PETSc build has no cuDSS factor-solver implementation, would still
need ownership checks, and would not expose a useful future batching seam.

Eligibility is deliberately narrower than communicator size alone:

- the build explicitly enables cuDSS;
- configuration explicitly requests `cudss_device` or `cudss_hybrid`;
- the matrix and vectors are real-double sequential PETSc objects with
  32-bit indices;
- the GridPACK world contains exactly one rank and that rank is the declared
  device owner.

Strict mode turns every rejection, CUDA/cuDSS error, residual failure, or
unexpected fallback into an error. Non-strict mode constructs PETSc/KLU only
when it is actually needed. CPU-only builds and the default configuration
retain the original path.

This boundary is a correctness and lifecycle probe, not the eventual batched
executor. The synchronous `solve()` interface cannot coordinate a nonlinear
wavefront across independent contingencies.

The implemented backend is build-time optional
(`GRIDPACK_ENABLE_CUDSS=OFF` by default) and runtime opt-in. Existing XML
therefore remains PETSc/KLU without change. Validation can select cuDSS on the
unchanged input with `GRIDPACK_LINEAR_SOLVER_BACKEND=cudss`; device/hybrid,
strictness, device ID, residual tolerance, and diagnostics have corresponding
optional XML keys and environment overrides.

Each scalar solver owns a synchronized CUDA stream and cuDSS lifecycle. It
copies and validates exact sequential PETSc CSR, caches analysis state by the
full row-offset/column-index identity, runs refactorization when values change,
and reuses the factor for repeated RHS data. Statistics expose analyses,
factorizations, refactorizations, solves, cache hits/misses, fallbacks,
residual, and owner identity. A global in-process mutex serializes cuDSS API
execution. Eligibility requires both a size-one matrix communicator and
`MPI_COMM_WORLD` size one/rank zero, so an MPI CA run cannot silently mix one
GPU rank with CPU ranks.

Three focused serial tests passed in both a CUDA-free build and the CUDA
13/cuDSS 0.8 build. The GPU run exercised device analysis/cache/refactor,
hybrid execution, and strict residual failure; the CPU build exercised lazy
PETSc fallback and strict no-fallback behavior.

## Batch execution decision

No production batch executor is selected from the present measurements. A
linear-solver-only improvement is capped near 1.152x end to end, while scalar
cuDSS pays a cold-analysis cost and the existing 20-rank CPU task farm already
keeps all host cores productive. cuDSS uniform batching also requires exact
pattern-compatible systems; hybrid mode is a separate scalar strategy.

If work continues as a research prototype, the selected ownership model is a
single world-rank GPU broker fed by process-isolated MPI workers. It retains
the only currently proven isolation boundary and groups ready systems by
exact CSR pattern. Before implementation it must demonstrate, with a bounded
broker/transfer spike, throughput above the measured CPU farm after CSR
packing, MPI transfer, synchronization, and result-return costs. Rare
patterns retain PETSc/KLU. The one-process/multiple-live-state design remains
rejected until process-static PF state is isolated and memory/state rollback
are proven.

## Native Arm64 container result

`Dockerfile.cudss-arm64` is a separate native-only Release path so the
existing CPU/multi-architecture image remains unchanged. Architecture-specific
manifests pin the GridPACK dependency image and NVIDIA CUDA 13.0 devel/runtime
stages. The build downloads the official cuDSS 0.8.0.10 CUDA 13 SBSA archive,
verifies SHA-256
`c5fe7e5796792e10c3c5971bbb169ab3040ba61fe6fc99bdcbc02cf0f1ed9409`,
discovers it through its CMake config package, and rejects non-Arm64/QEMU
builds.

The runtime entrypoint preserves:

```text
docker run --rm [--gpus all] -v "$PWD":/work IMAGE input.xml
  -> /opt/gridpack/bin/ca.x input.xml
```

The 7.62 GB validation image built in 47.5 seconds. Loader inspection found no
unresolved dependencies and no startup dependency on `libcuda.so.1`, while
retaining PETSc, KLU, and SuiteSparse. A bounded IEEE-14 run passed through
the mounted entrypoint, passed without GPU by lazily falling back from
non-strict cuDSS to PETSc/KLU (26 converged, one slack overload), and passed
with strict GPU cuDSS and zero fallbacks at scaled residuals near `1e-17`.

The pinned dependency image is Ubuntu 25.10 while NVIDIA's CUDA stages are
Ubuntu 24.04. Copying the self-contained CUDA user-space runtime into the
newer-glibc image passed the loader and device smokes, but remains a documented
compatibility risk for future version changes. The host supplies the NVIDIA
driver.

## Result-preservation plan

The existing CPU contingency loop remains the reference executor. Any later
executor must produce a task-ID-indexed outcome before invoking the shared
result pipeline:

```text
task ID and contingency identity
  setup / island classification
  nonlinear convergence and controller history
  slack-capacity classification
  voltage and branch violation sets
  final bus, branch, and generator state
  Q-limit warnings
```

StatBlock columns and `success.txt` already use task identity and are suitable
for deterministic comparison. JSON/CSV fragments currently follow dynamic
world-rank completion order; they must be sorted or keyed by task ID before a
batched executor can claim byte-stable ordering.

Validation has two layers:

1. Each exported linear system is checked with a scaled residual and selected
   KLU/cuDSS solution comparison.
2. CPU and GPU power-flow runs compare convergence, islands, slack overload,
   PV-to-PQ behavior, bus voltage, branch flow, generator output, violation
   sets, `success.txt`, and structured output. Discrete operational
   classifications are pass/fail requirements; floating-point fields use
   documented tolerances.

Scalar cuDSS validation will use the existing loop and result collection, so
it cannot bypass cleanup or output work. A future wavefront executor must
share or extract that code before changing execution order.

The final golden comparison ran the seven-case Texas subset once with
PETSc/KLU and once with strict device cuDSS. `success.txt` was byte-identical
(including converged, divergent, islanded, slack-overload, and violation
classifications), and the first ten profile fields were identical for every
case: solver calls, applied updates, reported Newton iterations, and outer
passes all matched. After normalizing signed zero, the 29.5 MB structured JSON
was identical for all serialized bus, branch, generator, violation, and
iteration fields. Its only difference was the difficult case's final
tolerance, `7.440161e-9` versus `7.440072e-9`; the profile's largest
non-divergent absolute tolerance difference was `3.89e-13`. The deliberately
divergent trajectory ended at 1026.944 versus 1026.906 while retaining the
same iteration count and classification.

## Investigation gates

| Deliverable | Status |
|---|---|
| Current-state architecture map | Complete |
| Optimized CPU baseline and bottleneck report | Complete |
| Jacobian pattern study | Complete for supplied controller configuration |
| Standalone KLU/cuDSS benchmark | Scalar lifecycle measurements complete |
| GPU ownership alternatives report | Complete; broker is research preference |
| Multiple-power-flow-state feasibility result | Complete; current threaded design rejected by process-static data race |
| Solver integration boundary | Optional real scalar backend implemented and tested |
| Batch execution architecture | Deferred: current measurements do not justify implementation |
| Result-preservation plan | Seven-case CPU/strict-cuDSS golden comparison complete |
| Container dependency/version plan | Pinned native Arm64 image implemented and smoke-tested |
