# GridPACK to GriDSSPack: retained evolution to the current working state

## Executive summary

GriDSSPack is a specialized, opt-in acceleration of GridPACK's contingency-analysis
application. A contingency study asks, repeatedly, “what happens to the electrical
network if this one generator or transmission circuit is unavailable?” GridPACK
already knew how to create those cases, solve them on CPUs, check islands and
equipment controls, distribute work, and report results. GriDSSPack retained that
workflow and changed the expensive middle and output stages:

- it added NVIDIA cuDSS as an optional GPU sparse-linear-solver backend;
- it created a conservative fast path for supported pure branch-outage events
  whose equation layout remains unchanged;
- it reuses matrix structure, component-to-matrix locations, GPU storage, and,
  when convergence permits, a case's numerical factorization;
- it replaces repeated connectivity checks with one bridge-finding graph pass;
- it gives every MPI process small, dynamically assigned waves of work;
- it sends irregular, unsafe, controller-changing, or unsuccessful fast-path
  cases through the established CPU solve;
- it writes complete branch records directly from GridPACK objects and avoids a
  rewrite of the largest output by one coordinating process; and
- it adds validation, comparison, profiling, packaging, and documentation around
  the new path.

The most important conclusions are:

| Concern | Retained result |
|---|---|
| **Accuracy** | The fast assembler calls the existing GridPACK bus and branch equations, accepted cases must satisfy the configured nonlinear mismatch tolerance, and structural/topology/controller checks route incompatible cases to the full CPU path. This preserves equation fidelity but does **not** guarantee identical CPU/GPU bits, iteration counts, or outcomes at every threshold. |
| **Reporting accuracy** | In the direct `groupSize=1` flat-output path, enumeration now emits every selected physical circuit, zeros unavailable-circuit flows, applies the selected A/B/C rating consistently to capacity, loading, and violation fields, escapes contingency names, and gives retained wave cases measured convergence diagnostics instead of placeholders. |
| **Speed** | Archived end-to-end comparisons record **2.532×** and **2.488×** stock-CPU-to-GriDSSPack speedups. These are combined application speedups—not isolated GPU-kernel speedups—because Release compilation, assembly, screening, scheduling, factor reuse, and output all changed. |
| **Tradeoff** | Modified Newton (“chord”) reduces factorization work but generally requires more steps. Across the complete mixed archived run—including its CPU-routed/fallback cases—GriDSSPack averaged 4.704 reported iterations versus 1.934 for stock. |
| **Scope** | The production wave engine handles cases sequentially in one reusable, single-system cuDSS workspace per MPI rank. “Wave” means bounded scheduling and resource reuse, not simultaneous solution of many matrices. |

This report describes only behavior present in the reviewed tree. It deliberately
excludes development paths that are absent from the current implementation.

## 1. Scope, sources, and version boundary

| Property | Reviewed value |
|---|---|
| Base boundary | [`cff541f4a0fff89ddd71f254833efad50419cd68`](https://github.com/alexluhuang/GriDSSPack/commit/cff541f4a0fff89ddd71f254833efad50419cd68), inclusive |
| Committed end | [`202273b18538f75d97423cb30596d493a209f538`](https://github.com/alexluhuang/GriDSSPack/commit/202273b18538f75d97423cb30596d493a209f538) |
| Commit range | 11 retained commits in one no-merge line after the base |
| Committed net diff | 57 files, 640,981 insertions, 72 deletions; most inserted lines are archived result data |
| Additional runtime state reviewed | Seven pre-existing modified files, 650 insertions and 390 deletions, all concerned with profiling attribution and its documentation/tests |
| Report date | 2026-08-17 |

The range is reproducible with:

```bash
git log --reverse cff541f4a0fff89ddd71f254833efad50419cd68..202273b18538f75d97423cb30596d493a209f538
git diff --stat cff541f4a0fff89ddd71f254833efad50419cd68..202273b18538f75d97423cb30596d493a209f538
git diff 202273b18538f75d97423cb30596d493a209f538
```

Repository-specific statements below come from the actual commits, current
source/configuration, tests, and archived results. External sources are used only
to explain electrical, numerical, and systems concepts. This is important:
literature can justify a method, but only this repository can prove that the
method was implemented here.

The seven pre-existing modified files are the comparison
[`README`](gpucputest/README.md), [`compare_results.py`](gpucputest/compare_results.py),
and [`test_log_performance.py`](gpucputest/test_log_performance.py); the CA
[`README`](src/applications/contingency_analysis/README.md) and
[`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp); and
[`pf_batch_ca.hpp`](src/applications/modules/powerflow/pf_batch_ca.hpp) plus
[`pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp).
This report edit is separate from that pre-review runtime state.

### 1.1 What the base already contained

At the base commit, GridPACK already provided:

- branch and generator N-1 case generation and custom contingency lists;
- dynamic, one-at-a-time work allocation through `TaskManager`;
- contingency application/restoration, slack transfer, island and lone-bus
  checks, CPU AC power flow, reactive-power-limit handling, and violation checks;
- `text`, `json`, `csv`, `csv_flat`, and `csv_delta` output, branch
  filters, rating selection, the bus sidecar, convergence CSV, and optional
  statistical summaries.

Those are visible in the base [CA driver](https://github.com/alexluhuang/GriDSSPack/blob/cff541f4a0fff89ddd71f254833efad50419cd68/src/applications/contingency_analysis/ca_driver.cpp)
and are consistent with GridPACK's documented
[power-flow/contingency module](https://gridpack.readthedocs.io/en/stable/Section9-ApplicationModules.html).
They are not credited as GriDSSPack additions. The fork changes how compatible
cases are assembled and solved, how work is grouped, how exhaustive output is
produced, and how those changes are validated and packaged.

### 1.2 Retained chronological change ledger

The labels below describe the retained diff, not the wording of commit messages.

| Commit | Date | Retained contribution |
|---|---|---|
| [`10e5cf26`](https://github.com/alexluhuang/GriDSSPack/commit/10e5cf26a9572c117050302f1d40d0d95762e1a0) | 2026-07-17 | Added the optional CUDA/cuDSS build, target Docker environment, Release build, PETSc/cuDSS runtime selector, PETSc-CSR access, FP64 cuDSS direct solver, reusable wave-solver foundations, examples, tests, and validation scaffolding. |
| [`31a98d56`](https://github.com/alexluhuang/GriDSSPack/commit/31a98d56d40f9e02bb16cdeba371f78d21ac7d0d) | 2026-07-18 | Added voltage snapshots, power-flow internals needed by the fast path, the fixed-pattern assembler, direct Jacobian/RHS placement, local Y-bus repair, sequential wave execution, contingency-specific factor reuse, hybrid CPU fallback, block CSV formatting, and optional background writing. |
| [`81d9015a`](https://github.com/alexluhuang/GriDSSPack/commit/81d9015a6c3569b45ebfcd9c767c7d6aca566c4f) | 2026-07-27 | Added iterative Tarjan bridge detection to eligibility screening, including circuit-safe edge identities; production Docker builds stopped compiling tests. |
| [`d939666f`](https://github.com/alexluhuang/GriDSSPack/commit/d939666fb28b08977506595d020004ef91dca6d9) | 2026-07-27 | Put bounded waves on every MPI rank, added global counters and explicit GPU opt-in, retained CPU base/fallback solves, added direct per-circuit flat CSV, shared append and optional buffered MPI-IO, made wave/screen settings operational, changed the default output to `csv_flat`, and expanded graph tests. |
| [`ca8c245d`](https://github.com/alexluhuang/GriDSSPack/commit/ca8c245ddf3984398e2e48c4370cb0f7a5defe04) | 2026-07-28 | Added Docker build-context exclusions, image metadata, dependency cleanup, parallel builds, and user build/run documentation. |
| [`5d66fd83`](https://github.com/alexluhuang/GriDSSPack/commit/5d66fd83fe7314d6af5e9cd573408c1b54cd336a) | 2026-07-28 | Added an out-of-core CPU/GPU comparison suite for keyed coverage, duplicates, numeric differences, angles, convergence, metadata, timing, and per-rank task distribution. |
| [`570bedb8`](https://github.com/alexluhuang/GriDSSPack/commit/570bedb8251756590e1cee314c171855af76e610) | 2026-07-28 | Archived the 8,891-contingency comparison evidence now under `gpucputest/results/Texas7k`. |
| [`67c80996`](https://github.com/alexluhuang/GriDSSPack/commit/67c80996089afcaf5b7249200b322ab9b999a8a9) | 2026-07-29 | Hardened correctness and performance: live fast/reference validation, CSR and graph invariants, adaptive refactorization, real mismatch diagnostics, strict wave bounds, persistent assembler reuse, output/fallback state isolation, exact cuDSS pattern keys, output/rating fixes, diagnostics, and tests. |
| [`f0a97125`](https://github.com/alexluhuang/GriDSSPack/commit/f0a97125ee7af7071e0f444937640a2424fb546c) | 2026-07-29 | Changed only the comparison program's default data directory; no runtime solver effect. |
| [`190b46fd`](https://github.com/alexluhuang/GriDSSPack/commit/190b46fd6b435401127cb937f65d4ab22544a1eb) | 2026-07-29 | Added the evolution report and README link and moved the Texas evidence byte-for-byte into its named archive directory. |
| [`202273b1`](https://github.com/alexluhuang/GriDSSPack/commit/202273b18538f75d97423cb30596d493a209f538) | 2026-07-29 | Corrected timer lifecycle errors, added exception-safe timer scopes, added profiling/comparison tests, and archived a second much larger result set; it did not change the power-flow equations. |
| Current pre-existing worktree | Reviewed 2026-08-17 | Maps optimized work into comparable legacy GridPACK timer categories, pre-creates category identifiers identically on every rank, emits a profiling-schema marker, and prevents the comparison script from computing per-category speedups unless scopes are comparable. This changes measurement attribution, not electrical results. |

## 2. Plain-language foundations

### 2.1 What a contingency study computes

An electrical network is represented as:

- **buses**: connection points where generation, demand, or equipment meets;
- **branches**: transmission lines or transformers joining two buses; one branch
  object can contain several separately identified physical circuits; and
- **generators and loads**: sources and consumers connected to buses.

An **N-1 contingency** removes one component and solves the network again. It is
a planning question, not a prediction that the component will fail. NERC's
TPL-001-5.1 standard includes loss of one generator or transmission circuit
among its P1 events and requires post-contingency limits to be assessed. The CA
application's branch/generator studies support part of that engineering
workflow; running it is not by itself a complete standards-compliance study.
[NERC TPL-001-5.1](https://www.nerc.com/pa/Stand/Reliability%20Standards/TPL-001-5.1.pdf)

Power has several related units:

- **MW** (megawatts) measures real power that performs net work;
- **MVAr** measures reactive power associated with alternating electric and
  magnetic fields; and
- **MVA** is the magnitude of the combined real/reactive flow.

Voltage is commonly reported in **per unit (p.u.)**, a ratio to a chosen base
value, and by a phase angle in degrees. The
[MATPOWER 8.1 manual](https://matpower.org/docs/manual.pdf), sections 3–4,
provides an accessible statement of the same network and AC power-flow model.

### 2.2 Why the solve is iterative

AC power balance is nonlinear: changing a bus voltage changes currents and
power in a way that is not a straight-line relationship. GridPACK uses Newton's
method:

1. compute the **mismatch** (also called residual), meaning the remaining
   real/reactive power imbalance at the current voltages;
2. build a **Jacobian**, a matrix describing how a small voltage change should
   change that mismatch;
3. solve a linear system for a voltage correction;
4. update the voltage magnitude/angle; and
5. repeat until the largest mismatch is below the configured tolerance.

Bus roles determine which equations and unknowns appear. A reference/slack bus
sets the angle reference and balances net real power; a PV bus fixes real power
and voltage magnitude; a PQ bus fixes real and reactive power. If a generator
hits a reactive limit, a PV bus can become PQ, changing the reduced equation
layout. That is why GriDSSPack accelerates only cases with a compatible,
unchanged structural signature and sends structure-changing cases to the
general CPU path. GridPACK documents this PV-to-PQ behavior in its
[application-module guide](https://gridpack.readthedocs.io/en/stable/Section9-ApplicationModules.html).

### 2.3 Sparse matrices, CSR, and factor reuse

The Jacobian is **sparse**: most bus pairs are not directly connected, so most
matrix entries are zero. **CSR** (compressed sparse row) stores only nonzero
entries using row offsets, column numbers, and values. A sparse direct solve has
three conceptually different costs:

1. **analysis** examines the nonzero pattern and chooses an ordering;
2. **numeric factorization** decomposes the current values, typically into
   triangular factors; and
3. **solve** applies those factors to the current right-hand side.

NVIDIA exposes these as distinct cuDSS phases and requires internally consistent
three-array CSR input. Therefore an unchanged pattern can reuse analysis, and
unchanged factors can serve several right-hand sides.
[cuDSS execution API](https://docs.nvidia.com/cuda/cudss/functions.html)
[cuDSS data types](https://docs.nvidia.com/cuda/cudss/types.html)

Ordinary Newton rebuilds and factors a fresh Jacobian each iteration.
**Modified Newton**, or a **chord iteration**, holds a Jacobian/factorization for
several corrections while still recomputing the true nonlinear mismatch. It
reduces expensive factorizations but usually converges more slowly and may need
a refresh or full-Newton fallback.
[Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 25](https://epubs.siam.org/doi/10.1137/1.9780898718027.ch25)
[Ortega and Rheinboldt, *Iterative Solution of Nonlinear Equations*](https://epubs.siam.org/doi/10.1137/1.9780898719468)

### 2.4 CPU, GPU, MPI rank, and CSV

A **CPU** is the general-purpose processor. A **GPU** contains many
throughput-oriented computing units; here, NVIDIA's cuDSS library performs
sparse analysis/factor/solve work. **MPI** is the process-to-process
communication system used by GridPACK. Each MPI process is called a **rank**.
With `groupSize=1`, one rank owns a complete CA task and a serial reduced
matrix suitable for this cuDSS path.

A **CSV** file is a table represented as comma-separated text. Output can become
larger than the in-memory electrical state: exhaustive `csv_flat` contains one
row for every case/circuit pair, so output work is part of end-to-end
performance rather than an afterthought.

## 3. Current end-to-end design

The retained implementation is a conservative hybrid:

1. **Build and start.** One `ca.x` binary contains the normal PETSc path and,
   when compiled with `GRIDPACK_WITH_CUDSS=ON`, the cuDSS path. The command
   line remains `mpirun -n K ca.x input.xml`.
2. **Solve the base case on CPU.** In wave mode, the default linear backend
   deliberately remains PETSc/direct LU for the one-off base case and later
   fallbacks, while the wave engine calls cuDSS directly.
3. **Prepare reusable state once per rank.** The assembler stores base or warm
   voltages, the reduced bus-role signature, a fixed CSR pattern, direct
   component-to-CSR destinations, and an active-network graph.
4. **Screen topology once.** An iterative bridge pass marks single circuits
   whose removal disconnects the active graph.
5. **Claim a bounded wave.** Every rank asks the existing dynamic task manager
   for up to the wave size, normally eight, then repeats until no work remains.
6. **Classify each case.** Only supported branch events that are found, preserve
   one connected island and the equation layout, and satisfy fixed-pattern
   checks enter the fast path. Other cases are queued for the normal CPU path.
7. **Solve eligible cases sequentially.** One rank-local cuDSS workspace shares
   structural analysis. Each case assembles its own outaged Jacobian. In
   constant-factor mode, it factors that first Jacobian and reuses the factors
   while recomputing the true mismatch after every correction. Poor progress
   can trigger a bounded current-Jacobian refactor.
8. **Validate controls and output.** Converged state is overlaid on the
   GridPACK network. Reactive-limit, switched-shunt, and tap-changing
   controller checks can send the case to the complete CPU controller loop.
   Safe retained cases are captured before a fallback can mutate shared caches.
9. **Finish the irregular tail and restore state.** CPU fallbacks run through
   the established per-contingency workflow; base topology and voltage state
   are restored on normal and exceptional paths.

The implementation is centered in the [CA driver](src/applications/contingency_analysis/ca_driver.cpp),
[wave/Newton engine](src/applications/modules/powerflow/pf_batch_ca.hpp), and
[GridPACK assembler](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp).

## 4. Detailed retained changes, reasoning, and tradeoffs

### 4.1 Build, packaging, and backend selection

**What changed.** The [top-level CMake](src/CMakeLists.txt) adds
`GRIDPACK_WITH_CUDSS`, default `OFF`. When enabled, it finds the CUDA runtime
and cuDSS, defines `GRIDPACK_WITH_CUDSS`, and links `cudss` and
`CUDA::cudart`. The [math build](src/math/CMakeLists.txt) installs the backend
headers and builds GPU checks when tests are enabled. The
[Dockerfile](Dockerfile) now uses a CUDA 13 Ubuntu 24.04 SBSA image, pins cuDSS
0.8.0.10, targets the DGX Spark/GB10 environment, builds GridPACK as `Release`,
uses parallel dependency/project builds, and disables production-image tests.
The source contains no project CUDA kernels; it is C++ linked to NVIDIA
libraries. [`.dockerignore`](.dockerignore) excludes builds, caches, part
files, and generated CA CSVs; [`.gitignore`](.gitignore) excludes the local
GPU build directory. The [CA CMake list](src/applications/contingency_analysis/CMakeLists.txt)
and [power-flow module CMake list](src/applications/modules/powerflow/CMakeLists.txt)
copy/install the new headers and sample input with the relevant targets.

**Why.** cuDSS needs a compatible CUDA runtime and library. Pinning creates a
repeatable target environment; optional CMake support leaves normal CPU builds
available. Release compilation removes debug-build overhead from every part of
the application, not only the GPU path. Excluding generated gigabyte-scale
files reduces Docker build context.

**Tradeoffs and impact.**

- **Speed:** Release compilation is a broad CPU-side acceleration; parallel
  builds and a smaller context shorten image construction.
- **Accuracy:** the solver equations do not change. Version pinning improves
  environmental repeatability.
- **Cost:** the production image is architecture/library specific and larger;
  it does not build tests. A test-enabled build is required for release
  validation. The `CMAKE_CUDA_ARCHITECTURES` setting does not imply that this
  repository compiles custom GPU kernels.

User instructions and image details were added in [DOCKER_CA.md](DOCKER_CA.md),
[GPU_CA_IMPLEMENTATION.md](GPU_CA_IMPLEMENTATION.md),
[CODE_REVIEW_GPU_CA.md](CODE_REVIEW_GPU_CA.md), the
[root README](README.md), and the [CA README](src/applications/contingency_analysis/README.md).

### 4.2 Generic cuDSS linear-solver integration

**What changed.**

- [`linear_solver_backend.hpp`](src/math/linear_solver_backend.hpp) and
  [`.cpp`](src/math/linear_solver_backend.cpp) define process-wide PETSc and
  cuDSS choices, availability detection, names, and fallback resolution.
- [`petsc_linear_solver.cpp`](src/math/petsc/petsc_linear_solver.cpp) keeps
  the existing GridPACK linear-solver interface but constructs the selected
  implementation. PETSc remains the default; unavailable cuDSS or cuDSS
  construction failure resolves to PETSc.
- [`cudss_csr_extractor.hpp`](src/math/cudss/cudss_csr_extractor.hpp) provides
  exception-safe access to serial PETSc SeqAIJ row offsets, columns, values,
  and vector arrays.
- [`cudss_linear_solver_implementation.hpp`](src/math/cudss/cudss_linear_solver_implementation.hpp)
  owns GPU buffers/descriptors, uploads CSR values and the right-hand side,
  analyzes an exact row/column pattern, factorizes, solves using FP64
  (64-bit floating-point arithmetic), and copies the correction back. Its
  cache compares full `rowptr` and `colind`, not
  merely dimensions and nonzero counts.
- [`cudss_exception.hpp`](src/math/cudss/cudss_exception.hpp) converts CUDA,
  cuDSS, and PETSc error codes into GridPACK exceptions and centralizes cleanup.

**Why.** Keeping the standard solver abstraction avoids a second application
binary and permits non-wave callers to select cuDSS. Exact pattern comparison
is required because two matrices can have equal size and nonzero count but
different entry locations; reusing analysis across those matrices would be
unsafe.

**Tradeoffs and impact.**

- **Speed:** analysis can be reused for the same structure, but this drop-in
  path still performs CPU assembly and synchronous host/device transfers.
  One-off solves may not repay GPU setup.
- **Accuracy:** FP64, exact pattern keys, dimension checks, and PETSc fallback
  protect correctness. The generic backend supports cuDSS iterative-refinement
  and deterministic settings.
- **Limit:** `hybridMemory` is read but not applied. The primary CA wave solver
  is a separate direct cuDSS wrapper and does not inherit the generic backend's
  refinement or deterministic settings.

NVIDIA documents that deterministic mode is needed for same-architecture
bitwise reproducibility and often uses slower kernels.
[cuDSS reproducibility](https://docs.nvidia.com/cuda/cudss/general.html)

### 4.3 Reusable case state and specialized assembly

**What changed.** [PF components](src/applications/components/pf_matrix/pf_components.hpp)
can save and restore absolute voltage magnitude/angle without overwriting their
original parsed reset target. [PFAppModule](src/applications/modules/powerflow/pf_app_module.hpp)
now exposes its network, factory, configuration, and convergence injection
needed by the wave path.

The [batch assembler](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp):

- snapshots base or warm-start voltage state for every case in a wave;
- records the reduced equation size and each bus's reference/PV/PQ/isolated
  signature;
- extracts and validates a fixed 32-bit CSR layout;
- builds once a map from each GridPACK component matrix entry to its exact CSR
  slot;
- calls the existing `PFBus` and `PFBranch` matrix/vector value routines
  directly into reusable arrays;
- updates each removed circuit and its endpoint Y-bus contributions for a
  supported pure branch-outage event instead of rebuilding the entire network;
- records actual worst real- and reactive-power mismatch buses and magnitudes;
  and
- persists across waves while explicitly repairing and checking base state.

**Why.** GridPACK's general mappers support arbitrary distributed applications.
For thousands of `groupSize=1` cases with the same reduced layout, repeatedly
creating mappers, moving component data through Global Arrays (GridPACK's
distributed-array layer), discovering
locations, and rebuilding the whole admittance matrix is avoidable work. A
single branch contributes locally to its endpoint blocks, as shown by the
standard branch-admittance equations in the
[MATPOWER manual](https://matpower.org/docs/manual.pdf).

**Tradeoffs and impact.**

- **Speed:** fixed scatter locations, reused arrays/mappers, direct residual
  construction, compact voltage snapshots, and local Y-bus repair remove
  repeated CPU work and data movement.
- **Accuracy:** the electrical formulas remain the existing GridPACK component
  formulas. `GRIDPACK_BATCH_VALIDATE=1` compares live fast Jacobian/RHS
  structure and values with canonical mapper output using a
  `1e-12 × scale` tolerance and throws on disagreement.
  `GRIDPACK_BATCH_NOFAST=1` provides the canonical reference route.
- **Cost:** the specialized path is tightly coupled to component indexing and a
  fixed equation layout. Missing scatter slots, malformed offsets,
  out-of-range columns, 32-bit overflow, or a changed bus signature fail
  closed to the CPU path rather than being silently approximated.

### 4.4 cuDSS wave solver and modified Newton

**What changed.** [`cudss_batched_solver.hpp`](src/math/cudss/cudss_batched_solver.hpp)
owns one fixed-pattern device workspace, validates CSR offsets/columns,
uploads structure once, performs analysis, accepts new numeric values for
factorization, reuses factors for solves, checks finite output, and releases
resources safely.

[`PFBatchNR`](src/applications/modules/powerflow/pf_batch_ca.hpp) creates that
workspace with one system slot and advances wave cases sequentially. In exact
mode each Newton step receives a fresh Jacobian factorization. In
`constantFactor`/large-`refactorEvery` mode:

1. apply the outage and assemble that case's own first Jacobian at its starting
   voltages;
2. factor it;
3. repeatedly assemble the current true mismatch and reuse the factors for a
   correction;
4. refactor the current Jacobian on the configured stride or when progress is
   poor; and
5. stop retaining the case after bounded poor progress, non-finite values,
   solver failure, or the iteration cap, so the driver can recompute it through
   full CPU Newton.

The poor-progress threshold is `max(0.5, 1 - 0.5 × damping)`; the retained
default permits at most two adaptive poor-progress refreshes.

**Why.** Numeric factorization is generally much more expensive than applying
existing factors. A nearby post-outage starting point often allows several
cheaper chord corrections. Recomputing the nonlinear mismatch—not the linear
model's predicted mismatch—ensures the stopping test still asks whether the
original AC equations are balanced.

**Tradeoffs and impact.**

- **Speed:** structural analysis is shared within a wave, and a healthy chord
  case avoids repeated numeric factorizations.
- **Accuracy:** each case starts with its own outaged Jacobian; adaptive
  refresh and full CPU fallback handle weak convergence. Accepted fast results
  meet the same configured mismatch tolerance.
- **Cost:** chord convergence is normally linear rather than Newton's faster
  local convergence, so iteration counts rise. The nonlinear loop and assembly
  remain on the CPU, and values/RHS/solution use synchronous transfers. No
  several-system concurrent GPU solve or stream pipeline is present.

### 4.5 Topology screening and conservative eligibility

**What changed.** [`pf_screen.hpp`](src/applications/modules/powerflow/pf_screen.hpp)
implements an iterative Tarjan depth-first traversal. A **bridge** is an edge
whose removal increases the number of connected components. One
`O(V+E)` pass—linear in buses plus active circuits—classifies all recognized
single-circuit outages. Explicit edge IDs preserve parallel circuits, and an
explicit stack avoids call-stack overflow on deep networks.
[Tarjan, “A Note on Finding the Bridges of a Graph”](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1974/29303.html)

The assembler builds the graph only from active, non-isolated buses and
in-service physical circuit IDs. It records each circuit's pre-event state,
trusts the bridge shortcut only when the active base graph is connected, and
uses exact topology checks otherwise.

Fast-path eligibility is limited to pure branch-outage events that are found,
do not form multiple islands or a lone bus, and preserve the equation
signature. The Tarjan shortcut handles recognized single-circuit events;
multi-branch events can still be admitted after GridPACK performs the complete
per-case topology/signature probe. Generator events, unsupported events,
bridge/island cases, slack/structure changes, and classification failures use
the normal path.

**Why.** Rebuilding or traversing the network separately for every simple
circuit outage repeats almost the same connectivity work. A single bridge pass
answers the disconnecting-edge question exactly for the active graph.

**Tradeoffs and impact.**

- **Speed:** the graph pass removes repeated topology work for the dominant
  simple-outage class.
- **Accuracy:** circuit identity, parallel edges, active status, connected-base
  validation, and fallback are essential; a collapsed or stale graph could
  classify outages incorrectly. The screen changes routing only, not power-flow
  physics.
- **Cost:** conservative routing leaves some work on the CPU and makes the
  obtainable speedup depend on the network and contingency mix.

### 4.6 Controllers, convergence, and state isolation

**What changed.**

- Wave eligibility checks static topology and reduced structure.
- After an apparently converged overlay, reactive-limit, switched-shunt, and
  load-tap-changing-transformer checks determine whether an outer controller
  action is needed. Such a case is recomputed through the established full
  controller loop.
- Area-interchange control disables the wave path because no safe per-case
  post-wave check exists.
- The driver emits retained GPU states before CPU fallbacks rebuild shared
  factory caches, then restores the base before the fallback tail.
- Overlay recomputes residual/injection data and injects the measured iteration,
  final-tolerance, and worst-mismatch summary into `PFAppModule`.
- Rank-local initialization, wave, overlay, or restore exceptions mark that
  rank's fast path unhealthy and route current/subsequent work through CPU
  handling.

**Why.** Power flow is more than its inner Newton equations. Discrete controller
actions can change bus types or network parameters after an inner solve. A
fast voltage state is not equivalent to the normal result until those checks
pass. Shared mutable network objects also require strict capture/restore order.

**Tradeoffs and impact.**

- **Accuracy:** unsupported controller behavior is never inferred from a
  static fast state; state contamination between cases is prevented.
- **Speed:** only cases that actually require an action pay the full controller
  redo, but checks and fallback reduce the accelerated fraction.

### 4.7 Bounded all-rank scheduling

**What changed.** Instead of asking for one task at a time, every rank repeatedly
claims at most a **wave**, preserving GridPACK's shared dynamic task queue. The
default and `auto` wave size is eight. Input parsing requires a positive
integer; the effective size is capped at 256 cases and approximately 256 MiB of
voltage-state storage per rank. One assembler persists on each rank. MPI
reductions report total waves, inspected/eligible/direct-fallback cases,
nonconvergence/controller fallbacks, retained GPU cases, adaptive refactors,
avoided topology checks, and bridge counts.

**Why.** Small waves amortize setup and expose repeated structure while dynamic
re-claiming keeps ranks working when cases have unequal cost. GridPACK's task
manager already uses first-available dynamic scheduling; the retained change
groups claims without replacing that mechanism.
[GridPACK Task Manager documentation](https://gridpack.readthedocs.io/en/latest/Section7-AdvancedFunctionality.html)

**Tradeoffs and impact.**

- **Speed:** archived variation in the number of tasks assigned per rank falls
  substantially, and persistent setup is reused. Because cases have unequal
  costs, task-count variation alone does not prove equal wall-clock work.
- **Memory:** explicit caps prevent a malformed or unsuitable wave size from
  allocating unbounded voltage snapshots.
- **Cost:** every rank owns a cuDSS context, so many ranks contend for one GPU
  and consume resources. Completion and CSV row order are intentionally
  nondeterministic; `event_idx` preserves logical input order for sorting.

### 4.8 Output pipeline and reporting correctness

**What changed.**

- The default `outputFormat` changed from `text` to exhaustive `csv_flat`.
- Unless `GRIDPACK_FLAT_LEGACY` requests the older route, the `groupSize=1`
  flat path walks actual `PFBranch` objects and every circuit ID selected by
  the configured monitor allowlist or area/kV filters, reads solved complex
  flow and endpoint voltages directly, and avoids generating and reparsing
  `flow_str`/`vr_str`.
- Out-of-service or isolated circuits are emitted with zero flow.
- Base rows use rate A. Contingency rows use configured A, B→A fallback, or
  C→B→A fallback. `rate_mva`, loading percentage, and `viol` all use that
  same selected rating. A rating is the allowed equipment-flow magnitude;
  `viol` records whether the calculated MVA exceeds it. Correct rating
  selection therefore directly affects the reported overload decision.
- Flat-output contingency names receive CSV quoting and doubled internal
  quotes; a reusable `snprintf` buffer grows and retries rather than truncating
  long rows. Delta, convergence, and metadata string fields do not all use this
  escaping path.
- Each event is formatted into one block. `overlapIO=true` lets a FIFO
  background thread write blocks while the next case computes.
- With default `sharedFlatFile=true`, rank 0 creates the header, all ranks
  synchronize, and then append complete event blocks to the final file. This
  removes final rank-0 concatenation.
- `bufferFlatOutput=true` instead retains each rank's text and writes at
  disjoint offsets using `MPI_File_write_at`. Disabling both shared and
  buffered modes retains per-rank part files and rank-0 concatenation.
- Bus metadata still uses per-rank parts followed by deduplication; convergence
  rows are gathered and sorted by event index. Existing JSON, ordinary CSV,
  delta CSV, monitor filters, and optional statistical summaries remain.

Sources: [CA driver](src/applications/contingency_analysis/ca_driver.cpp),
[asynchronous writer](src/applications/contingency_analysis/ca_async_writer.hpp),
and [CA output reference](src/applications/contingency_analysis/README.md).

**Why.** Once solve time falls, exhaustive formatting and rewriting gigabytes
of part files can dominate elapsed time. Direct object access removes a
number→text→parse→number→text round trip and also exposes every circuit in a
multi-circuit object selected for that direct flat output.

**Tradeoffs and impact.**

- **Speed:** direct formatting, concurrent rank output, removal of serial
  concatenation, and optional overlap reduce output overhead.
- **Reporting accuracy:** circuit coverage, unavailable-flow handling,
  selected-rating consistency, buffer growth, and contingency-name quoting
  correct or harden the emitted dataset without changing voltages.
- **Cost:** `csv_flat` can create multi-gigabyte files and may be slower than
  the former text default. Shared C++ append was validated in the target
  environment but one application block is not guaranteed to be one atomic
  operating-system write on every NFS/parallel filesystem. Buffered MPI-IO has
  explicit offsets but can consume several gigabytes of RAM. The async FIFO is
  unbounded and close must wait for it to drain. Stream/MPI write return states
  are not comprehensively checked, so this is not a transactional output layer.
- **CSV boundary:** direct-flat contingency names are escaped, but circuit IDs,
  other output modes, and bus metadata names still assume CSV-safe strings;
  the output is therefore not an arbitrary-string RFC 4180 guarantee.

[MPI explicit-offset I/O](https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_File_write_at.3.html)
[RFC 4180 CSV rules](https://www.rfc-editor.org/info/rfc4180/)

### 4.9 Timing, comparison, tests, and documentation

**Timer integrity.** [`coarse_timer.hpp`](src/timer/coarse_timer.hpp) adds
`ScopedTimer`, whose destructor stops a category even when an exception exits
the scope. [PFAppModule](src/applications/modules/powerflow/pf_app_module.cpp)
removes duplicate/unbalanced starts and stops.

The current pre-existing worktree goes further:

- it pre-creates the 24 stock categories in identical order on every rank
  because `CoarseTimer` reduces numeric category IDs;
- it charges equivalent fast operations to the same legacy categories
  (factory work, mapper construction, matrix/vector assembly, solver
  construction/solve, bus mapping/update, and result production);
- it keeps `Contingency: Batch Preparation` as a new diagnostic for
  classification/snapshot work with no stock equivalent;
- it prints `[profiling] schema=legacy-v1 gpu_legacy_mapping=1`; and
- [`compare_results.py`](gpucputest/compare_results.py) computes category
  speedups only for an exact comparable allowlist when that marker is present.

This improves **measurement accuracy**, not electrical accuracy, and adds small
instrumentation overhead. Parent and child categories overlap and must not be
summed. A reduction in a solver category includes work avoided through reuse,
not merely faster GPU kernels.

**Validation tooling.**

- [`gpu_validation/run_validation.sh`](src/applications/contingency_analysis/gpu_validation/run_validation.sh)
  derives CPU and GPU configurations from one input and uses
  [`compare_ca_csv.py`](src/applications/contingency_analysis/gpu_validation/compare_ca_csv.py)
  to compare schemas and discrete fields exactly and numeric fields with
  absolute/relative tolerances.
- [`gpucputest/compare_results.py`](gpucputest/compare_results.py) stages
  multi-gigabyte CSVs into event-bucketed Parquet with Dask/cuDF, detects
  duplicate keys, compares row/event/circuit coverage, computes absolute and
  relative numerical metrics, treats angles circularly, compares convergence,
  statuses, buses, timing, and per-rank task distribution, and writes
  machine-readable plus Markdown reports.
- [`gpucputest/Dockerfile`](gpucputest/Dockerfile) pins the RAPIDS comparison
  environment, [`run_comparison.sh`](gpucputest/run_comparison.sh) wraps it,
  and [`gpucputest/.gitignore`](gpucputest/.gitignore) excludes generated
  comparison work directories.
- [`cudss_batched_test.cpp`](src/math/test/cudss_batched_test.cpp) checks
  known same-pattern linear systems.
- [`pf_batch_ca_test.cpp`](src/math/test/pf_batch_ca_test.cpp) checks nonlinear
  exact/chord behavior and adaptive refresh.
- [`pf_screen_test.cpp`](src/math/test/pf_screen_test.cpp) checks bridges,
  parallel circuits, invalid inputs, and a 100,000-node path.
- [`test_log_performance.py`](gpucputest/test_log_performance.py) checks the
  current profiling/comparison contract.

The comparison suite improves confidence and diagnosis but does not make the
solver faster. The two GPU-dependent C++ programs are built only with cuDSS and
are not registered as CTest tests; the production Docker image disables tests,
so release qualification remains a separate step.

## 5. Accuracy impact

“Accuracy” has three meanings here and should not be collapsed into one number.

### 5.1 Equation and convergence fidelity

| Safeguard | Accuracy effect | Tradeoff |
|---|---|---|
| Existing GridPACK component formulas feed the direct assembler | Avoids a second electrical model | Specialized indexing is coupled to current components |
| Per-case first-outage Jacobian | Factors the correct case, not merely a generic layout | One initial factorization per case |
| True nonlinear mismatch after every correction | Acceptance tests the original AC equations | Residual assembly remains a CPU cost |
| Same configured tolerance and damping | Keeps the established stopping scale/update policy | Same tolerance does not force the same iteration path or nonlinear root |
| Adaptive current-Jacobian refresh | Recovers deteriorating chord progress | Extra GPU factorization |
| Full CPU recomputation for unsuccessful/unsafe cases | Does not emit a known nonconverged chord state as a solution | Reduces accelerated fraction |
| Controller checks and area-interchange exclusion | Preserves discrete post-solve behavior where supported | Some otherwise numerically solvable cases are recomputed |

### 5.2 Structural and state safety

| Safeguard | Failure prevented |
|---|---|
| Exact CSR offsets/columns, monotonic-row and column-range validation | Reusing analysis for a different pattern or passing malformed cuDSS input |
| 32-bit size/index bounds | Integer truncation in cuDSS's retained index representation |
| Bus-role signature and missing-slot rejection | Writing values into an incompatible reduced system |
| Active, circuit-aware bridge graph and connected-base condition | Misclassifying open, parallel, or already-disconnected topology |
| Original circuit-status capture/restoration | Accidentally energizing an originally open circuit or accumulating outages |
| Finite-result checks | Accepting NaN/infinite corrections |
| Output-before-fallback ordering and paired base restoration | One case contaminating another through shared mutable caches |
| Optional live canonical-assembly oracle | Undetected disagreement between direct and standard assembly |

### 5.3 Reporting accuracy

The current source improves the truthfulness and completeness of the direct
flat data surface: every selected physical circuit can be keyed separately;
open/isolated flows are explicitly zero; selected rating, loading, and
violation agree; row buffers do not truncate silently; and flat contingency
names are escaped. Retained wave cases receive measured
iteration/tolerance/worst-mismatch fields instead of placeholders. These are
real accuracy improvements even though they do not alter the solved bus
voltage; they are not a guarantee that every string in every output mode is
escaped or that early-skipped cases have meaningful numerical diagnostics.

### 5.4 What the archived results do—and do not—show

| Archived comparison | 8,891-case Texas set | Larger 37,063-case set |
|---|---:|---:|
| Convergence outcome agreement | 100% (8,887 converged; 4 failed in both) | 99.9784% (37,011 converged in both, 44 failed in both, 5 GPU-only, 3 CPU-only) |
| Status difference | None | One `SLACK_OVERLOAD → OK` |
| Matched flat rows | 74,701,440 | 938,077,050 |
| Additional GriDSSPack rows | 4,268,160 across 494 additional circuit keys | 66,111,986 across 32,231 additional circuit keys |
| Stock-only rows | 0 | 0 |
| Violation-flag differences on matched rows | 3,361 (0.004499%) | 12,835 (0.001368%) |
| Mean absolute P difference | 0.0000697 MW | 0.0000105 MW |
| Maximum absolute P difference | 0.8464 MW | 1.9211 MW |
| Mean / maximum absolute Q difference | 0.000494 / 40.5746 MVAr | 0.0000624 / 39.233 MVAr |
| Mean / maximum absolute MVA-flow difference | 0.000171 / 10.2524 MVA | 0.0000251 / 19.046 MVA |
| Mean / maximum loading difference | 0.0000694 / 3.87 percentage points | 0.00000943 / 14.93 percentage points |
| Equipment-rating difference | 0 MVA | 0 MVA |
| Mean / maximum endpoint-voltage difference | about 0.00000061 / 0.01549 p.u. | about 0.000000088 / 0.022869 p.u. |
| Mean / maximum bus-angle difference | about 0.000060 / 0.1828° | about 0.0000264 / 0.2269° |

Sources:
[Texas summary](gpucputest/results/Texas7k/summary.md),
[Texas full report](gpucputest/results/Texas7k/comparison_report.json),
[larger summary](gpucputest/results/summary.md), and
[larger full report](gpucputest/results/comparison_report.json).

The additional rows are reporting coverage, not extra solved contingencies:
the direct groupSize-1 flat path enumerates every selected circuit ID, while
the stock output contains no corresponding keys. The nonzero maxima and rare
outcome/violation differences are why the correct claim is **tolerance-based,
measured agreement with documented outliers**, not byte-identical equivalence.

The comparator used absolute and relative tolerances of `1e-6`. On matched
rows, the within-tolerance fractions were: Texas P 92.76%, Q 90.27%, MVA
92.66%, loading 99.69%, endpoint voltage 97.95–98.02%, and angles
76.33–76.34%; larger-set P/Q/MVA 98.95–99.01%, loading 99.96%, endpoint
voltage about 99.77%, and angles about 88.21%. Ratings matched 100% in both.
These are row-weighted statistics; small means do not imply that every row or
every contingency met the comparison tolerance. The GriDSSPack-only circuit
rows have no stock counterpart, so the archives establish their added coverage
but do not numerically cross-validate their values.

In the larger archive, the eight convergence-flag disagreements and the single
status transition are not contradictory: `converged` and `status_code` are
separate recorded fields. A case can cross a numerical convergence threshold
without changing the same discrete status category, or vice versa.

Evidence limitations must remain explicit:

- The Texas files were archived by `570bedb8` before commit `67c80996`; they
  therefore cannot establish that the later hardening was exercised.
- Neither archive retains the raw CA CSV/log inputs or records an input hash,
  tested binary/source-build identity, exact run command, or hardware in its
  metadata. The larger artifact was committed with `202273b1` and contains
  25,001 bus metadata rows, but its metadata still uses the prefix
  `Texas7k_v2`. Both are tangible observations, not fully reproducible release
  certificates.
- No archived result validates the seven current profiling-only worktree edits;
  those do not change the numerical path.
- CPU and GPU floating-point operations can be ordered differently. cuDSS's
  default mode is not a bitwise-reproducibility promise, and the main wave
  solver does not enable its deterministic option. Small low-order differences
  are expected; proximity to nonlinear, overload, or controller thresholds can
  turn a small numerical difference into a discrete outcome difference.
  [NVIDIA floating-point guide](https://docs.nvidia.com/cuda/archive/9.0/floating-point/index.html)

Finally, numerical agreement with stock GridPACK is not the same as agreement
with field measurements. Both paths share the same network data and physical
model; errors in those inputs or assumptions would be shared.

## 6. Speed impact

### 6.1 Tangible end-to-end evidence

| Archived run | GriDSSPack maximum elapsed | Stock maximum elapsed | Speedup | Time reduction | Rank task-count CV, GriDSSPack vs stock |
|---|---:|---:|---:|---:|---:|
| 8,891 contingencies, 20 ranks | 84.5626 s | 214.1355 s | **2.532×** | 60.51% | 3.884% vs 17.278% |
| 37,063 contingencies, 20 ranks | 1,314.2149 s | 3,269.9674 s | **2.488×** | 59.81% | 2.053% vs 14.178% |

These values come from the archived summaries linked above. Both GriDSSPack
runs wrote **more** flat rows than stock, so the end-to-end result was not
obtained by emitting fewer flat-result rows. The archives do not establish
equality of every other output or unit of work.

Only `Total Application` is used here. The archived detailed timer files
predate the current comparable-scope marker/mapping, so their category speedups
are not reliable evidence for per-subsystem acceleration.

### 6.2 Where the retained speed comes from

| Change | Work reduced or accelerated | Main tradeoff |
|---|---|---|
| Release Docker build | All compiled CPU work | Harder source-level debugging than Debug |
| Persistent fixed CSR/scatter map | Mapper construction and location discovery | Fixed-layout eligibility requirement |
| Direct Jacobian/RHS assembly | Global Arrays/PETSc mapping traffic | More specialized code |
| Local endpoint Y-bus repair | Whole-network admittance refresh for a supported pure branch-outage event | Only safe for classified component forms |
| Warm-started voltage snapshots | Initial nonlinear distance and repeated resets | More per-wave state memory |
| One bridge pass | Repeated connectivity work | Requires an exact active multigraph |
| cuDSS structural reuse | Repeated sparse ordering/symbolic analysis | Fixed structure; GPU setup/resources |
| Per-case factor reuse | Repeated numeric factorization | More nonlinear iterations and possible refresh/fallback |
| Adaptive refactorization | Avoids wasting a whole GPU pass before CPU redo on recoverable cases | Some extra factorization/decision overhead |
| Hybrid CPU/GPU routing | Avoids GPU setup on the base and irregular tail | Not every case is accelerated |
| Bounded dynamic waves on all ranks | Setup amortization and a more even task-count distribution | Multiple contexts share one GPU; equal counts do not imply equal case cost |
| Persistent assembler across waves | Repeated pattern, scatter, and screen setup | Requires careful base-state repair |
| Retained-state output before fallback | Avoids re-solving safe GPU cases for reporting | More state-lifecycle complexity |
| Direct branch formatting | String generation/parsing round trip | Specialized `groupSize=1` path |
| Shared append / MPI offsets | Final serial file merge | Filesystem portability or memory cost |
| Optional writer thread | Overlaps output with solving | Unbounded queue and close-time drain |
| Docker ignore/parallel build | Image-build context and build wall time | No solver-runtime effect |

The archived Texas wave counters make the hybrid nature concrete: 8,891 cases
were inspected, 7,099 were initially eligible, and 6,411 were retained after
nonconvergence/controller checks. The larger artifact records 21,291 eligible
and 17,615 retained among 37,063 cases. The CPU tail is therefore part of the
measured product rather than excluded from the timing.

The 8,891-case comparison also demonstrates the central factor-reuse tradeoff:
the complete mixed GriDSSPack output reports mean/95th-percentile iterations of
4.704/11 versus stock 1.934/3. The complete pipeline was faster despite more
reported iterations; the archive does not isolate the cost of an individual
step or separate retained-GPU iteration statistics from CPU-routed cases.

### 6.3 Attribution limits

No retained benchmark isolates one modification at a time. Therefore it would
be unsound to assign exact seconds or percentages separately to cuDSS, Release
mode, direct assembly, scheduling, or output. The reported 2.532× and 2.488×
are **stock GridPACK versus the complete GriDSSPack pipeline**.

This follows the general scaling constraint that accelerating one fraction of a
program cannot remove time spent in the rest of it.
[Amdahl, 1967](https://doi.org/10.1145/1465482.1465560)
It also explains why GriDSSPack accelerates assembly, topology, scheduling, and
I/O in addition to sparse factorization.

Performance remains workload- and platform-dependent:

- small networks or few eligible cases may not amortize GPU setup;
- every rank's context contends for one physical GPU;
- CPU assembly and synchronous transfers remain;
- controller-heavy, island-heavy, generator-heavy, or structure-changing
  studies have a larger CPU tail;
- exhaustive default `csv_flat` can become I/O-bound; and
- shared append behavior and storage throughput vary by filesystem.

## 7. Current user-visible contract and limitations

### 7.1 Build and activation

CPU source builds remain the default:

```text
GRIDPACK_WITH_CUDSS=OFF
```

Wave acceleration is explicitly opt-in in the CA block:

```xml
<GPU>
  <enabled>true</enabled>
  <batched>true</batched>
  <waveSize>auto</waveSize>
  <warmStart>true</warmStart>
  <screen>true</screen>
</GPU>
```

`enabled=true` is the master switch; `Powerflow/LinearSolver/Backend=cudss`
alone does not enable CA GPU execution. In batched mode, the base and fallback
backend deliberately remains PETSc while the wave calls cuDSS directly.
`constantFactor=true`, `refactorEvery`, `chordCap`, the ordinary
`dampingFactor`, `tolerance`, and `maxIteration` configure the retained
wave/Newton behavior. The annotated sample is
[`input_14_gpu.xml`](src/applications/data_sets/input/ca/input_14_gpu.xml); the
ordinary [`input_14.xml`](src/applications/data_sets/input/ca/input_14.xml) now
also documents the disabled-by-default GPU/backend stanza.

The wave requires a cuDSS-enabled binary, a visible CUDA device, and
`groupSize=1`. Area-interchange control disables it. Reactive limits,
switched shunts, and tap changers are handled by post-wave checks plus CPU
recomputation when they act.

### 7.2 Options that should not be overstated

- `fastDecoupled` exists in a configuration structure/sample but is not used
  by the current CA wave implementation.
- `hybridMemory` is read by the generic cuDSS backend but has no retained
  effect.
- Generic-backend iterative refinement and deterministic controls do not
  configure `CuDSSBatchedSolver`, the main wave path.
- `batched` describes wave scheduling/reuse; current cases are solved
  sequentially, not with a simultaneous multi-matrix cuDSS batch.
- The sample XML's byte-for-byte output comment is stronger than the archived
  evidence and cuDSS default-mode guarantee. Use the comparison tools and
  numerical/discrete acceptance criteria instead.

### 7.3 Output behavior

`csv_flat` is now the default and deliberately favors machine-auditable
coverage over small output. Rows are not globally ordered by completion;
`event_idx` is the stable logical key. Users needing portable shared-file
behavior should validate default append on their filesystem or choose buffered
explicit-offset MPI-IO, accepting its memory cost.

## 8. Code and evidence map

| Concern | Tangible source |
|---|---|
| Build and image | [`src/CMakeLists.txt`](src/CMakeLists.txt), [`src/math/CMakeLists.txt`](src/math/CMakeLists.txt), [`Dockerfile`](Dockerfile), [`.dockerignore`](.dockerignore) |
| Backend selection | [`linear_solver_backend.hpp`](src/math/linear_solver_backend.hpp), [`linear_solver_backend.cpp`](src/math/linear_solver_backend.cpp), [`petsc_linear_solver.cpp`](src/math/petsc/petsc_linear_solver.cpp) |
| Generic cuDSS bridge | [`src/math/cudss/`](src/math/cudss/) |
| Voltage state and PF integration | [`pf_components.hpp`](src/applications/components/pf_matrix/pf_components.hpp), [`pf_app_module.hpp`](src/applications/modules/powerflow/pf_app_module.hpp), [`pf_app_module.cpp`](src/applications/modules/powerflow/pf_app_module.cpp) |
| Wave, chord, adaptive refresh | [`pf_batch_ca.hpp`](src/applications/modules/powerflow/pf_batch_ca.hpp) |
| Eligibility and direct assembly | [`pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp) |
| Bridge screen | [`pf_screen.hpp`](src/applications/modules/powerflow/pf_screen.hpp) |
| CA routing, output, profiling | [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp), [`ca_async_writer.hpp`](src/applications/contingency_analysis/ca_async_writer.hpp) |
| Small validation | [`gpu_validation/`](src/applications/contingency_analysis/gpu_validation/), [`input_14_gpu.xml`](src/applications/data_sets/input/ca/input_14_gpu.xml) |
| Large comparison tooling | [`gpucputest/README.md`](gpucputest/README.md), [`compare_results.py`](gpucputest/compare_results.py), [`run_comparison.sh`](gpucputest/run_comparison.sh) |
| Archived evidence | [Texas7k results](gpucputest/results/Texas7k/), [larger results](gpucputest/results/) |
| User documentation | [`DOCKER_CA.md`](DOCKER_CA.md), [`GPU_CA_IMPLEMENTATION.md`](GPU_CA_IMPLEMENTATION.md), [CA README](src/applications/contingency_analysis/README.md) |

Verification performed while reviewing the current tree:

- `gpubuild/math/pf_screen_test`: passed;
- `python3 gpucputest/test_log_performance.py`: 8 of 8 tests passed; and
- `git diff --check`: passed.

The cuDSS linear and nonlinear executables could not be run in this host
environment because the required PETSc/cuDSS shared libraries are not present.
No fresh full CA benchmark was generated, so this report makes no new runtime
or numerical claim beyond archived artifacts.

## 9. Overall assessment

GriDSSPack's central change is not “move GridPACK to a GPU.” It is a retained
hybrid pipeline that recognizes a regular subset of a repetitive study, removes
general-purpose setup around that subset, reuses sparse work, and preserves the
general CPU workflow as a correctness boundary.

The highest-impact **accuracy** changes are the canonical live assembly oracle,
exact CSR/structure checks, active circuit-aware topology screen, per-case
outaged factorization, true mismatch test, bounded adaptive refresh, controller
fallbacks, state restoration/isolation, measured retained-case diagnostics,
and rating-consistent selected-circuit direct flat output.

The highest-impact **speed** changes are Release compilation, persistent
fixed-pattern direct assembly, local Y-bus repair, one-pass bridge screening,
warm states, shared symbolic analysis, within-case factor reuse, dynamic bounded
all-rank waves, retained-state output, direct formatting, and removal or overlap
of serial output work.

The two archived workloads show about a **2.5× end-to-end improvement** while
writing more exhaustive output. They also show why the result must be stated
carefully: modified Newton uses more iterations, numerical maxima are not zero,
and the larger artifact has rare discrete outcome differences. The current
design manages those risks through explicit eligibility, validation, and
fallback rather than claiming universal CPU/GPU identity.

## References

1. Palmer et al., “GridPACK™: A Framework for Developing Power Grid Simulations on High-Performance Computing Platforms,” *International Journal of High Performance Computing Applications* 30(2), 2016. [PNNL record and DOI](https://www.pnnl.gov/publications/gridpacktm-framework-developing-power-grid-simulations-high-performance-computing).
2. GridPACK project, [Application Modules](https://gridpack.readthedocs.io/en/stable/Section9-ApplicationModules.html) and [Advanced Functionality / Task Manager](https://gridpack.readthedocs.io/en/latest/Section7-AdvancedFunctionality.html).
3. NERC, [TPL-001-5.1: Transmission System Planning Performance Requirements](https://www.nerc.com/pa/Stand/Reliability%20Standards/TPL-001-5.1.pdf).
4. Zimmerman and Murillo-Sánchez, [*MATPOWER User's Manual*, version 8.1](https://matpower.org/docs/manual.pdf), 2025.
5. Higham, [*Accuracy and Stability of Numerical Algorithms*, 2nd ed., ch. 25](https://epubs.siam.org/doi/10.1137/1.9780898718027.ch25), SIAM, 2002.
6. Ortega and Rheinboldt, [*Iterative Solution of Nonlinear Equations in Several Variables*](https://epubs.siam.org/doi/10.1137/1.9780898719468), SIAM reprint, 2000.
7. Davis, [*Direct Methods for Sparse Linear Systems*, ch. 8](https://epubs.siam.org/doi/10.1137/1.9780898718881.ch8), SIAM, 2006.
8. NVIDIA, [cuDSS API](https://docs.nvidia.com/cuda/cudss/functions.html), [data types](https://docs.nvidia.com/cuda/cudss/types.html), [reproducibility](https://docs.nvidia.com/cuda/cudss/general.html), and [release notes](https://docs.nvidia.com/cuda/cudss/release_notes.html).
9. Tarjan, [“A Note on Finding the Bridges of a Graph”](https://doi.org/10.1016/0020-0190(74)90003-9), *Information Processing Letters* 2(6), 1974.
10. Goldberg, [“What Every Computer Scientist Should Know About Floating-Point Arithmetic”](https://doi.org/10.1145/103162.103163), *ACM Computing Surveys* 23(1), 1991.
11. Amdahl, [“Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities”](https://doi.org/10.1145/1465482.1465560), AFIPS, 1967.
12. MPI Forum/Open MPI, [`MPI_File_write_at`](https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_File_write_at.3.html); Shafranovich, [RFC 4180](https://www.rfc-editor.org/info/rfc4180/).

## Glossary

| Term | Plain meaning |
|---|---|
| AC power flow | Calculation of steady alternating-current voltages and power flows for a specified network |
| Admittance / Y-bus | Numerical description of how readily current flows through equipment / the assembled network matrix |
| Analysis (symbolic) | Sparse-solver work based mainly on where nonzeros occur |
| Branch / circuit | A line or transformer object / one separately identified physical element within it |
| Bus | Electrical connection point |
| Chord or modified Newton | Newton-like iteration that reuses a Jacobian/factorization for several corrections |
| Controller | Discrete logic such as reactive-limit switching, shunt steps, or transformer tap changes |
| CPU fallback | Recompute a case through GridPACK's established general PETSc/full-controller path |
| CSR | Three-array compressed storage for a sparse matrix |
| CSV | Comma-separated tabular text |
| Factorization | Expensive decomposition that prepares a matrix for one or more solves |
| FP64 | Standard 64-bit floating-point number format used by the retained solvers |
| GPU / cuDSS | Graphics processor / NVIDIA sparse direct-solver library used on it |
| Jacobian | Matrix relating a small voltage change to a small mismatch change |
| Mismatch / residual | Remaining power-balance error; convergence means it is below tolerance |
| MPI rank | One parallel GridPACK process |
| MW / MVAr / MVA | Real power / reactive power / magnitude of combined apparent power |
| N-1 | Study in which one component is removed |
| PETSc | Portable scientific-computing library used by the established CPU linear-solver path |
| Per unit (p.u.) | Quantity normalized by a chosen base value |
| PV / PQ / slack bus | Bus roles that determine fixed quantities, unknowns, and balance equations |
| RHS | Right-hand side: the mismatch vector supplied to a linear solve |
| Sparse matrix | Matrix storing relatively few nonzero entries |
| Wave | Small rank-local scheduling and resource-reuse group; not a simultaneous matrix batch |
