# Optional cuDSS scalar backend

GridPACK continues to use PETSc unless cuDSS is enabled at build time and
selected explicitly:

```bash
cmake -S src -B build \
  -DGRIDPACK_ENABLE_CUDSS=ON \
  -DCMAKE_PREFIX_PATH=/path/to/cudss
```

The build uses the cuDSS config package and CMake's CUDA toolkit package; it
does not assume a cuDSS archive layout. A CPU-only build needs neither CUDA nor
cuDSS.

Select the backend under the existing `LinearSolver` configuration:

```xml
<LinearSolver>
  <Backend>cudss</Backend>
  <CUDSSMode>device</CUDSSMode>
  <CUDSSStrict>false</CUDSSStrict>
  <CUDSSDevice>0</CUDSSDevice>
  <CUDSSCacheMaxEntries>8</CUDSSCacheMaxEntries>
  <CUDSSResidualTolerance>1.0e-10</CUDSSResidualTolerance>
  <CUDSSDiagnostics>false</CUDSSDiagnostics>
</LinearSolver>
```

`CUDSSMode` accepts `device` and `hybrid`. Strict mode reports an error for an
unavailable/ineligible backend, a cuDSS failure, or a scaled residual above the
configured tolerance. Non-strict mode constructs PETSc lazily and uses it for
that solve and subsequent solves after a cuDSS failure.

For an unchanged input file, the same settings can be supplied as environment
overrides:

```bash
GRIDPACK_LINEAR_SOLVER_BACKEND=cudss \
GRIDPACK_CUDSS_MODE=device \
GRIDPACK_CUDSS_STRICT=true \
ca.x input.xml
```

Optional environment overrides are `GRIDPACK_CUDSS_DEVICE`,
`GRIDPACK_CUDSS_CACHE_MAX_ENTRIES`, `GRIDPACK_CUDSS_RESIDUAL_TOLERANCE`,
and `GRIDPACK_CUDSS_DIAGNOSTICS`.

The scalar validation backend is deliberately narrow: GridPACK `RealType`
must be `double`, PETSc indices must be 32-bit, PETSc must use real scalars,
the matrix and vectors must be exact sequential `MATSEQAIJ`/`VECSEQ` objects
on congruent communicators, and `MPI_COMM_WORLD` must contain only rank zero.
Complex, distributed, and otherwise ineligible systems remain on PETSc in
non-strict mode.

The process-persistent backend cache is partitioned by CUDA device, execution
mode, and configured capacity, then keyed by exact CSR row-offset and
column-index identity. Its symbolic analysis and device structure therefore
survive `RealLinearSolver` destruction without one solver changing another
solver's cache bound.
It performs analysis and factorization on a new pattern, refactorization when
only values change, and solve-only reuse when both pattern and values are
unchanged. Each device/mode partition uses least-recently-used eviction and
retains at most `CUDSSCacheMaxEntries` patterns (default 8). The limit must be
positive. `CUDSSDiagnostics=true` reports the selected device owner, configured
cache limit, and phase/cache/fallback counters when the solver is destroyed.

## Contingency-analysis GPU broker

The scalar backend above intentionally accepts only a one-rank MPI job. For
contingency analysis, one additional MPI rank can instead own the GPU and batch
systems assembled concurrently by the remaining ranks. Configure the broker
under `Contingency_analysis`, and keep the worker fallback backend on
PETSc/KLU:

```xml
<Contingency_analysis>
  <groupSize>1</groupSize>
  <CUDSSBroker>
    <Enabled>true</Enabled>
    <Device>0</Device>
    <BatchSize>8</BatchSize>
    <MinimumGpuBatchSize>8</MinimumGpuBatchSize>
    <BatchWaitMicroseconds>0</BatchWaitMicroseconds>
    <MaximumRegisteredPatterns>64</MaximumRegisteredPatterns>
    <MaximumDevicePatterns>16</MaximumDevicePatterns>
    <ScheduleByExpectedPattern>true</ScheduleByExpectedPattern>
    <ValidateResiduals>false</ValidateResiduals>
    <ResidualTolerance>1.0e-10</ResidualTolerance>
    <Strict>false</Strict>
    <Diagnostics>false</Diagnostics>
  </CUDSSBroker>
</Contingency_analysis>
<Powerflow>
  <LinearSolver>
    <Backend>petsc</Backend>
    <PETScOptions>
      -ksp_type preonly
      -pc_type lu
      -pc_factor_mat_solver_type klu
    </PETScOptions>
  </LinearSolver>
</Powerflow>
```

`BatchSize` cannot exceed the number of worker ranks. The final launch rank is
reserved as the GPU owner, so a 20-rank job has 19 contingency workers. Exact
CSR patterns are registered once; later requests send only values and the
right-hand side. A bucket is sent to cuDSS when it reaches `BatchSize` or its
wait expires with at least `MinimumGpuBatchSize` systems. Smaller and failed
buckets return to local KLU when `Strict=false`. Keep `ValidateResiduals=false`
for production because it performs an additional CPU sparse residual pass for
every GPU result.

`ScheduleByExpectedPattern` is an opt-in setting. It classifies contingencies
from the loaded network topology and generator state, groups equal predicted
initial Jacobian layouts, and dispatches them in batch-aligned epochs. For 19
workers and B=8, each full epoch contains 16 contingencies. Original task IDs
remain unchanged, preserving task-indexed success, profile, and StatBlock
outputs. The classifier has no case-name, bus-range, or network-size rules; it
handles parallel circuits, N-k outages, islands, lone buses, PV-to-PQ changes,
and reference-bus transfers from component state. Numerical Q-limit and
controller transitions that occur after solving begins are not predictable and
continue to use exact broker matching and KLU fallback. Keep this disabled for
the previously validated input-order baseline and enable it for measured A/B
runs.

For the full Texas workload on the GB10, use a zero wait. The owner drains all
request headers that are already ready before it flushes queues, so naturally
synchronized full B=8 waves still execute on cuDSS while undersized buckets
return to KLU immediately. A 5 ms wait increased GPU coverage but took 84.50 s
end to end; zero wait retained 21 full GPU batches and completed in 78.77 s.
Retune this value for workloads with different worker synchronization.

cuDSS 0.8 executes every configured uniform-batch slot. A short bucket is
padded internally, so production should normally keep
`MinimumGpuBatchSize == BatchSize` and send smaller timeout buckets to KLU.
The defaults use the B=8 gate that performed best on the target GB10; retest
before changing both values together.

The image entrypoint passes `mpirun` through, so a 20-rank container launch is:

```bash
docker run --rm --gpus all --shm-size=2g \
  --volume /absolute/path/to/case:/work \
  --workdir /work \
  gridpack-cudss-arm64:texas-broker \
  mpirun --allow-run-as-root --bind-to none --map-by slot --oversubscribe \
    -n 20 ca.x input.xml
```

Invoking the image with only `input.xml` starts one `ca.x` process. That is the
scalar validation path: it cannot form broker batches and low average GPU
utilization is expected for these short sparse solves. Broker mode requires
both the `CUDSSBroker` XML block above and the multi-rank launch shown here.

The image also contains a bounded IEEE-14 broker smoke deck. It exercises the
selected B=8 configuration without running the Texas workload:

```bash
docker run --rm --gpus all --shm-size=1g \
  gridpack-cudss-arm64:texas-broker \
  bash -lc 'cd /opt/gridpack/share/gridpack/smoke/contingency_analysis && \
    mpirun --allow-run-as-root --bind-to none --map-by slot --oversubscribe \
      -n 9 ca.x input_14_cudss_broker.xml'
```

Require the terminal `CUDSS_BROKER_SUMMARY` to report `completed > 0` and
`errors=0`.

Core binding is disabled in this command because the OpenMPI/hwloc build in
the Arm64 base image detects only 10 bindable core objects on the 20-CPU GB10
host. Linux still schedules the 20 ranks over the container's CPU set.

Successful broker runs print `CUDSS_BROKER_SUMMARY`. `completed` is the number
of systems solved on the GPU; `fallbacks` counts systems deliberately returned
to KLU, and `errors` must remain zero. Batch size and wait time should be chosen
with the capture-replay benchmark on the target host rather than inferred from
instantaneous `nvidia-smi` utilization.

The installed benchmark syntax is:

```bash
mpirun -n $((B + 1)) cudss_broker_benchmark \
  REPETITIONS MINIMUM_SYSTEMS_PER_SECOND dominant-pattern/*.gpcsr
```

It assigns B workers plus the final owner rank, requires every capture to have
one exact CSR pattern, and requires `B * REPETITIONS` to cover the capture
list. It performs one warmup wave, then times fresh CSR packing, MPI transfer,
host/device copies, refactorization, solve, and result return. Run it for
B=4, 8, 16, and the worker maximum. The mandatory gate is greater than the
measured 16-process KLU pool's 735 systems/s; use `900` as the recommended
minimum argument to retain useful margin. A nonzero exit means the throughput,
residual, exact-cache, no-fallback, or full-batch gate failed.
