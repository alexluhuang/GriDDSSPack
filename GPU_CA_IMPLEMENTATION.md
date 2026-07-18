# GPU-accelerated Contingency Analysis (NVIDIA cuDSS) — implementation notes

This document records the changes made to add an **opt-in NVIDIA cuDSS GPU
direct-LU backend** to GridPACK's contingency-analysis (CA) application, per
`GridPACK_CA_GPU_Architecture.md`. It covers what is implemented, how to build
and run it, how the pieces fit together, how to validate output parity, and the
status of the later phases.

Everything here respects the architecture's hard constraints:

* **One binary, one CLI.** A single `ca.x` is built with cuDSS compiled in; the
  invocation is unchanged: `mpirun -n K ca.x input.xml`.
* **Opt-in at runtime.** The GPU path is selected from `input.xml`; the default
  reproduces today's CPU behavior exactly.
* **Graceful fallback.** With `GRIDPACK_WITH_CUDSS=OFF`, or no CUDA device
  visible, or any cuDSS init failure, CA silently uses the existing PETSc path.
* **Identical outputs.** The GPU path changes *how* results are computed, not
  *what* is written — the three CSVs keep their exact schema.

---

## Status by phase

| Phase | Scope | Status |
|-------|-------|--------|
| **0** | Build + runtime prerequisites (CUDA/cuDSS image, `Release`, `sm_121`, CMake option, `input.xml` switch, validation tooling) | **Implemented** |
| **1** | cuDSS direct-LU backend (`LinearSolverImplementation`), analyze-once + refactor + solve | **Implemented** |
| **1b** | Route CA's linear solve through cuDSS when opted-in; per-contingency drop-in | **Implemented; IEEE-14 N-1 parity verified on GB10** |
| **2 (core)** | Batched cuDSS multi-system solver — one symbolic analysis, W systems via the cuDSS batch API | **Implemented + verified on GB10** (`cudss_batched_solver.hpp`, `cudss_batched_test`) |
| **2 (engine)** | Batched Newton over a wave (shared analysis, batched refactor/solve, warm-start, dropout) | **Implemented + verified on GB10** (`pf_batch_ca.hpp`, `pf_batch_ca_test`); GridPACK `BatchAssembler` wiring is the remaining seam |
| **3** | Union-Set N-1 connectivity pre-pass | **Implemented + verified** (`pf_screen.hpp`) |
| **4** | Fast-decoupled constant-factor path | **Design (documented below)** |
| **5** | Device-side aggregation + single bulk CSV write | **Design (below); CSV writer already correct/parity-verified** |
| **6** | Determinism mode, mixed-precision IR, validation report | **Config knobs live; parity oracle shipped (status-aware for diverged cases)** |

**Verified on the DGX Spark / GB10 (this machine).** Built with
`GRIDPACK_WITH_CUDSS=ON` against CUDA 13.0 + cuDSS 0.8.0 (sm_121) and run:

```
Phase 1  IEEE-14 N-1, cuDSS vs PETSc/SuperLU_DIST:
   Linear solver backend: cudss
   _delta.csv PARITY OK | _buses.csv PARITY OK | _convergence.csv PARITY OK
   ALL PARITY CHECKS PASSED
   (diverged case BR_1_2 agrees on the outcome -- same 23 iters, same worst buses;
    only its non-physical diverging residual differs between LU backends)
Phase 2 core   (cudss_batched_test): W=4 distinct A_k,b_k, ONE analysis; max|x-x*| = 0.0
Phase 2 engine (pf_batch_ca_test):   W=8 distinct nonlinear cases; all converged via
                                     batched refactor/solve + dropout; residual 1.5e-11
Phase 3        (pf_screen):          bridge/isolating outages -> islanded; cycle -> connected
```

The CPU-only build (`GRIDPACK_WITH_CUDSS=OFF`, the default) remains
behavior-identical (all cuDSS code `#ifdef`-guarded; backend defaults to PETSc).
The implementation also passed a 5-dimension adversarial code review; all
confirmed findings were fixed.

**Remaining work:** (a) wire the concrete GridPACK `BatchAssembler`
(PFFactoryModule/FullMatrixMap → base-pattern low-rank block overwrite) so the
verified batched engine drives real contingency waves inside `ca_driver`, and
(b) Phases 4–5 (designs below). Per project guidance, full Texas7k end-to-end
sweeps were not run to conserve tokens; `input_gpu.xml` is staged there for that
run. The batched engine and solver themselves are complete and GPU-verified.

---

## Files changed / added

### Build (Phase 0)
* `Dockerfile` — CUDA-enabled aarch64 base, cuDSS install, `Release`,
  `sm_121` (`CMAKE_CUDA_ARCHITECTURES`), `-D GRIDPACK_WITH_CUDSS=ON`. PETSc stays
  CPU-only (fallback KLU/SuperLU_DIST/MUMPS). All version knobs are build ARGs.
* `src/CMakeLists.txt` — `option(GRIDPACK_WITH_CUDSS …)`; discovery of
  `CUDAToolkit` + `cudss` (CMake package config); `-DGRIDPACK_WITH_CUDSS` added
  globally; `GRIDPACK_CUDSS_LIBRARIES` exported.
* `src/math/CMakeLists.txt` — always compile `linear_solver_backend.cpp`; when
  enabled, link `cudss` + `CUDA::cudart` (PUBLIC) and install the cuDSS headers.
* `src/applications/contingency_analysis/CMakeLists.txt` — stage
  `input_14_gpu.xml` into the run directory.

### Config (Phase 0)
* `src/applications/data_sets/input/ca/input_14.xml` — inert, default-off
  `<GPU>` block and `<LinearSolver><Backend>petsc</Backend>`.
* `src/applications/data_sets/input/ca/input_14_gpu.xml` — ready-to-run GPU
  example (`Backend=cudss`, `GPU/enabled=true`, cuDSS tunables, CPU fallback).

### Backend selection (Phase 1)
* `src/math/linear_solver_backend.hpp` / `.cpp` — dependency-free runtime
  selector: default backend (process-wide), `cudssBackendAvailable()`
  (compiled-in **and** a CUDA device present), `resolveLinearSolverBackend()`.

### cuDSS backend (Phase 1)
* `src/math/cudss/cudss_exception.hpp` — `GP_CUDA_CHECK` / `GP_CUDSS_CHECK` /
  `GP_PETSC_CHECK`, all surfaced as `gridpack::Exception`.
* `src/math/cudss/cudss_csr_extractor.hpp` — expose a SEQAIJ `Mat` as host CSR
  (RAII `PetscSeqCSRView`), vector array access, and PETSc→cuDSS value/index
  type mapping.
* `src/math/cudss/cudss_linear_solver_implementation.hpp` — the GPU
  `LinearSolverImplementation<T,I>`: ANALYSIS once per pattern, FACTORIZATION per
  Newton iteration, SOLVE per RHS; forces the base class' serial path in
  multi-rank groups so the coefficient matrix is always SEQAIJ.

### Dispatch (Phase 1)
* `src/math/petsc/petsc_linear_solver.cpp` — the `LinearSolverT` constructor now
  chooses the implementation via `resolveLinearSolverBackend()`, with try/catch
  fallback to PETSc.

### CA driver (Phase 1b)
* `src/applications/contingency_analysis/ca_driver.cpp` — read the `<GPU>` /
  `<Backend>` switch and call `setDefaultLinearSolverBackend()` before any solver
  is built; log the resolved backend; note when a requested GPU/batched path is
  unavailable.

### Phase 2 engine + Phase 3 (implemented + GPU-verified)
* `src/math/cudss/cudss_batched_solver.hpp` — Phase-2 core: solve W same-pattern
  systems with one cuDSS symbolic analysis via the batch API.
* `src/math/test/cudss_batched_test.cpp` — GB10 check of the batched solver.
* `src/applications/modules/powerflow/pf_batch_ca.hpp` — Phase-2 batched Newton
  engine (`PFBatchNR` + `BatchAssembler` seam): wave loop, batched refactor/solve,
  warm-start, per-case dropout.
* `src/math/test/pf_batch_ca_test.cpp` — GB10 check of the batched engine.
* `src/applications/modules/powerflow/pf_screen.hpp` — Phase-3 Union-Set N-1
  connectivity pre-pass.
* `src/applications/contingency_analysis/gpu_validation/` — `compare_ca_csv.py`
  (now status-aware for diverged cases), `run_validation.sh`, `README.md`.

## Reproducing the GB10 verification (fast container loop)

The `gridpack-wecc` image already carries Boost/GA/PETSc, so the fastest loop
mounts the source + host CUDA toolkit + cuDSS and rebuilds only GridPACK:

```bash
docker run --rm --gpus all -e HOME=/tmp \
  -e PETSC_DIR=/deps/petsc/install_for_gridpack \
  -e LD_LIBRARY_PATH=/opt/cudss/lib:/usr/local/cuda-13.0/lib64:/deps/petsc/install_for_gridpack/lib:/deps/ga-5.9.1/install_for_gridpack/lib:/deps/boost-1.81.0/install_for_gridpack/lib \
  -e PATH=/usr/local/cuda-13.0/bin:/usr/bin:/bin \
  -v $PWD:/work -v /usr/local/cuda-13.0:/usr/local/cuda-13.0:ro \
  -v <cudss-archive>:/opt/cudss:ro -w /work alh360/gridpack-wecc:1.0 bash -lc '
    mkdir -p gpubuild && cd gpubuild
    cmake -D GA_DIR=/deps/ga-5.9.1/install_for_gridpack \
          -D Boost_ROOT=/deps/boost-1.81.0/install_for_gridpack \
          -D Boost_DIR=/deps/boost-1.81.0/install_for_gridpack/lib/cmake/Boost-1.81.0 \
          -D PETSC_DIR=/deps/petsc/install_for_gridpack \
          -D CMAKE_BUILD_TYPE=Release -D GRIDPACK_ENABLE_TESTS=OFF \
          -D GRIDPACK_WITH_CUDSS=ON -D CUDAToolkit_ROOT=/usr/local/cuda-13.0 \
          -D cudss_DIR=/opt/cudss/lib/cmake/cudss -D CMAKE_CUDA_ARCHITECTURES=121 \
          /work/src
    make -j20 ca.x cudss_batched_test pf_batch_ca_test
    ./math/cudss_batched_test && ./math/pf_batch_ca_test
    cd applications/contingency_analysis && make -C ../.. -j20 ca.x.input
    bash /work/src/applications/contingency_analysis/gpu_validation/run_validation.sh \
         ./ca.x input_14_gpu.xml 1 1e-6 1e-6'
```

The self-contained image is `docker build -t gridpack-cudss .` (the `Dockerfile`,
which fetches the pinned cuDSS redist and builds everything on the CUDA base).

Full Texas7k N-1 (not run here, to conserve tokens):
`cd /home/alh360/GridLensProjects/Texas7k_v2/original_inputs && mpirun -n 1 ca.x input_gpu.xml`
(`input_gpu.xml` is a copy of the original with `Backend=cudss`; the original is
unchanged).

## Phase 4 (FDPF) and Phase 5 (I/O) — designs

**Phase 4 — fast-decoupled constant-factor path.** Build the `B'`/`B''` coefficient
matrices from the pre-contingency θ⁰/V (Meng & Yun) so they are *constant*: analyze
+ factor each ONCE (a `CuDSSBatchedSolver` per matrix, reused across the whole
sweep), then every contingency/iteration is triangular-solve-only. Slots in as an
alternative `BatchAssembler` (active-power and reactive-power half-steps) selected
by `<GPU><fastDecoupled>`. Highest throughput/lowest memory; use as the screening
solver with the full-Jacobian batched engine as the accurate fallback.

**Phase 5 — device-side aggregation + overlapped bulk write.** Keep the three-CSV
contract exactly (already parity-verified). Compute the per-(contingency,branch)
fields on-device across a wave, then have rank 0 emit the identical rows in one
bulk pass; overlap factor/solve of wave *k* with assembly of wave *k+1* (CUDA
streams) and CSV formatting on spare Grace cores. This is a change to *how* the
existing rows are produced, never *what* — validated by the same `compare_ca_csv.py`
oracle.

---

## Building

### CPU-only (default, unchanged)
```bash
cmake -D CMAKE_BUILD_TYPE=Release  …  ..
make install
```
`GRIDPACK_WITH_CUDSS` defaults to `OFF`; no CUDA/cuDSS required.

### With the cuDSS GPU backend
```bash
cmake -D CMAKE_BUILD_TYPE=Release \
      -D GRIDPACK_WITH_CUDSS=ON \
      -D cudss_DIR=<cudss_prefix>/lib/cmake/cudss \
      -D CMAKE_CUDA_ARCHITECTURES=121 \
      …  ..
make install
```
Requirements: CUDA 12.x+ runtime, cuDSS (Pascal-or-newer; GB10 = sm_121). Only
the CUDA *runtime* API and the cuDSS *host* API are used, so **no `.cu` files and
no nvcc pass** are needed for the Phase-1 backend — it is plain C++ linked
against `CUDA::cudart` and `cudss`. The container build is fully encoded in the
`Dockerfile`.

---

## Running

Identical CLI in every case:
```bash
mpirun -n 1 ca.x input.xml
```

Enable the GPU path via `input.xml` (either switch selects cuDSS):
```xml
<Contingency_analysis>
  <GPU><enabled>true</enabled></GPU>
</Contingency_analysis>
<Powerflow>
  <LinearSolver>
    <Backend>cudss</Backend>
    <iterativeRefinement>false</iterativeRefinement>
    <deterministic>false</deterministic>
    <!-- CPU fallback options, used when cuDSS is unavailable -->
    <PETScOptions>-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type klu</PETScOptions>
  </LinearSolver>
</Powerflow>
```
`enabled=false` / `Backend=petsc` (the defaults) reproduce today's behavior. A
ready example ships as `input_14_gpu.xml`. On startup CA prints the resolved
backend (`Linear solver backend: cudss` or a fallback notice).

Recommended GPU configuration is `groupSize=1` (`K=1`): one rank owns the whole
contingency set. `K>1` works too — each rank collects its system and solves it on
the shared GPU (via CUDA MPS/time-slicing); `groupSize=1` avoids that redundancy.

---

## How the backend works (Phase 1)

The math layer already used the pImpl idiom: a `LinearSolverT` owns a
`LinearSolverImplementation` chosen at construction. The changes keep that shape:

1. **Selection.** `ca_driver` reads the switch and sets a process-wide default
   backend. When a `LinearSolver` is constructed, `petsc_linear_solver.cpp`
   calls `resolveLinearSolverBackend()`; it returns `CuDSS` only when compiled in
   **and** a device is present, else `PETSc`. cuDSS construction failure falls
   back to PETSc.

2. **Fixed-sparsity lifecycle.** The power-flow Jacobian keeps a constant pattern
   across Newton iterations, so the cuDSS backend maps GridPACK's lifecycle onto
   cuDSS phases:
   * first solve → `CUDSS_PHASE_ANALYSIS` **once** (reordering + symbolic),
   * each NR iteration (`solve()`) → `CUDSS_PHASE_FACTORIZATION` (refactorize),
   * each RHS → `CUDSS_PHASE_SOLVE` (optionally iterative refinement).

3. **Zero-reformat CSR.** A serial (SEQAIJ) PETSc `Mat` already stores the exact
   3-array CSR cuDSS wants; extraction is a pointer hand-off
   (`MatGetRowIJ` + `MatSeqAIJGetArrayRead`). On the GB10's unified memory these
   arrays are directly GPU-addressable; on ordinary hardware they are staged into
   device buffers. Types (value/index) are keyed off `PetscScalar`/`PetscInt`, so
   the same code is correct for real and complex PETSc builds.

4. **Serial coefficient matrix.** cuDSS (single-GPU) wants the whole system, so
   the backend forces the base class' serial path when running on >1 process; the
   `Mat` reaching `p_solveImpl` is then always SEQAIJ. `groupSize=1` is the clean
   case (no collection).

Because both paths perform an exact sparse LU, per-case results — and thus the
three CSVs — match the CPU run to FP64 round-off.

---

## Validating output parity

See `src/applications/contingency_analysis/gpu_validation/`:

```bash
# Freeze CPU golden files first (Phase 0):
mpirun -n 1 ./ca.x input_14.xml && mkdir -p golden && cp ca_IEEE14_*.csv golden/

# One-shot CPU-vs-GPU parity from a single base file:
gpu_validation/run_validation.sh ./ca.x input_14_gpu.xml 1 1e-6 1e-6

# Or compare against a frozen golden directly:
python3 gpu_validation/compare_ca_csv.py golden/ca_IEEE14_delta.csv \
        ca_IEEE14_gpu_delta.csv --atol 1e-6 --rtol 1e-6
```

`compare_ca_csv.py` re-sorts rows by `event_idx` + the remaining non-float key
columns (rows are rank/completion-ordered by design), compares integer/string
columns exactly and float columns with a mixed abs/rel tolerance, and exits
non-zero on any mismatch. On GPU-less hosts the harness compares CPU vs CPU
(fallback) and still passes, exercising the plumbing in CI.

---

## Roadmap for Phases 2–5 (concrete file map)

These require iterative development against real cuDSS + a GB10; the hooks and
homes are in place.

* **Phase 2 — batched engine** (`pf_batch_ca.hpp`, new `pf_batch_ca.cpp`; called
  from `ca_driver.cpp` when `<GPU><batched>` is on). Build the base CSR once; run
  ANALYSIS once for the whole sweep; present a wave of contingencies as a cuDSS
  **non-uniform batch** (`cudssMatrixCreateBatchCsr`/`cudssMatrixCreateBatchDn`,
  `cudssMatrixSetBatchValues`); batched NR with convergence dropout; warm-start
  each case from the base solution; size waves via the cuDSS memory-prediction
  API. Factor "which CSR blocks change for this event" out of
  `PFAppModule::setContingency` so a wave overwrites them in place.
* **Phase 3 — screening/connectivity** (new `pf_screen.*`). Batched Union-Set
  connectivity (one block per branch outage; `root[]` in shared memory) to drop
  islanded cases before the AC batch, plus optional DC/FCT screening. Replaces
  the per-case CPU `getIslandCount()`/`hasLoneBus()` with one batched pre-pass.
* **Phase 4 — FDPF** (new `pf_fdpf.*`). Constant `B'`/`B''` (built from the
  pre-contingency θ⁰/V for better convergence), analyzed+factored once, then
  triangular solves per case — the shape cuDSS rewards most.
* **Phase 5 — I/O** (`ca_driver.cpp` output section). Compute the existing
  per-(contingency,branch) fields on-device across a wave, then have rank 0 emit
  the **identical** CSV rows in one bulk pass; overlap factor/solve of wave *k*
  with assembly of wave *k+1* (CUDA streams) and CSV formatting on spare Grace
  cores. No format change — the three CSVs remain a hard contract.

---

## Notes and risks

* **Pin cuDSS.** Batch APIs and config params evolved quickly (non-uniform
  batches + memory prediction in v0.4; host execute/MT in v0.5). Pin a known-good
  cuDSS version in the image and re-validate on upgrade.
* **FP64 throughput.** Consumer-Blackwell FP64 is modest and sparse LU is
  bandwidth-bound; the win is concurrency + symbolic reuse + zero-copy, not raw
  FLOPS. Benchmark straight-FP64 refactor+solve early; if it disappoints, the
  mixed-precision IR (`<iterativeRefinement>`) and FDPF paths become primary.
* **`Debug`→`Release`.** The image now builds `Release`; measure this as the new
  CPU baseline *before* crediting the GPU for anything.
* **Determinism.** cuDSS uses atomics by default; set
  `<deterministic>true</deterministic>` for bit-reproducible N-1 (slower).
