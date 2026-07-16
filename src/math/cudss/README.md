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
`GRIDPACK_CUDSS_RESIDUAL_TOLERANCE`, and `GRIDPACK_CUDSS_DIAGNOSTICS`.

The scalar validation backend is deliberately narrow: GridPACK `RealType`
must be `double`, PETSc indices must be 32-bit, PETSc must use real scalars,
the matrix and vectors must be exact sequential `MATSEQAIJ`/`VECSEQ` objects
on congruent communicators, and `MPI_COMM_WORLD` must contain only rank zero.
Complex, distributed, and otherwise ineligible systems remain on PETSc in
non-strict mode.

The backend caches cuDSS analysis state by exact CSR row-offset and
column-index identity. It performs analysis and factorization on a new pattern,
refactorization when only values change, and solve-only reuse when both pattern
and values are unchanged. `CUDSSDiagnostics=true` reports the selected device
owner and phase/cache/fallback counters when the solver is destroyed.
