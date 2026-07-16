# GridPACK CSR solver benchmark

This standalone tool compares SuiteSparse KLU with scalar NVIDIA cuDSS for
exported real, double-precision, unsymmetric sparse systems. It does not link
GridPACK and is not included by `src/CMakeLists.txt`, so the normal GridPACK
CPU build has no CUDA or cuDSS dependency.

The benchmark reports three solver lifecycles:

- `cold_analysis_factor_solve`: new symbolic and numeric state for every
  measured solve.
- `warm_factor_solve`: symbolic analysis once, then factorization and solve
  for every repetition.
- `repeated_rhs_solve`: analysis and factorization once, then repeated solves.

KLU consumes CSC, so its one-time CSR-to-CSC conversion is reported separately.
Common CUDA/cuDSS initialization is reported once as `CUDSS_RUNTIME setup_ms`;
each cuDSS backend then reports its mode-specific allocation, upload, and
matrix-wrapper setup as `BACKEND setup_ms`. Every scenario reports
right-hand-side refresh and solver-state lifecycle independently from the
analysis, factorization, and solve phases. Device-to-host solution transfer is
reported as `solution_download` and is included in `end_to_end` but not
`solver_total`; hybrid solutions are already host-resident and therefore
report zero for that phase. All cuDSS phase timings synchronize the configured
CUDA stream.

## Interchange format

The production interchange is the exporter-compatible `.gpcsr` binary v1
format. Every integer and floating-point field is serialized explicitly in
little-endian byte order; native C/C++ structs are never written directly.
The 40-byte header is:

| Offset | Type | Value |
| ---: | --- | --- |
| 0 | 8 bytes | ASCII `GPCSR001` |
| 8 | `uint32` | version `1` |
| 12 | `uint32` | header size `40` |
| 16 | `uint32` | flags `0x00000007` |
| 20 | `uint32` | rows |
| 24 | `uint32` | columns |
| 28 | `uint32` | nonzeros |
| 32 | `uint32` | RHS count |
| 36 | `uint32` | reserved, must be zero |

Flag bit 0 specifies little-endian encoding, bit 1 specifies zero-based CSR
indices, and bit 2 specifies IEEE-754 binary64 matrix and RHS values. No other
flag bits are permitted in v1. The payload follows immediately:

1. `uint32 row_offsets[rows + 1]`
2. `uint32 column_indices[nonzeros]`
3. `binary64 values[nonzeros]`
4. `binary64 rhs[rows * rhs_count]`

RHS vectors are consecutive and column-major. The benchmark widens the
serialized 32-bit indices to checked 64-bit indices before passing them to
cuDSS and the KLU `klu_l_*` API.

The strict reader requires version 1's exact header, flags, and file length; a
square nonempty matrix; a positive RHS count; zero-based monotonic row offsets;
strictly increasing, in-range column indices within each row; and finite
matrix/RHS values. Metadata describing the case, contingency, nonlinear
iteration, and convergence state belongs in the exporter's companion manifest,
not the matrix file.

The v1 exporter currently writes one RHS per system, and the scalar cuDSS
backend requires `rhs_count == 1`. KLU can read and solve diagnostic files with
multiple column-major RHS vectors.

For small hand-written tests, the reader also accepts a UTF-8,
whitespace-delimited diagnostic format with fields in exactly this order:

```text
GRIDPACK_CSR_SYSTEM_V1
name "descriptive name"
contingency "base case or contingency identifier"
newton_iteration 0
controller_iteration 0
convergence_status "converged, diverged, or not_run"
state "PV/PQ, Q-limit, and controller state description"
nrows 4
ncols 4
nnz 11
index_base 0
rhs_count 1
row_offsets 5
0 2 5 8 11
column_indices 11
0 1 0 1 2 1 2 3 0 2 3
values 11
10 2 3 9 4 1 7 5 2 3 8
rhs 4
14 33 43 43
END_GRIDPACK_CSR_SYSTEM_V1
```

Quoted strings use C++ `std::quoted` escaping. Numeric arrays may span any
number of lines. Each section includes its serialized element count. RHS
vectors are stored consecutively in column-major order, so `rhs` contains
`nrows * rhs_count` values.

## Build

Both backends are enabled by default:

```bash
export PKG_CONFIG_PATH=/path/to/suitesparse/lib/pkgconfig
cmake -S tools/csr_solver_benchmark \
      -B build/csr_solver_benchmark \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/path/to/cudss \
      -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build/csr_solver_benchmark --parallel
```

`CMAKE_PREFIX_PATH` must name the cuDSS archive or installation containing
`lib/cmake/cudss/cudss-config.cmake`. KLU is discovered through its `KLU.pc`
file. Either backend can be built independently:

```bash
cmake -S tools/csr_solver_benchmark -B build/klu-only \
      -DGRIDPACK_CSR_BENCHMARK_ENABLE_CUDSS=OFF

cmake -S tools/csr_solver_benchmark -B build/cudss-only \
      -DGRIDPACK_CSR_BENCHMARK_ENABLE_KLU=OFF \
      -DCMAKE_PREFIX_PATH=/path/to/cudss \
      -DCUDAToolkit_ROOT=/usr/local/cuda
```

## Run

```bash
build/csr_solver_benchmark/gridpack_csr_solver_benchmark \
  --iterations 20 \
  --warmups 2 \
  --scenario all \
  --cudss-mode both \
  exported_system.gpcsr
```

Use `--scenario cold`, `warm`, or `repeated` to isolate one lifecycle, for
example when running an external pool of independent benchmark processes.

Use `tools/csr_solver_benchmark/examples/unsymmetric_4x4.csr` in place of the
`.gpcsr` file for the included diagnostic example.

The default scaled residual is the maximum over all RHS vectors of

```text
||A*x - b||_2 / (||A||_F * ||x||_2 + ||b||_2).
```

The process exits with status 3 if any residual exceeds `--residual-tol`, or
if a cuDSS solution differs from the corresponding KLU solution by more than
`--agreement-tol` in both relative L2 and maximum absolute error. It exits
with status 2 for malformed input, unavailable requested backends, or checked
KLU, CUDA, or cuDSS API failures.

This scalar harness is a correctness and lifecycle-reuse benchmark. It does
not measure batched cuDSS throughput or full contingency-analysis speedup.
