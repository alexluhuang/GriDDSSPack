# GPU contingency-analysis validation

Parity tooling for the NVIDIA cuDSS GPU path added to the contingency-analysis
application.  The acceptance criterion is simple: **the GPU run must reproduce
the CPU (PETSc/direct-LU) run's three CSVs to FP64 round-off.**  Both paths do
an exact sparse LU factorization, so agreement should be tight.

## Files

| File | Purpose |
|------|---------|
| `compare_ca_csv.py` | Compare two CA CSVs after re-sorting by their key columns (rows are emitted rank/completion-ordered). Integer/string columns compared exactly; float columns with `--atol`/`--rtol`. Exit 0 on parity. |
| `run_validation.sh` | From one base `input.xml`, derive a CPU run and a GPU run that differ only in the backend, run both through `ca.x`, and compare `_delta`/`_flat`/`_buses`/`_convergence`. |

## Freezing golden files (Phase 0, step 4)

Before any GPU work, capture the CPU KLU outputs as golden files for the suite
(case14, case118, case1354pegase, case2869pegase). Example:

```bash
mpirun -n 1 ./ca.x input_14.xml           # writes ca_IEEE14_*.csv
mkdir -p golden && cp ca_IEEE14_*.csv golden/
```

## Running the parity check

```bash
# From the ca.x build/run directory:
../../.../gpu_validation/run_validation.sh ./ca.x input_14_gpu.xml 1 1e-6 1e-6
```

`run_validation.sh` toggles `<Backend>` and `<GPU><enabled>` to produce a CPU
(golden) and a GPU (candidate) run from the same base file, then compares. On a
host without a GPU (or a binary built without `GRIDPACK_WITH_CUDSS`), the GPU run
falls back to the CPU path and the check still passes (CPU vs CPU) — handy in CI.

Direct comparison against a frozen golden file:

```bash
python3 compare_ca_csv.py golden/ca_IEEE14_delta.csv ca_IEEE14_gpu_delta.csv \
        --atol 1e-6 --rtol 1e-6
```

## Notes

* Row order is rank/completion-dependent by design (see the CA `README.md`); the
  comparator sorts on `event_idx` plus the remaining non-float key columns, so
  ordering never causes a false mismatch.
* For **regulatory / bit-reproducible** runs, enable cuDSS deterministic mode
  (`<LinearSolver><deterministic>true</deterministic>`) and compare with a very
  tight tolerance.
* Complement CSV parity with the field-level checks the literature uses
  (max ΔV / Δθ vs MATPOWER), and keep an eye on `_convergence.csv` for iteration
  count / accuracy regressions.
