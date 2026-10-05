# GPU versus CPU result comparison

`compare_results.py` compares the full GriDSSPack GPU result set with the stock
GridPACK CPU result set. It reports:

- keyed row coverage, including GPU-only and CPU-only branch-result rows;
- mean absolute percent difference, symmetric percent difference, mean/maximum
  absolute difference, RMSE, bias, and tolerance agreement for every solution
  column;
- shortest-path angular errors for `ang_from_deg` and `ang_to_deg`;
- convergence-rate agreement, iteration distributions, mismatch/tolerance
  differences, and status transitions;
- application and legacy-category timing speedups plus MPI task-load balance.

`timing_comparison.csv` marks each row as comparable or diagnostic. Whole-run
`Total Application` is comparable with stock logs.

New GPU logs declare
`[profiling] schema=legacy-v1 gpu_legacy_mapping=1`. That capability
marker enables comparisons for legacy categories into which the optimized path
explicitly charges equivalent work:

- `Powerflow: Total Application`
- `Powerflow: Network Parser`
- `Powerflow: Partition`
- `Powerflow: Factory Load`
- `Powerflow: Factory Set Components`
- `Powerflow: Factory Set Exchange`
- `Powerflow: Bus Update`
- `Powerflow: Factory Operations`
- `Powerflow: Create Mappers`
- `Powerflow: Map to Matrix`
- `Powerflow: Map to Vector`
- `Vector Map: New Vector`
- `Vector Map: Load Bus Data`
- `loadBusData: Add Vector Elements`
- `loadBusData: Fill Buffer`
- `loadBusData: Add Elements`
- `Vector Map: Set Vector`
- `Powerflow: Create Linear Solver`
- `Powerflow: Solve Linear Equation`
- `Powerflow: Map to Bus`
- `mapToBus: get Data`
- `mapToBus: set Data`
- `Contingency: Total Application`
- `Contingency: Write Results`

The stock CPU log does not need a schema marker. Old `ca-v2` GPU logs must not
receive legacy-category speedups because their optimized work was not charged
into those labels.

The comparison allowlist uses exact timer names rather than broad prefixes. The
sole new optimized-path category, `Contingency: Batch Preparation`, has no stock
counterpart and remains diagnostic. Any unrecognized timer also retains its raw
duration without a speedup. Nested legacy rows overlap their parents and must
not be summed; compare the same row across implementations.

The direct GPU assembler charges equivalent work to the parent `Map to Matrix`,
`Map to Vector`, and `Map to Bus` categories; it does not fabricate calls to
the lower-level `Vector Map:*`, `loadBusData:*`, or `mapToBus:*` helpers.
Reductions in those child rows therefore show mapper work that the optimized
path avoided, not the speed of an equivalent GPU kernel.

The JSON report lists legacy categories found in both logs and categories
present on only one side.

The 8–9 GB flat CSVs are never merged as one in-memory table. dask-cuDF streams
each CSV into event-bucketed Parquet partitions, and cuDF compares one bounded
event bucket at a time on the GPU.

## Run

The wrapper builds the pinned multi-architecture RAPIDS 26.06 CUDA 13 image and
runs the full comparison:

```bash
cd /home/alh360/Documents/GriDSSPack/gpucputest
./run_comparison.sh
```

Pass another result root as the first argument if needed. It must contain the
GPU files directly and the CPU files in `cpu_results`:

```bash
./run_comparison.sh /path/to/verification_gpu_7k
```

Reports are written to `gpucputest/results`:

```text
summary.md
comparison_report.json
numerical_differences.csv
timing_comparison.csv
convergence_status_transitions.csv
flat_coverage_by_contingency.csv
flat_coverage_by_branch.csv
```

The flat-file key is event index plus from bus, to bus, and circuit ID.
Contingency names are compared independently. Duplicate keys stop the comparison
instead of silently producing a many-to-many join. The coverage CSV identifies
every contingency associated with rows that exist on only one side and attaches
its GPU and CPU convergence outcomes. The branch-coverage CSV identifies
systematic differences in which network elements each implementation emits.

## Direct invocation

Inside an environment containing cuDF, dask-cuDF, and dask-cuda:

```bash
python compare_results.py \
  --gpu-dir /data \
  --cpu-dir /data/cpu_results \
  --output-dir /output \
  --temp-dir /scratch
```

Useful controls:

- `--block-size-mib 256` bounds each CSV parsing task.
- `--bucket-events 64` bounds each cuDF merge to 64 contingency indices.
- `--device-memory-limit default` controls Dask-CUDA spilling.
- `--enable-cudf-spill` enables device-to-host spilling on discrete-memory
  GPUs. Do not use it on the GB10 unified-memory SoC.
- `--scheduler synchronous` avoids a distributed worker for small test data.
- `--keep-intermediate` retains staged Parquet data for debugging or reuse.

MAPD excludes CPU reference values whose magnitude is at or below
`--zero-threshold` (default `1e-9`) and reports the excluded count. Symmetric
MAPD is also reported because it remains bounded when one result is near zero.
For angle columns, absolute errors use the shortest circular difference;
percentage statistics remain relative to the CPU angle and should be interpreted
together with the mean absolute difference in degrees.
