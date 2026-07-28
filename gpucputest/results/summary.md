# GriDSSPack GPU versus stock GridPACK CPU

## Row coverage

| Dataset | GPU rows | CPU rows | Matched | GPU only | CPU only |
|---|---:|---:|---:|---:|---:|
| Flat branch results | 78,969,600 | 74,701,440 | 74,701,440 | 4,268,160 | 0 |
| Convergence | 8,891 | 8,891 | 8,891 | 0 | 0 |

Flat rows present only in the GPU output span 8,640 contingencies; flat rows present only in the CPU output span 0 contingencies.
Those unmatched rows represent 494 GPU-only branch keys and 0 CPU-only branch keys.
Matched rows with different violation flags: 3,361 (0.00449924%).

## Numerical differences

MAPD is the mean absolute percent difference relative to the stock CPU value. CPU reference values at or below the configured zero threshold are excluded from MAPD and counted separately.

| Column | Unit | CPU mean | GPU mean | Mean absolute difference | MAPD | Symmetric MAPD | Max absolute difference |
|---|---|---:|---:|---:|---:|---:|---:|
| `p_from_mw` | MW | -18.0584 | -18.0584 | 6.97378e-05 | 0.0016889% | 0.00211363% | 0.8464 |
| `q_from_mvar` | MVAr | 0.112751 | 0.112737 | 0.000494296 | 0.0204553% | 0.0150933% | 40.5746 |
| `mva_from` | MVA | 73.9533 | 73.9532 | 0.00017064 | 0.000682171% | 0.00156242% | 10.2524 |
| `rate_mva` | MVA | 274.834 | 274.834 | 0 | 0% | 0% | 0 |
| `loading_percent` | percent | 24.0394 | 24.0393 | 6.93736e-05 | 0.000671894% | 0.000763961% | 3.87 |
| `v_from_pu` | p.u. | 1.02765 | 1.02765 | 6.17294e-07 | 6.02526e-05% | 6.02992e-05% | 0.014437 |
| `v_to_pu` | p.u. | 1.02791 | 1.02791 | 6.12825e-07 | 5.97921e-05% | 5.98342e-05% | 0.01549 |
| `ang_from_deg` | degrees | -6.32231 | -6.32234 | 6.04562e-05 | 0.00279207% | 0.00248405% | 0.1718 |
| `ang_to_deg` | degrees | -6.02095 | -6.02098 | 6.02946e-05 | 0.00233896% | 0.00215364% | 0.1828 |

Bus-angle differences use the shortest circular displacement in degrees for absolute-error statistics.

## Convergence

| Characteristic | GPU | CPU |
|---|---:|---:|
| Converged cases | 8,887 (99.955%) | 8,887 (99.955%) |
| Mean iterations | 4.70352 | 1.9342 |
| Median iterations | 4 | 2 |
| 95th-percentile iterations | 11 | 3 |

Convergence outcome agreement: 100%
Convergence-column numerical differences are calculated over the 8,639 events with `OK` status in both outputs; all-case metrics remain available in the JSON report.

| Convergence column | Unit | CPU mean | GPU mean | Mean absolute difference | MAPD |
|---|---|---:|---:|---:|---:|
| `iterations` | iterations | 1.93298 | 4.77115 | 2.83818 | 142.879% |
| `final_tolerance` | p.u. | 9.46641e-06 | 3.12526e-05 | 2.61854e-05 | 114187% |
| `max_p_mismatch` | p.u. | 0.000683806 | 6.71721e-05 | 0.000616634 | 79.021% |
| `max_q_mismatch` | p.u. | 0.000726936 | 0.000178805 | 0.0005482 | 71.5895% |

## Performance

| Characteristic | GPU | CPU | Comparison |
|---|---:|---:|---:|
| Maximum total application time | 84.5626 s | 214.136 s | 2.53227x speedup |
| Mean tasks per rank | 444.55 | 444.55 | — |
| Task-count coefficient of variation | 3.88414% | 17.2781% | — |

See `comparison_report.json`, `numerical_differences.csv`, and `timing_comparison.csv` for complete statistics. `flat_coverage_by_contingency.csv` identifies every event associated with unmatched flat-output rows, and `flat_coverage_by_branch.csv` identifies the branch keys.
