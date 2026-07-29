# GriDSSPack GPU versus stock GridPACK CPU

## Row coverage

| Dataset | GPU rows | CPU rows | Matched | GPU only | CPU only |
|---|---:|---:|---:|---:|---:|
| Flat branch results | 1,004,189,036 | 938,077,050 | 938,077,050 | 66,111,986 | 0 |
| Convergence | 37,063 | 37,063 | 37,063 | 0 | 0 |

Flat rows present only in the GPU output span 31,156 contingencies; flat rows present only in the CPU output span 0 contingencies.
Those unmatched rows represent 32,231 GPU-only branch keys and 0 CPU-only branch keys.
Matched rows with different violation flags: 12,835 (0.00136822%).

## Numerical differences

MAPD is the mean absolute percent difference relative to the stock CPU value. CPU reference values at or below the configured zero threshold are excluded from MAPD and counted separately.

| Column | Unit | CPU mean | GPU mean | Mean absolute difference | MAPD | Symmetric MAPD | Max absolute difference |
|---|---|---:|---:|---:|---:|---:|---:|
| `p_from_mw` | MW | -13.0044 | -13.0044 | 1.04619e-05 | 0.00020081% | 0.000590213% | 1.9211 |
| `q_from_mvar` | MVAr | -3.97008 | -3.97008 | 6.237e-05 | 0.00753393% | 0.00176294% | 39.233 |
| `mva_from` | MVA | 73.5822 | 73.5822 | 2.51342e-05 | 8.21794e-05% | 0.000524961% | 19.046 |
| `rate_mva` | MVA | 463.308 | 463.308 | 0 | 0% | 0% | 0 |
| `loading_percent` | percent | 18.5506 | 18.5506 | 9.4344e-06 | 8.25531e-05% | 0.000112131% | 14.93 |
| `v_from_pu` | p.u. | 1.03534 | 1.03534 | 8.88037e-08 | 8.58419e-06% | 8.58431e-06% | 0.022869 |
| `v_to_pu` | p.u. | 1.03465 | 1.03465 | 8.74789e-08 | 8.46135e-06% | 8.46102e-06% | 0.022869 |
| `ang_from_deg` | degrees | -63.5155 | -63.5155 | 2.64082e-05 | 9.14292e-05% | 9.20401e-05% | 0.2269 |
| `ang_to_deg` | degrees | -63.3162 | -63.3162 | 2.6397e-05 | 0.000104158% | 0.000104411% | 0.2269 |

Bus-angle differences use the shortest circular displacement in degrees for absolute-error statistics.

## Convergence

| Characteristic | GPU | CPU |
|---|---:|---:|
| Converged cases | 37,016 (99.8732%) | 37,014 (99.8678%) |
| Mean iterations | 3.93236 | 2.02706 |
| Median iterations | 3 | 2 |
| 95th-percentile iterations | 10 | 3 |

Convergence outcome agreement: 99.9784%
Convergence-column numerical differences are calculated over the 31,154 events with `OK` status in both outputs; all-case metrics remain available in the JSON report.

| Convergence column | Unit | CPU mean | GPU mean | Mean absolute difference | MAPD |
|---|---|---:|---:|---:|---:|
| `iterations` | iterations | 2.02388 | 4.13748 | 2.1136 | 101.09% |
| `final_tolerance` | p.u. | 1.01908e-05 | 2.47777e-05 | 1.73799e-05 | 30622.2% |
| `max_p_mismatch` | p.u. | 0.000875008 | 0.000309286 | 0.000565722 | 55.4206% |
| `max_q_mismatch` | p.u. | 0.000909203 | 0.000386358 | 0.000522845 | 54.5778% |

## Performance

| Characteristic | GPU | CPU | Comparison |
|---|---:|---:|---:|
| Maximum total application time | 1314.21 s | 3269.97 s | 2.48815x speedup |
| Mean tasks per rank | 1853.15 | 1853.15 | — |
| Task-count coefficient of variation | 2.05335% | 14.1783% | — |

See `comparison_report.json`, `numerical_differences.csv`, and `timing_comparison.csv` for complete statistics. `flat_coverage_by_contingency.csv` identifies every event associated with unmatched flat-output rows, and `flat_coverage_by_branch.csv` identifies the branch keys.
