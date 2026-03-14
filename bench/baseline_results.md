# Baseline Results

| Function | Median time | Min time | Allocations | Bytes | Type stable? |
|----------|-------------|----------|-------------|-------|--------------|
| `compute_loglike` | 13.98 ms | 11.93 ms | 105,471 | 11.52 MiB | Yes |
| `compute_chi2` | 9.89 ms | 8.30 ms | 6 | 24.77 KiB | Yes |
| `build_residual_vector` | 3.17 ms | 2.81 ms | 105,465 | 11.50 MiB | Yes (mostly) |
| `compute_residuals` | 459 μs | 393 μs | 614 | 2.62 MiB | Yes |
| `compute_fg_model` | 358 μs | 303 μs | 497 | 2.02 MiB | Yes |
| `xspectra_to_xfreq` | 93.9 μs | 80.1 μs | 9 | 352.0 KiB | Yes |
| `compute_foreground_dl` | 24.4 μs | 20.2 μs | 33 | 137.9 KiB | Yes |
| `select_spectra` | 12.8 μs | 9.28 μs | 24 | 167.2 KiB | Yes |
| `_cl_to_dl` | 3.37 μs | 2.74 μs | 3 | 19.6 KiB | Yes |

*Note: `compute_chi2` is taking the majority of the time. `build_residual_vector` is responsible for almost all allocations (105,465).*