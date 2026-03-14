# Optimisation Summary

## End-to-End Speedup
The full likelihood evaluation (`compute_loglike`) experienced a massive performance improvement:
- **Baseline:** 13.98 ms median time, 105,471 allocations (11.52 MiB)
- **Final:** 1.79 ms median time, 536 allocations (6.01 MiB)
- **Total Speedup:** ~7.8x faster, with ~99.5% reduction in allocations.

## Top 3 Changes by Impact
1. **Sparse Matrix conversion for binning (`compute_chi2`)**
   The most expensive function was `compute_chi2`, taking ~10 ms because it involved a dense matrix-vector product with a 99.9% sparse binning matrix. Changing `binning_matrix` to a `SparseMatrixCSC` during `load_hillipop` dropped the time to **293 μs** (a ~34x speedup).
2. **Type stability fixes (`build_residual_vector`)**
   By swapping the `dlth` parameter container from a `Dict{String, Vector{Float64}}` to a `NamedTuple` and removing dynamic dispatch caused by keyword arguments in `xspectra_to_xfreq`, the number of allocations dropped from over 105,000 down to less than 1,000. Runtime for building the residual vector halved (3.17 ms to 1.37 ms).
3. **Array mutation and `@views` (`compute_foreground_dl` & `select_spectra`)**
   Applying `@view` directly inside slicing and providing a pre-allocated vector `dlmodel_fg` to `compute_foreground_dl!` reduced inner loop and dynamic memory pressure inside `compute_residuals`, providing noticeable reduction in GC sweeps. Furthermore, replacing string-concatenation symbols (`Symbol("cal$(map)")`) with a static `_get_subpixel`/`_get_cal` dispatcher completely removed all string-based allocations in the hot path.

## Known Bottlenecks and Future Directions
- **`CMBForegrounds.jl` Allocations:** `dust_model_template_power`, `tsz_cross_power`, and other CMB foreground evaluation methods currently return newly allocated arrays. Extending the `CMBForegrounds` package API to support in-place mutating versions (e.g. `dust_model_template_power!`) would eliminate the remaining ~5 MB of allocations inside the likelihood execution loop.
- **Multithreading:** The loop computing residuals over the 15 combinations of map pairs inside `compute_residuals` could easily be multithreaded or evaluated in parallel using `Tullio.jl` to distribute the heavy foreground computations, though at ~1.7 ms, the evaluation is already extremely competitive for MCMC sampling.