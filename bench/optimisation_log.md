# Prioritised Optimization Plan

1. **Target:** `build_residual_vector` (and type stability of likelihood)
   - **Cost:** 3.17 ms, 105,465 allocs
   - **Root cause hypothesis:** `dlth = Dict(...)` causes `dlth[mode]` to be type-unstable, forcing dynamic dispatch on every array access in `compute_residuals` and `build_residual_vector` (`Rl_te[i] = ...`).
   - **Proposed fix:** Change `dlth` to a `NamedTuple`.
   - **Expected gain:** ~1-2 ms, >100,000 allocations removed.
   - **Risk:** Low.

2. **Target:** `compute_foreground_dl`
   - **Cost:** 24.4 μs, 33 allocs (138 KiB)
   - **Root cause hypothesis:** `result = zeros(...)` allocates memory. `ll2pi = _ll2pi(lmax)` allocates memory but is unused! `key = Symbol("Asbpx_$(f1)x$(f2)")` allocates strings and symbols on each call.
   - **Proposed fix:** Remove `ll2pi`. Pass `result` buffer as an argument to mutate in place. Replace string-based symbol lookup with an explicit `if/else` or dictionary for sub-pixel effects.
   - **Expected gain:** 0 allocations per call.
   - **Risk:** Low.

3. **Target:** `compute_residuals`
   - **Cost:** 459 μs, 614 allocs (2.62 MiB)
   - **Root cause hypothesis:** `pairs = Tuple{...}[]` with `push!` allocates. `dlmodel_xs = dlth .+ dlmodel_fgs[xs]` allocates intermediate vectors. `_cal_factor` allocates via `Symbol("cal$(map1)")`.
   - **Proposed fix:** Fused in-place broadcast: `R[xs, l] = dldata_mode[xs, l] - cal * (dlth[l] + dlmodel_fgs[xs][l])`. Change `_cal_factor` to use static mapping or pass a function that avoids `Symbol`.
   - **Expected gain:** Significant allocation reduction.
   - **Risk:** Medium.

# Optimisation Log

| Function | Baseline median | Optimised median | Speedup | Δ allocations | Change summary |
|----------|-----------------|------------------|---------|---------------|----------------|
| `build_residual_vector` | 3.17 ms | 1.37 ms | 2.31x | -104,739 | Unrolled `for` loop over modes, removed `Dict` `dlth` using `NamedTuple`, fixed keyword argument dynamic dispatch in `xspectra_to_xfreq`. |
| `compute_residuals` | 459 μs | 452 μs | 1.01x | -333 | Passed pre-allocated buffer to `compute_foreground_dl!`, removed dynamic `push!` in loop, fused broadcast loop. |
| `compute_foreground_dl` | 24.4 μs | 22.0 μs | 1.1x | -15 | Mutate pre-allocated buffer `result` to remove array allocation, removed `ll2pi` dead allocation, replaced `Symbol` subpixel loop with explicit static `_get_subpixel` function. |
| `select_spectra` | 12.8 μs | 7.2 μs | 1.8x | -4 | Used `@views` to avoid intermediate array allocations during slicing. |
| `compute_chi2` | 9.89 ms | 293 μs | 33.7x | 0 | Converted dense `binning_matrix` to `SparseMatrixCSC` during loading, heavily accelerating matrix-vector product. |
