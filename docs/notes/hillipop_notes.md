# Hillipop.jl Summary

1. **Top-level Signature:**
   - `compute_loglike(ClTT, ClTE, ClEE, pars, h; modes=("TT", "EE", "TE"))`
   - Returns scalar `Float64` log-likelihood value.

2. **Ell Range and Spectra:**
   - `ell_max` defaults to 2500. `ClTT`, `ClTE`, `ClEE` must be `Vector{Float64}` of length `ell_max - 1`.

3. **Input Cl Normalization:**
   - `ClTT`, `ClTE`, `ClEE` must be $C_\ell$ in $K^2$ (not $\mu K^2$), starting at $\ell=2$.
   - Internally, Hillipop uses `_cl_to_dl` which computes: `dl[l+1] = Cl[l-1] * 1e12 * l * (l + 1) / (2π)`. This converts $C_\ell [K^2]$ at $\ell \ge 2$ to $D_\ell [\mu K^2]$.

4. **Nuisance Parameters:**
   - Passed as `HillipopNuisance` struct containing nested `HillipopCalibration`, `HillipopDust`, etc.
   - They enter the likelihood by calculating foregrounds and calibrating the theory + foregrounds, then subtracting from observed data.

5. **Data Files:**
   - Pre-loaded into `HillipopData` via `load_hillipop()`. Needs artifact data (binning fits, templates, cross spectra, inverse covariance).
