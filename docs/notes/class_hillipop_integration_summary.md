# Phase 1: Study Resources

## 1.1 JAX reference notebook summary
1. **CLASS parameters:** The notebook sets `h = 0.6737`, `omega_b = 0.02237`, `omega_cdm = 0.1200`, `A_s = 2.1e-9`, `n_s = 0.9649`, `tau_reio = 0.0544`, `N_ur = 2.0328`, `m_ncdm = 0.06`, `l_max_scalars = 2508`, and `lensing = 'yes'`.
2. **Output spectra:** It computes lensed CMB power spectra (due to `lensing='yes'`), specifically using `tt`, `te`, and `ee`.
3. **ell range & post-processing:** Evaluated up to `l_max_scalars = 2508`. Python's `classy.lensed_cl` returns dimensionless $C_\ell$ directly.
4. **Nuisance parameters:** Passed as a dictionary directly to `compute_loglike`. They include `A_planck=1.0`, foreground amplitudes (e.g., `AdustT=1.0`, `AdustP=1.0`), and calibration parameters.
5. **Chi-square/LogL computation:** The log-likelihood is computed via `camspec.compute_loglike(ClTT, ClTE, ClEE, nuisance_params)`.
6. **Reference cosmology:** The computed fiducial JAX log-likelihood is `-18693.669380`.

## 1.2 CLASS.jl wrapper summary
1. **Constructing problem:** Use `prob = CLASSProblem(Dict_of_params...)` with standard `.ini` string keys like `"H0"`, `"omega_b"`, `"output"`, `"lensing"`, `"l_max_scalars"`.
2. **Running CLASS:** Executed via `sol = solve(prob)`. It searches for the `class` executable in `$PATH` by default.
3. **Retrieving output:** `sol["lCl"]` returns a DataFrame for lensed spectra. The columns are named `:l`, `:TT`, `:EE`, `:TE`, etc. The values are the standard CLASS `.dat` output, which is dimensionless $D_\ell = \frac{\ell(\ell+1)}{2\pi} C_\ell$.
4. **Installation:** Can be installed via `Pkg.add("CLASS")`. It does not vendor CLASS, requiring the user to have `class` compiled and in their system PATH.
5. **Limitations:** Requires external system dependencies. Output formatting is strictly whatever the C CLASS writes to the `.dat` file.

## 1.3 Hillipop.jl summary
1. **Signature:** `compute_loglike(ClTT, ClTE, ClEE, pars, h)` where `h` is a `HillipopData` struct and `pars` is `HillipopNuisance` or a Dict.
2. **ell range & spectra:** Uses `lmax = 2500`. It processes TT, TE, and EE spectra.
3. **Normalisation convention:** Expects `ClTT`, `ClTE`, `ClEE` to be vectors of $C_\ell$ in units of $K^2$. The vectors must start at $\ell=2$ (i.e. index 1 is $\ell=2$).
4. **Nuisance parameters:** Added as a `HillipopNuisance` struct or Dict to the `compute_loglike` function. They modify the theory Cls directly via foreground models or calibrations before computing the residuals.
5. **Data files:** Loaded via `load_hillipop()`, which uses `Artifacts` or similar mechanisms to get the necessary covariance and bandpower files.

---

# Phase 2: Identify and Resolve Convention Mismatches

**Mismatch 1: Normalisation of Cls**
- *CLASS.jl produces:* Dimensionless $D_\ell = \frac{\ell(\ell+1)}{2\pi} C_\ell$.
- *Hillipop.jl expects:* $C_\ell$ in $K^2$.
- *Resolution:* Multiply the CLASS columns by $\frac{2\pi}{\ell(\ell+1)} \times T_{\text{CMB}}^2$, where $T_{\text{CMB}} = 2.7255$ K.

**Mismatch 2: ℓ indexing**
- *CLASS.jl produces:* A DataFrame where rows correspond to ℓ values starting from $\ell=2$ (typically, as output by CLASS).
- *Hillipop.jl expects:* Flat Julia Vectors starting exactly at $\ell=2$.
- *Resolution:* Extract `df.TT[df.l .>= 2]`, `df.TE[df.l .>= 2]`, and `df.EE[df.l .>= 2]` ensuring we only grab the values from $\ell=2$ up to `h.lmax`.

**Mismatch 3: Column names and Types**
- *CLASS.jl produces:* Column name for multipole is `:l` (lowercase L).
- *Hillipop.jl expects:* No specific struct, just standard Arrays.
- *Resolution:* Extract the columns using `df.l`, `df.TT`, etc., and cast to `Vector{Float64}`.
