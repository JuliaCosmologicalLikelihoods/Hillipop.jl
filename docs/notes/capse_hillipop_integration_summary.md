# Phase 1: Study Resources

## 1.1 Capse.jl repository summary
a) **Public API:** Pretrained emulators are accessed via an eagerly loaded dictionary `Capse.trained_emulators`, specifically under keys like `"CAMB_LCDM"`. Available spectra include `"TT"`, `"TE"`, `"EE"`, and `"PP"`. Emulators are evaluated using `Capse.get_Cℓ(x, emu)`. Metadata is accessible via `Capse.get_emulator_description(emu)`.
b) **Parameter vector:** For the CAMB ΛCDM emulator, the expected parameter vector `x` is ordered as: `[ln10As, ns, H0, ωb, ωc, τ]`. Notice that it uses $H_0$ directly (not $h$), and physical densities $\omega_b, \omega_c$ instead of $\Omega$.
c) **Output format & units:** The emulator outputs $D_\ell$ in units of $\mu\text{K}^2$, matching CAMB's default output convention. The output vector has 4999 elements corresponding to $\ell = 2$ to $\ell = 5000$. The spectra are evaluated independently (one emulator for TT, one for TE, one for EE).
d) **Emulator access:** Bundled within the package, no external file paths or downloads are required. `emu_TT = Capse.trained_emulators["CAMB_LCDM"]["TT"]`.
e) **Limitations:** Parameters must fall within the specific minimum/maximum bounds they were trained on (visible via `emu.InMinMax`). Output is restricted to $\ell \in [2, 5000]$.

## 1.2 capse_paper Planck notebook summary
a) **Emulator used:** Standard `Capse` trained emulator (`Emu_TT`, `Emu_TE`, `Emu_EE`).
b) **Parameter vector:** `θ = [10*ln10As, ns, 100*h0, ωb/10, ωc, τ]`. This uses slightly different priors scaled down, so for physical standard cosmology the vector is simply `[ln10As, ns, H0, ωb, ωc, tau]`.
c) **Post-processing:** The returned 1D arrays are trimmed to specific ranges (`1:2507` for TT, meaning $\ell \le 2508$). Then, they are multiplied by $2\pi / (\ell(\ell+1))$ to convert back from $D_\ell$ to $C_\ell$.
d) **Likelihood:** The notebook uses `PlanckLite.jl` and its `bin_Cℓ` functions.
e) **Fiducial nuisances:** Only one nuisance $y_p$ is modeled (for PlanckLite absolute calibration).
f) **Chi-square/LogL:** Uses a reparameterized standard normal likelihood after whitening the prediction with the Cholesky inverse of the covariance matrix.

## 1.3 Hillipop.jl summary
a) **Loglikelihood signature:** `compute_loglike(ClTT, ClTE, ClEE, pars, h)` where `h` is the likelihood data loaded via `load_hillipop()`. Spectra must be $C_\ell$ in units of $\text{K}^2$ starting at $\ell=2$.
b) **ell range:** Operates up to `lmax = 2500`.
c) **Spectra required:** TT, TE, and EE.
d) **Nuisance parameters:** `HillipopNuisance` or a standard dictionary handles complex foregrounds and calibration components.

---

# Phase 2: Convention Mismatches and Resolution

**Mismatch 1: Normalisation**
- *Capse.jl produces:* $D_\ell = \frac{\ell(\ell+1)}{2\pi} C_\ell$ in units of $\mu\text{K}^2$.
- *Hillipop.jl expects:* $C_\ell$ in units of $\text{K}^2$.
- *Resolution:* Multiply the Capse output by $\frac{2\pi}{\ell(\ell+1)} \times 10^{-12}$.

**Mismatch 2: ℓ indexing and Range**
- *Capse.jl produces:* Vectors of size 4999 corresponding exactly to $\ell=2$ to $\ell=5000$.
- *Hillipop.jl expects:* Vectors starting at $\ell=2$ and ending at $h.lmax$ (which is $2500$). So the expected length is $2499$.
- *Resolution:* Trim the Capse output arrays to `1:(h.lmax - 1)` (i.e., `1:2499`).

**Mismatch 3: Spectrum layout**
- *Capse.jl produces:* Separate outputs for each evaluated emulator (e.g. `get_Cℓ(x, emu_TT)`).
- *Hillipop.jl expects:* Separate vectors `Cl_TT`, `Cl_TE`, `Cl_EE`.
- *Resolution:* No mismatch; simply pass the three trimmed and converted arrays as separate arguments to `compute_loglike`.
