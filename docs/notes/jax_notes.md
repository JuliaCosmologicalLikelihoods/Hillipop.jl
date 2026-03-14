# JAX Reference Notebook Summary

1. **CLASS Parameters:**
   - h: 0.6737
   - omega_b: 0.02237
   - omega_cdm: 0.1200
   - A_s: 2.1e-9
   - n_s: 0.9649
   - tau_reio: 0.0544
   - N_ur: 2.0328
   - N_ncdm: 1
   - m_ncdm: 0.06
   - output: 'tCl,pCl,lCl'
   - l_max_scalars: 2508
   - lensing: 'yes'

2. **Output Spectra Requested:**
   - TT, TE, EE (lensed spectra used)
   - Computed up to ell_max = 2500

3. **Ell Range and Post-Processing:**
   - Range: ell = 2 to 2500
   - Post-processing: Multiplied by T_cmb^2 to convert to muK^2 (Class outputs dimensionless C_ell, or D_ell? If it outputs C_ell, wait, JAX says T_cmb^2 * M.lensed_cl(2500)['tt']. Usually CLASS outputs dimensionless D_ell or C_ell. Let's verify CLASS output in CLASS.jl).

4. **Nuisance Parameters:**
   - cal100A, etc... (as listed in test/compare_julia.jl)
   
5. **Likelihood Computation:**
   - `compute_like(ClTT, ClTE, ClEE, nuisances)` where Cl are in K^2 or muK^2? Wait, Hillipop expects K^2 starting at l=2.
   - Wait, `T_cmb**2` might just mean muK^2 if T_cmb is in muK, or K^2 if T_cmb is in K. Usually `T_cmb = 2.7255`. If they multiply by `T_cmb**2` with T_cmb in K, they get K^2.
   
6. **Fiducial values:**
   - (Same as above)
