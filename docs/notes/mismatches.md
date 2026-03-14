# Convention Mismatches

1. **Mismatch:** C_ell vs D_ell normalisation and Units.
   - **CLASS.jl produces:** Usually dimensionless $D_\ell = \ell(\ell+1)C_\ell/(2\pi)$. 
   - **Hillipop.jl expects:** $C_\ell$ in $K^2$.
   - **Resolution:** Convert via `C_ell_K2 = (D_ell * 2 * pi / (ell * (ell + 1))) * (T_cmb_K^2)`. Use `T_cmb_K = 2.7255`.

2. **Mismatch:** ell indexing.
   - **CLASS.jl produces:** A DataFrame with an `:ell` column starting at $\ell=2$.
   - **Hillipop.jl expects:** A flat `Vector{Float64}` of length `lmax - 1`, starting exactly at $\ell=2$ and ending at `lmax=2500` (for fiducial setup).
   - **Resolution:** Filter DataFrame to `ell <= 2500` and assert it starts at $\ell=2$.

3. **Mismatch:** Lensing flag and output string.
   - **CLASS.jl expects:** `"output" => "tCl, pCl, lCl"`, `"lensing" => "yes"`.
   - **Resolution:** Extract `:TT`, `:TE`, `:EE` from the **lensed** Cls DataFrame.
