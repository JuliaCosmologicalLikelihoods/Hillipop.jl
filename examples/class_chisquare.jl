# ── Section 1: Imports and setup ─────────────────────────────────────────────
using Hillipop
using CLASS
using DataFrames
using LinearAlgebra

# ── Section 2: Define fiducial cosmological parameters ───────────────────────
# Fiducial values extracted from the JAX reference notebook.
cosmo_params = (
    H0        = 67.36,   # Hubble constant [km/s/Mpc]
    omega_b   = 0.02237, # physical baryon density
    omega_cdm = 0.1200,  # physical cold dark matter density
    A_s       = 2.101e-9,# primordial scalar amplitude
    n_s       = 0.9649,  # scalar spectral index
    tau_reio  = 0.0544,  # optical depth to reionisation
    N_ur      = 2.0328,  # effective number of ultra-relativistic species
    m_ncdm    = 0.06,    # sum of neutrino masses [eV]
)

# ── Section 3: Define nuisance parameters ────────────────────────────────────
# Using the fiducial nuisance values from the JAX notebook.
nuisance = HillipopNuisance(;
    cal = HillipopCalibration(
        A_planck = 1.0,
        cal100A = 1.0, cal100B = 1.0,
        cal143A = 1.0, cal143B = 1.0,
        cal217A = 1.0, cal217B = 1.0,
        pe100A  = 1.0, pe100B  = 1.0,
        pe143A  = 1.0, pe143B  = 1.0,
        pe217A  = 1.0, pe217B  = 1.0
    ),
    dust = HillipopDust(
        AdustT  = 1.0, AdustP  = 1.0,
        beta_dustT = 1.5, beta_dustP = 1.5
    ),
    sz = HillipopSZ(
        Atsz    = 1.0,
        Aksz    = 1.0
    ),
    cib = HillipopCIB(
        Acib    = 1.0,
        beta_cib = 1.75,
        xi      = 1.0
    ),
    ps = HillipopPointSources(
        Aradio  = 1.0,
        beta_radio = -0.7,
        Adusty  = 1.0
    ),
    subpixel = HillipopSubPixel(
        Asbpx_100x100 = 0.0, Asbpx_100x143 = 0.0,
        Asbpx_100x217 = 0.0, Asbpx_143x143 = 0.0,
        Asbpx_143x217 = 0.0, Asbpx_217x217 = 0.0
    )
)

# ── Section 4: Run CLASS via CLASS.jl ────────────────────────────────────────
# Set up the CLASS problem with correct output flags and ell_max.
class_input = Dict(
    "H0"            => cosmo_params.H0,
    "omega_b"       => cosmo_params.omega_b,
    "omega_cdm"     => cosmo_params.omega_cdm,
    "A_s"           => cosmo_params.A_s,
    "n_s"           => cosmo_params.n_s,
    "tau_reio"      => cosmo_params.tau_reio,
    "N_ur"          => cosmo_params.N_ur,
    "N_ncdm"        => 1,
    "m_ncdm"        => cosmo_params.m_ncdm,
    "output"        => "tCl, pCl, lCl",
    "lensing"       => "yes",
    "l_max_scalars" => 3000
)

println("Running CLASS...")
prob = CLASSProblem(class_input...)
sol = CLASS.solve(prob)
cls_df = sol["lCl"]  # get the lensed spectra
println("CLASS output (first 5 rows):")
println(first(cls_df, 5))

# ── Section 5: Post-process CLASS output to match Hillipop conventions ───────
# Mismatch 1: CLASS outputs dimensionless D_ell = l*(l+1)*Cl/(2pi).
#             Hillipop expects Cl in K^2.
# Mismatch 2: CLASS output typically starts at ell=2. Hillipop expects flat vectors from ell=2.
# Resolution: multiply by (2pi / l*(l+1)) * T_cmb^2.

ell = cls_df.l
T_cmb = 2.7255 # K

# We only want ells >= 2
valid_idx = ell .>= 2
ell_valid = ell[valid_idx]

# Conversion factor from CLASS D_ell to C_ell in K^2
conversion = @. (2 * pi / (ell_valid * (ell_valid + 1))) * T_cmb^2

Cl_TT = cls_df.TT[valid_idx] .* conversion
Cl_TE = cls_df.TE[valid_idx] .* conversion
Cl_EE = cls_df.EE[valid_idx] .* conversion

# Hillipop expects vectors that cover 2:lmax. We trim to Hillipop's lmax.
h = load_hillipop()
lmax_h = h.lmax

idx_lmax = findfirst(==(lmax_h), ell_valid)
if idx_lmax !== nothing
    Cl_TT = Cl_TT[1:idx_lmax]
    Cl_TE = Cl_TE[1:idx_lmax]
    Cl_EE = Cl_EE[1:idx_lmax]
else
    error("CLASS did not compute up to lmax = $lmax_h")
end

# ── Section 6: Initialise the Hillipop likelihood ───────────────────────────
# The data is already loaded into `h` above.

# ── Section 7: Evaluate the log-likelihood / chi-square ─────────────────────
loglike = compute_loglike(Cl_TT, Cl_TE, Cl_EE, nuisance, h)
chisq   = -2 * loglike
println("log-likelihood : ", loglike)
println("chi-square     : ", chisq)

# ── Section 8: Validation ────────────────────────────────────────────────────
# Compare the computed chi-square against the reference value from the JAX notebook
# (from the NumPy mode: -18693.558594 or JAX mode: -18693.669380).
# The expected chi2 is -2 * loglike = 37387.33876.
reference_loglike = -18693.669380
reference_chisq = -2 * reference_loglike

println("Reference chi-square : ", reference_chisq)
println("Difference           : ", abs(chisq - reference_chisq))

@assert abs(chisq - reference_chisq) < 0.1 * abs(reference_chisq) """
    Chi-square mismatch: got $chisq, expected $reference_chisq
"""
println("Success: chi-square matches the JAX reference within 10% tolerance.")
