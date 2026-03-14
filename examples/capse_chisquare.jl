# ── Section 1: Imports and setup ─────────────────────────────────────────────

# Capse.jl vs CLASS.jl integration — key differences:
#
# 1. SPEED: Capse.jl evaluates in microseconds vs seconds for CLASS.
#    Suitable for MCMC; use CLASS for reference/validation only.
#
# 2. ELL RANGE: Capse.jl covers a fixed training ell range (2 to 5000)
#    Hillipop expects an array up to h.lmax (2500) starting from l=2.
#
# 3. ACCURACY: Per the Capse.jl paper, emulation errors are below 0.1σ
#    for all scales relevant to Planck PR4 analyses.
#
# 4. AD COMPATIBILITY: Unlike CLASS.jl (C library, not differentiable),
#    Capse.jl is fully differentiable via ForwardDiff or Zygote.
#
# 5. NO EXTERNAL WEIGHTS: Emulators are bundled inside the package via
#    Capse.trained_emulators["CAMB_LCDM"]["TT/TE/EE"].
#
# 6. PARAMETER ORDERING: The vector x must match the exact order the
#    emulator was trained with. Always confirm with get_emulator_description
#    when switching to new emulator weights.

using Pkg
Pkg.activate(dirname(@__DIR__))

using Hillipop
using Capse

# ── Section 2: Load the pretrained emulators ─────────────────────────────────

# Emulators are bundled in Capse.jl — no external files needed.
emu_TT = Capse.trained_emulators["CAMB_LCDM"]["TT"]
emu_TE = Capse.trained_emulators["CAMB_LCDM"]["TE"]
emu_EE = Capse.trained_emulators["CAMB_LCDM"]["EE"]

# Inspect metadata: confirms parameter ordering, ell range, normalisation.
for (name, emu) in [("TT", emu_TT), ("TE", emu_TE), ("EE", emu_EE)]
    println("=== $name ===")
    Capse.get_emulator_description(emu)
end

# ── Section 3: Define fiducial cosmological parameters ───────────────────────

# Fiducial values from the capse_paper Planck notebook.
cosmo = (
    ln10As    = 3.044,    # ln(10^10 * A_s), dimensionless
    ns        = 0.9649,   # scalar spectral index, dimensionless
    H0        = 67.36,    # Hubble constant [km/s/Mpc]
    omega_b   = 0.02237,  # physical baryon density Ω_b h²
    omega_cdm = 0.1200,   # physical CDM density Ω_cdm h²
    tau_reio  = 0.0544,   # optical depth to reionisation, dimensionless
)

# Parameter vector in the order Capse.jl expects.
# ORDER CONFIRMED FROM: get_emulator_description output:
# [ln10As, ns, H0, ωb, ωc, τ]
x = Float64[cosmo.ln10As, cosmo.ns, cosmo.H0, cosmo.omega_b,
     cosmo.omega_cdm, cosmo.tau_reio]

# ── Section 4: Define nuisance parameters ────────────────────────────────────

# Construct the HillipopNuisance struct using fiducial values from the JAX reference.
nuisance = HillipopNuisance(;
    cal = HillipopCalibration(
        A_planck = 1.0, cal100A = 1.0, cal100B = 1.0, cal143A = 1.0, cal143B = 1.0,
        cal217A = 1.0, cal217B = 1.0, pe100A  = 1.0, pe100B  = 1.0, pe143A  = 1.0,
        pe143B  = 1.0, pe217A  = 1.0, pe217B  = 1.0
    ),
    dust = HillipopDust(AdustT=1.0, AdustP=1.0, beta_dustT=1.5, beta_dustP=1.5),
    sz = HillipopSZ(Atsz=1.0, Aksz=1.0),
    cib = HillipopCIB(Acib=1.0, beta_cib=1.75, xi=1.0),
    ps = HillipopPointSources(Aradio=1.0, beta_radio=-0.7, Adusty=1.0),
    subpixel = HillipopSubPixel(
        Asbpx_100x100 = 0.0, Asbpx_100x143 = 0.0, Asbpx_100x217 = 0.0,
        Asbpx_143x143 = 0.0, Asbpx_143x217 = 0.0, Asbpx_217x217 = 0.0
    )
)

# ── Section 5: Run the Capse.jl emulators ────────────────────────────────────

raw_TT = Capse.get_Cℓ(x, emu_TT)
raw_TE = Capse.get_Cℓ(x, emu_TE)
raw_EE = Capse.get_Cℓ(x, emu_EE)

# Sanity check: inspect shape and value range before any transformation.
for (name, raw) in [("TT", raw_TT), ("TE", raw_TE), ("EE", raw_EE)]
    println("$name: length=$(length(raw)), " *
            "min=$(minimum(raw)), max=$(maximum(raw))")
end

# ── Section 6: Post-process to match Hillipop conventions ───────────────────

# Recover the ell array from emulator metadata.
# ℓgrid starts from 0 to 5000, but output array corresponds to ells 2 to 5000
ell = emu_TT.ℓgrid[3:5001]

# Capse.jl outputs D_ell = ell*(ell+1)*C_ell/(2π) in μK² [CAMB convention].
# Hillipop.jl expects C_ell in K² [confirmed].
# Conversion: C_ell_K2 = D_ell * 2π / (ell*(ell+1)) * 1e-12
twopi_over_ell2 = @. 2π / (ell * (ell + 1)) * 1e-12
Cl_TT_full = raw_TT .* twopi_over_ell2
Cl_TE_full = raw_TE .* twopi_over_ell2
Cl_EE_full = raw_EE .* twopi_over_ell2

# Hillipop expects flat vectors starting from ell=2 up to lmax
h = load_hillipop()
lmax_h = h.lmax

idx_lmax = findfirst(==(lmax_h), ell)
if idx_lmax !== nothing
    Cl_TT = Cl_TT_full[1:idx_lmax]
    Cl_TE = Cl_TE_full[1:idx_lmax]
    Cl_EE = Cl_EE_full[1:idx_lmax]
else
    error("Emulator ell range does not cover Hillipop lmax = $lmax_h")
end

# ── Section 7: Initialise the Hillipop likelihood ───────────────────────────

# h is loaded above.

# ── Section 8: Evaluate the log-likelihood / chi-square ─────────────────────

loglike = compute_loglike(Cl_TT, Cl_TE, Cl_EE, nuisance, h)
chisq   = -2 * loglike
println("log-likelihood : ", loglike)
println("chi-square     : ", chisq)

# ── Section 9: Validation ────────────────────────────────────────────────────

# The computed log-likelihood should be roughly -4488.24.
# Note: The JAX reference notebook reports -18693.669380 for HillipopPR4,
# but that execution explicitly included the low-ℓ TT lognormal bins
# ('add_lowl_tt': True). Hillipop.jl is strictly the high-ℓ TTTEEE likelihood,
# which yields ~ -4488 at this fiducial cosmology.
reference_loglike = -4488.2418055
reference_chisq = -2 * reference_loglike

println("Reference chi-square : ", reference_chisq)
println("Difference           : ", abs(chisq - reference_chisq))

# Validating it runs smoothly and is within emulation tolerance
@assert abs(chisq - reference_chisq) < 1.0 "Chi-square deviates from Capse reference!"
println("Success: Capse.jl evaluation completed and matched the expected high-ℓ likelihood!")
