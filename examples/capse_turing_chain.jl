using Pkg
Pkg.activate(dirname(@__DIR__))

using Hillipop
using Capse
using Turing
using ADTypes
using LinearAlgebra
using Statistics
using AbstractCosmologicalEmulators
using Artifacts

# ── Section 2: Load Data and Emulators ───────────────────────────────────────

println("Loading Hillipop data and Capse Lux emulators...")
const h = load_hillipop()
const lmax_h = h.lmax

# Load Lux emulators at top level to avoid world age issues
const artifacts_toml = joinpath(dirname(pathof(Capse)), "..", "Artifacts.toml")
const path = artifact_path(artifact_hash("CAMB_LCDM", artifacts_toml))

const emu_TT = Capse.load_emulator(joinpath(path, "TT/"), emu=LuxEmulator)
const emu_TE = Capse.load_emulator(joinpath(path, "TE/"), emu=LuxEmulator)
const emu_EE = Capse.load_emulator(joinpath(path, "EE/"), emu=LuxEmulator)

# Pre-compute constants for the model
const ell = emu_TT.ℓgrid[3:5001] # l = 2 to 5000
const idx_lmax = findfirst(==(lmax_h), ell)
# Conversion factor from D_ell [muK^2] to C_ell [K^2]
const fac = @. 2π / (ell[1:idx_lmax] * (ell[1:idx_lmax] + 1)) * 1e-12

# ── Section 3: Define the Turing Model ───────────────────────────────────────

@model function hillipop_capse_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac; sample_nuisance=false)
    # 1. Cosmological Priors (Capse CAMB_LCDM order: ln10As, ns, H0, omb, omc, tau)
    ln10As    ~ Uniform(2.5, 3.5)
    ns        ~ Uniform(0.88, 1.06)
    H0        ~ Uniform(60.0, 80.0)
    omega_b   ~ Uniform(0.019, 0.025)
    omega_cdm ~ Uniform(0.08, 0.20)
    tau_reio  ~ Normal(0.0506, 0.0086)

    # 2. Nuisance Parameters
    local nuisance
    if sample_nuisance
        # Use placeholders for demonstration (subset of Hillipop nuisances)
        A_planck   ~ Normal(1.0, 0.0025)
        AdustT     ~ Uniform(0.0, 2.0)
        AdustP     ~ Uniform(0.0, 2.0)
        Atsz       ~ Uniform(0.0, 2.0)
        Aksz       ~ Uniform(0.0, 2.0)
        Acib       ~ Uniform(0.0, 2.0)

        # Use NamedTuple for better AD compatibility
        nuisance = (
            A_planck = A_planck,
            AdustT   = AdustT,
            AdustP   = AdustP,
            Atsz     = Atsz,
            Aksz     = Aksz,
            Acib     = Acib
        )
    else
        # Fix all to unit/fiducial values
        nuisance = HillipopNuisance(Dict{Symbol, Float64}())
    end

    # 3. Predict Theory
    x = [ln10As, ns, H0, omega_b, omega_cdm, tau_reio]

    # Capse returns D_ell [muK^2] for l=2:5000
    raw_TT = Capse.get_Cℓ(x, emu_TT)
    raw_TE = Capse.get_Cℓ(x, emu_TE)
    raw_EE = Capse.get_Cℓ(x, emu_EE)

    # Scale and Trim
    Cl_TT = raw_TT[1:idx_lmax] .* fac
    Cl_TE = raw_TE[1:idx_lmax] .* fac
    Cl_EE = raw_EE[1:idx_lmax] .* fac

    # 4. Likelihood
    Turing.@addlogprob! compute_loglike(Cl_TT, Cl_TE, Cl_EE, nuisance, h)
end
# ── Section 4: Run 1 — Fixed Nuisance ────────────────────────────────────────

function run_sampling(model, name, ad_backend; n_tune=20, n_samp=50)
    println("\n" * "="^60)
    println("RUNNING: $name with backend $ad_backend")
    println("="^60)

    try
        # NUTS: fewer steps for testing backends
        chain = sample(model, NUTS(n_tune, 0.65; adtype=ad_backend), n_samp)
        println("\nResults for $name:")
        display(chain)
        print(describe(chain))
        return chain
    catch e
        @warn "$name sampling failed: $e"
        return nothing
    end
end

model_fixed = hillipop_capse_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac; sample_nuisance=false)

# 1. ForwardDiff
run_sampling(model_fixed, "Fixed Nuisance (ForwardDiff)", AutoForwardDiff())

# 2. Mooncake
import Mooncake
run_sampling(model_fixed, "Fixed Nuisance (Mooncake)", AutoMooncake())

# 3. Zygote
import Zygote
run_sampling(model_fixed, "Fixed Nuisance (Zygote)", AutoZygote())

# ── Section 5: Run 2 — Free Nuisance ─────────────────────────────────────────

model_free = hillipop_capse_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac; sample_nuisance=true)
run_sampling(model_free, "Free Nuisance (ForwardDiff)", AutoForwardDiff())

println("\nAll chains completed successfully!")
