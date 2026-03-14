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

@model function hillipop_cosmo_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac)
    # Cosmological Priors (Capse CAMB_LCDM order: ln10As, ns, H0, omb, omc, tau)
    ln10As    ~ Uniform(2.5, 3.5)
    ns        ~ Uniform(0.88, 1.06)
    H0        ~ Uniform(60.0, 80.0)
    omega_b   ~ Uniform(0.019, 0.025)
    omega_cdm ~ Uniform(0.08, 0.20)
    tau_reio  ~ Normal(0.0506, 0.0086)

    # Fix all nuisance to unit/fiducial values
    nuisance = HillipopNuisance(Dict{Symbol, Float64}())

    # Predict Theory
    x = [ln10As, ns, H0, omega_b, omega_cdm, tau_reio]

    raw_TT = Capse.get_Cℓ(x, emu_TT)
    raw_TE = Capse.get_Cℓ(x, emu_TE)
    raw_EE = Capse.get_Cℓ(x, emu_EE)

    # Scale and Trim
    Cl_TT = raw_TT[1:idx_lmax] .* fac
    Cl_TE = raw_TE[1:idx_lmax] .* fac
    Cl_EE = raw_EE[1:idx_lmax] .* fac

    # Likelihood
    Turing.@addlogprob! compute_loglike(Cl_TT, Cl_TE, Cl_EE, nuisance, h)
end

# ── Section 4: Sampling Function ─────────────────────────────────────────────

function run_sampling(model, name, ad_backend; n_tune=50, n_samp=100)
    println("\n" * "="^60)
    println("RUNNING: $name with backend $ad_backend")
    println("="^60)

    # NUTS: 50 tuning steps, 100 accepted steps
    start_time = time()
    chain = sample(model, NUTS(n_tune, 0.65; adtype=ad_backend), n_samp)
    end_time = time()

    println("\nResults for $name:")
    println("Wall time: ", round(end_time - start_time, digits=2), " seconds")
    display(chain)
    return chain
end

# ── Section 5: Run Comparisons ───────────────────────────────────────────────

model = hillipop_cosmo_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac)

# 1. ForwardDiff
chain_fd = run_sampling(model, "ForwardDiff", AutoForwardDiff())

# 2. Mooncake
import Mooncake
chain_mc = run_sampling(model, "Mooncake", AutoMooncake())

println("\nAll comparisons completed!")
