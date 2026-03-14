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

# ── Section 1: Load Data and Emulators ───────────────────────────────────────

println("Loading Hillipop data and Capse Lux emulators...")
const h_data = load_hillipop()
const lmax_h = h_data.lmax

# Load Lux emulators
const artifacts_toml = joinpath(dirname(pathof(Capse)), "..", "Artifacts.toml")
const path = artifact_path(artifact_hash("CAMB_LCDM", artifacts_toml))
const emu_TT = Capse.load_emulator(joinpath(path, "TT/"), emu=LuxEmulator)
const emu_TE = Capse.load_emulator(joinpath(path, "TE/"), emu=LuxEmulator)
const emu_EE = Capse.load_emulator(joinpath(path, "EE/"), emu=LuxEmulator)

# Pre-compute constants
const ell = emu_TT.ℓgrid[3:5001] 
const idx_lmax = findfirst(==(lmax_h), ell)
const fac = @. 2π / (ell[1:idx_lmax] * (ell[1:idx_lmax] + 1)) * 1e-12

# ── Section 2: Define Turing Model with Realistic Priors ─────────────────────

@model function hillipop_production_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac)
    # --- Cosmological Parameters ---
    ln10As    ~ Uniform(2.5, 3.5)
    ns        ~ Uniform(0.88, 1.06)
    H0        ~ Uniform(60.0, 80.0)
    omega_b   ~ Uniform(0.019, 0.025)
    omega_cdm ~ Uniform(0.08, 0.20)
    tau_reio  ~ Normal(0.0506, 0.0086)

    # --- Nuisance Parameters (Based on paper suggested priors) ---
    A_planck   ~ Normal(1.0, 0.0025)
    
    # Per-map calibration (c_143A is fixed to 1.0)
    cal100A    ~ Uniform(0.9, 1.1)
    cal100B    ~ Uniform(0.9, 1.1)
    cal143B    ~ Uniform(0.9, 1.1)
    cal217A    ~ Uniform(0.9, 1.1)
    cal217B    ~ Uniform(0.9, 1.1)

    # Galactic dust rescaling
    AdustT     ~ Normal(1.0, 0.1)
    AdustP     ~ Normal(1.0, 0.1)
    beta_dustT ~ Normal(1.51, 0.01)
    beta_dustP ~ Normal(1.59, 0.02)

    # SZ
    Atsz       ~ Uniform(0.0, 50.0)
    Aksz       ~ Uniform(0.0, 50.0)

    # CIB
    Acib       ~ Uniform(0.0, 20.0)
    beta_cib   ~ Normal(1.75, 0.06)
    xi         ~ Uniform(-1.0, 1.0)

    # Point Sources
    Aradio     ~ Uniform(0.0, 150.0)
    Adusty     ~ Uniform(0.0, 150.0)

    # Pack into NamedTuple for AD-friendly construction
    nuisance_nt = (
        A_planck = A_planck,
        cal100A = cal100A, cal100B = cal100B, cal143B = cal143B, 
        cal217A = cal217A, cal217B = cal217B,
        AdustT = AdustT, AdustP = AdustP, 
        beta_dustT = beta_dustT, beta_dustP = beta_dustP,
        Atsz = Atsz, Aksz = Aksz,
        Acib = Acib, beta_cib = beta_cib, xi = xi,
        Aradio = Aradio, Adusty = Adusty
    )

    # --- Theory Prediction ---
    x_cosmo = [ln10As, ns, H0, omega_b, omega_cdm, tau_reio]
    raw_TT = Capse.get_Cℓ(x_cosmo, emu_TT)
    raw_TE = Capse.get_Cℓ(x_cosmo, emu_TE)
    raw_EE = Capse.get_Cℓ(x_cosmo, emu_EE)

    Cl_TT = raw_TT[1:idx_lmax] .* fac
    Cl_TE = raw_TE[1:idx_lmax] .* fac
    Cl_EE = raw_EE[1:idx_lmax] .* fac

    # --- Likelihood ---
    Turing.@addlogprob! compute_loglike(Cl_TT, Cl_TE, Cl_EE, nuisance_nt, h)
end

# ── Section 3: Performance Comparison ───────────────────────────────────────

function benchmark_sampling(model, name, ad_backend; n_tune=50, n_samp=100)
    println("\n" * "="^60)
    println("RUNNING: $name with backend $ad_backend")
    println("="^60)

    start_time = time()
    chain = sample(model, NUTS(n_tune, 0.65; adtype=ad_backend), n_samp)
    wall_time = time() - start_time

    println("\nResults for $name:")
    println("Wall time: ", round(wall_time, digits=2), " seconds")
    println("Time per sample (incl. tuning): ", round(wall_time / (n_tune + n_samp), digits=3), " s")
    display(chain)
    return chain
end

model = hillipop_production_model(h_data, emu_TT, emu_TE, emu_EE, idx_lmax, fac)

# 1. ForwardDiff
chain_fd = benchmark_sampling(model, "ForwardDiff", AutoForwardDiff())

# 2. Mooncake
import Mooncake
chain_mc = benchmark_sampling(model, "Mooncake", AutoMooncake())

println("\nBenchmarks completed!")
