using Hillipop
using Capse
using AbstractCosmologicalEmulators
using Turing
using Artifacts
using LinearAlgebra
using Mooncake
using ADTypes
using Lux

# 1. Setup Data and Emulators
println("Loading Hillipop data and Capse emulators...")
const h = load_hillipop(lmax=2500)
const lmax_h = h.lmax

const artifacts_toml = joinpath(dirname(pathof(Capse)), "..", "Artifacts.toml")
const path = artifact_path(artifact_hash("CAMB_LCDM", artifacts_toml))
const emu_TT = Capse.load_emulator(joinpath(path, "TT/"), emu=LuxEmulator)
const emu_TE = Capse.load_emulator(joinpath(path, "TE/"), emu=LuxEmulator)
const emu_EE = Capse.load_emulator(joinpath(path, "EE/"), emu=LuxEmulator)

const ell = emu_TT.ℓgrid[3:5001]
const idx_lmax = findfirst(==(lmax_h), ell)
const fac = @. 2π / (ell[1:idx_lmax] * (ell[1:idx_lmax] + 1)) * 1e-12

# 2. Define the Combined Turing Model
@model function hillipop_full_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac)
    # --- Cosmology Priors ---
    ln10As    ~ Uniform(2.0, 4.0)
    ns        ~ Uniform(0.84, 1.1)
    H0        ~ Uniform(60.0, 82.0)
    omega_b   ~ Uniform(0.019, 0.026)
    omega_cdm ~ Uniform(0.05, 0.255)
    tau_reio  ~ Normal(0.0506, 0.0086)

    # --- Nuisance Priors (HiLLiPoP PR4) ---
    A_planck   ~ Truncated(Normal(1.0, 0.0025), 0.9, 1.1)
    cal100A    ~ Uniform(0.9, 1.1); cal100B ~ Uniform(0.9, 1.1)
    cal143B    ~ Uniform(0.9, 1.1)
    cal217A    ~ Uniform(0.9, 1.1); cal217B ~ Uniform(0.9, 1.1)

    AdustT     ~ Truncated(Normal(1.0, 0.1), 0.5, 1.5)
    AdustP     ~ Truncated(Normal(1.0, 0.1), 0.7, 1.3)
    beta_dustT ~ Truncated(Normal(1.51, 0.01), 1.4, 1.6)
    beta_dustP ~ Truncated(Normal(1.59, 0.01), 1.5, 1.7)

    Atsz       ~ Uniform(0.0, 50.0); Aksz ~ Uniform(0.0, 50.0)
    Acib       ~ Uniform(0.0, 20.0); beta_cib ~ Truncated(Normal(1.75, 0.06), 1.6, 1.9)
    xi         ~ Uniform(-1.0, 1.0)

    Aradio     ~ Uniform(0.0, 150.0)
    Adusty     ~ Uniform(0.0, 150.0)

    # --- Theory Spectra ---
    x_cosmo = [ln10As, ns, H0, omega_b, omega_cdm, tau_reio]

    # ForwardDiff through emulators and likelihood
    raw_TT = Capse.get_Cℓ(x_cosmo, emu_TT)
    raw_TE = Capse.get_Cℓ(x_cosmo, emu_TE)
    raw_EE = Capse.get_Cℓ(x_cosmo, emu_EE)

    Cl_TT = raw_TT[1:idx_lmax] .* fac
    Cl_TE = raw_TE[1:idx_lmax] .* fac
    Cl_EE = raw_EE[1:idx_lmax] .* fac

    # --- Likelihood ---
    pars = (
        A_planck=A_planck, cal100A=cal100A, cal100B=cal100B, cal143B=cal143B,
        cal217A=cal217A, cal217B=cal217B, AdustT=AdustT, AdustP=AdustP,
        beta_dustT=beta_dustT, beta_dustP=beta_dustP, Atsz=Atsz, Aksz=Aksz,
        Acib=Acib, beta_cib=beta_cib, xi=xi, Aradio=Aradio, Adusty=Adusty
    )

    Turing.@addlogprob! compute_loglike(Cl_TT, Cl_TE, Cl_EE, pars, h)
end

# 3. Instantiate and Sample
model = hillipop_full_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac)

println("\nStarting NUTS sampling (50 adaptation, 50 samples)...")
println("AD Backend: AutoForwardDiff()")

# Syntax for modern Turing NUTS with explicit AD choice
chain = sample(model, NUTS(500, 0.65; adtype=AutoMooncake()), 1000)

println("\nSampling Complete!")
println(describe(chain))
