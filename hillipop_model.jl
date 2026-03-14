"""
    hillipop_model.jl

Defines a Turing.jl probabilistic model wrapping the Hillipop CMB likelihood
with Capse.jl emulators. All components are ForwardDiff compatible.
"""

using Turing
using Capse, Artifacts, Lux
using Hillipop
using LinearAlgebra
using Distributions
using AbstractCosmologicalEmulators

# --- 1. Global Emulator Setup (Initialized once) ---
println("Loading Capse emulators...")
const lmax_h = 2500
const artifacts_toml = joinpath(dirname(pathof(Capse)), "..", "Artifacts.toml")
const path = artifact_path(artifact_hash("CAMB_LCDM", artifacts_toml))

const emu_TT = Capse.load_emulator(joinpath(path, "TT/"), emu=LuxEmulator)
const emu_TE = Capse.load_emulator(joinpath(path, "TE/"), emu=LuxEmulator)
const emu_EE = Capse.load_emulator(joinpath(path, "EE/"), emu=LuxEmulator)

# The ell grid starts at index 3 (ell=2) up to 5001. 
# We truncate at Hillipop's lmax.
const ell = emu_TT.ℓgrid[3:5001]
const idx_lmax = findfirst(==(lmax_h), ell)

# Normalization factor: Capse returns D_ell [muK^2], Hillipop expects C_ell [K^2].
# fac = (2π / (ℓ(ℓ+1))) * 1e-12
const fac = @. 2π / (ell[1:idx_lmax] * (ell[1:idx_lmax] + 1)) * 1e-12

@model function cmb_hillipop(hillipop_obj)
    # --- 2. Cosmological parameter priors ---
    ln10As    ~ Uniform(2.5, 3.5)
    ns        ~ Uniform(0.88, 1.06)
    H0        ~ Uniform(55.0, 91.0)
    omega_b   ~ Uniform(0.019, 0.026)
    omega_cdm ~ Uniform(0.08, 0.16)
    tau_reio  ~ Truncated(Normal(0.0543, 0.0073), 0.01, 0.15)

    # --- 3. Nuisance parameter priors (Planck PR4 Config) ---
    # Global calibration
    A_planck   ~ Truncated(Normal(1.0, 0.0025), 0.9, 1.1)
    
    # Map-level absolute calibrations
    cal100A    ~ Uniform(0.9, 1.1)
    cal100B    ~ Uniform(0.9, 1.1)
    cal143B    ~ Uniform(0.9, 1.1)
    cal217A    ~ Uniform(0.9, 1.1)
    cal217B    ~ Uniform(0.9, 1.1)
    
    # Galactic dust (Amplitudes and Spectral Indices)
    AdustT     ~ Truncated(Normal(1.0, 0.1), 0.5, 1.5)
    AdustP     ~ Truncated(Normal(1.0, 0.1), 0.7, 1.3)
    beta_dustT ~ Truncated(Normal(1.51, 0.01), 1.4, 1.6)
    beta_dustP ~ Truncated(Normal(1.59, 0.01), 1.5, 1.7)
    
    # Sunyaev-Zeldovich (tSZ and kSZ)
    Atsz       ~ Uniform(0.0, 50.0)
    Aksz       ~ Uniform(0.0, 50.0)
    
    # Cosmic Infrared Background (CIB)
    Acib       ~ Uniform(0.0, 20.0)
    beta_cib   ~ Truncated(Normal(1.75, 0.06), 1.6, 1.9)
    xi         ~ Uniform(-1.0, 1.0) # SZ x CIB correlation
    
    # Point Sources
    Aradio     ~ Uniform(0.0, 150.0)
    Adusty     ~ Uniform(0.0, 150.0)

    # --- 4. Theory spectra via Capse ---
    # Parameter order: [ln10As, ns, H0, omb, omc, tau]
    x_cosmo = [ln10As, ns, H0, omega_b, omega_cdm, tau_reio]
    
    raw_TT = Capse.get_Cℓ(x_cosmo, emu_TT)
    raw_TE = Capse.get_Cℓ(x_cosmo, emu_TE)
    raw_EE = Capse.get_Cℓ(x_cosmo, emu_EE)

    # Apply truncation and normalization
    Cl_TT = raw_TT[1:idx_lmax] .* fac
    Cl_TE = raw_TE[1:idx_lmax] .* fac
    Cl_EE = raw_EE[1:idx_lmax] .* fac

    # --- 5. Assemble Nuisance Struct ---
    # We use a NamedTuple which HillipopNuisance constructor handles.
    # Fixed parameters (cal143A, peXXX, beta_radio) fall back to struct defaults.
    nuis = (
        A_planck=A_planck, cal100A=cal100A, cal100B=cal100B, cal143B=cal143B, 
        cal217A=cal217A, cal217B=cal217B, AdustT=AdustT, AdustP=AdustP, 
        beta_dustT=beta_dustT, beta_dustP=beta_dustP, Atsz=Atsz, Aksz=Aksz, 
        Acib=Acib, beta_cib=beta_cib, xi=xi, Aradio=Aradio, Adusty=Adusty
    )

    # --- 6. Log-Likelihood Evaluation ---
    # We inject the Gaussian likelihood into the model using @addlogprob!
    Turing.@addlogprob! compute_loglike(Cl_TT, Cl_TE, Cl_EE, nuis, hillipop_obj)
end
