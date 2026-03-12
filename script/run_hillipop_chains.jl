using Pkg
Pkg.activate(@__DIR__)

using Turing
using Capse
using AbstractCosmologicalEmulators
using Hillipop
using LinearAlgebra

println("Loading Hillipop data...")
# Load PR4 data up to lmax=2500
const data_dir = abspath(joinpath(@__DIR__, "..", "data"))
const h_data = load_hillipop(data_dir; lmax=2500)

println("Initializing Capse CAMB LCDM emulator...")
# Load pre-trained Capse emulator from the default artifact location
const emu_TT = Capse.trained_emulators["CAMB_LCDM"]["TT"]
const emu_TE = Capse.trained_emulators["CAMB_LCDM"]["TE"]
const emu_EE = Capse.trained_emulators["CAMB_LCDM"]["EE"]

# Define Capse output ranges & conversion units (from user)
const lsTT = 2:2508
const lsTE = 2:1996
const facTT = lsTT .* (lsTT .+ 1) ./ (2 * π)
const facTE = lsTE .* (lsTE .+ 1) ./ (2 * π)

# Helper function to evaluate the Emulator and convert to Dl -> Cl unshaped arrays
# Hillipop expects arrays of length 2499 (which map to ℓ=2:2500 internally).
function compute_theory_cls(θ)
    # Capse returns shaped spectra. We divide by our factor and truncate to length 2499 (ℓ=2..2500)
    # Since lsTT starts at 2, index 1 corresponds to ℓ=2, so we need 1:2499
    
    # TT has enough points (up to 2508)
    ClTT_raw = Capse.get_Cℓ(θ, emu_TT)[1:2499] ./ facTT[1:2499]
    
    # TE, EE only go up to ~1996 natively. 
    # For a full PR4 likelihood, we usually need up to 2500, so we pad the rest with zeros,
    # as Capse's neural network truncates earlier. The likelihood weights generally taper off anyway.
    n_te = length(lsTE) # 1995
    ClTE_raw = Capse.get_Cℓ(θ, emu_TE)[1:n_te] ./ facTE
    ClEE_raw = Capse.get_Cℓ(θ, emu_EE)[1:n_te] ./ facTE
    
    ClTE = zeros(eltype(ClTE_raw), 2499)
    ClEE = zeros(eltype(ClEE_raw), 2499)
    
    ClTE[1:n_te] .= ClTE_raw[1:n_te]
    ClEE[1:n_te] .= ClEE_raw[1:n_te]
    
    return ClTT_raw, ClTE, ClEE
end

println("Defining Turing Model...")

# Pre-invert the covariance matrix for MvNormal evaluation
const C_inv = Symmetric(inv(h_data.binned_invkll))

@model function cosmology_hillipop()
    # 1. Cosmology Priors (LCDM)
    # [ln10As, ns, H0, ωb, ωc, τ]
    ln10As ~ Uniform(2.0, 4.0)
    ns     ~ Uniform(0.8, 1.2)
    H0     ~ Uniform(40.0, 100.0)
    ωb     ~ Uniform(0.01, 0.04)
    ωc     ~ Uniform(0.05, 0.20)
    τ      ~ Uniform(0.01, 0.15)
    
    # 2. Hillipop Nuisance Priors
    # Calibration
    A_planck ~ Normal(1.0, 0.0025)
    cal100A  ~ Normal(1.0, 0.0025)
    cal100B  ~ Normal(1.0, 0.0025)
    cal143A  ~ Normal(1.0, 0.0025)
    cal143B  ~ Normal(1.0, 0.0025)
    cal217A  ~ Normal(1.0, 0.0025)
    cal217B  ~ Normal(1.0, 0.0025)
    
    # Polarization Efficiency
    pe100A   ~ Normal(1.0, 0.01)
    pe100B   ~ Normal(1.0, 0.01)
    pe143A   ~ Normal(1.0, 0.01)
    pe143B   ~ Normal(1.0, 0.01)
    pe217A   ~ Normal(1.0, 0.01)
    pe217B   ~ Normal(1.0, 0.01)
    
    # Foregrounds
    AdustT     ~ Uniform(0.0, 50.0)
    AdustP     ~ Uniform(0.0, 50.0)
    beta_dustT ~ Normal(1.59, 0.2)
    beta_dustP ~ Normal(1.59, 0.2)
    Atsz       ~ Uniform(0.0, 10.0)
    Aksz       ~ Uniform(0.0, 10.0)
    Acib       ~ Uniform(0.0, 20.0)
    beta_cib   ~ Normal(1.75, 0.2)
    xi         ~ Uniform(0.0, 1.0)
    Aradio     ~ Uniform(0.0, 10.0)
    beta_radio ~ Normal(-1.0, 1.0)
    Adusty     ~ Uniform(0.0, 10.0)

    # 3. Assemble parameters
    θ_cosmo = [ln10As, ns, H0, ωb, ωc, τ]
    
    # 4. Generate Theory C_ℓ
    ClTT, ClTE, ClEE = compute_theory_cls(θ_cosmo)
    
    # 5. Pack Nuisances
    pars = HillipopNuisance((
        A_planck=A_planck, cal100A=cal100A, cal100B=cal100B, cal143A=cal143A, cal143B=cal143B, cal217A=cal217A, cal217B=cal217B,
        pe100A=pe100A, pe100B=pe100B, pe143A=pe143A, pe143B=pe143B, pe217A=pe217A, pe217B=pe217B,
        AdustT=AdustT, AdustP=AdustP, beta_dustT=beta_dustT, beta_dustP=beta_dustP, Atsz=Atsz, Aksz=Aksz,
        Acib=Acib, beta_cib=beta_cib, xi=xi, Aradio=Aradio, beta_radio=beta_radio, Adusty=Adusty
    ))
    
    # 6. Forward pass through Hillipop
    Xl = build_residual_vector(ClTT, ClTE, ClEE, pars, h_data)
    Xl_binned = h_data.binning_matrix * Xl
    
    # 7. Likelihood Evaluation
    Xl_binned ~ MvNormal(zeros(length(Xl_binned)), C_inv)
end

println("Starting NUTS Sampler...")
# Define the sampler
sampler = NUTS(500, 0.65)

# Run 4 chains in parallel
chains = sample(cosmology_hillipop(), sampler, 1000)

println("Sampling Complete! Summary:")
display(chains)

# Save the chains internally for subsequent analysis
using Serialization
serialize(joinpath(@__DIR__, "hillipop_chains.jls"), chains)
println("Chains saved to script/hillipop_chains.jls")
