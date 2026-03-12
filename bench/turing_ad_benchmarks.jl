using Hillipop
using Capse
using Turing
using ADTypes
using LinearAlgebra
using AbstractCosmologicalEmulators
using Artifacts
using DynamicPPL.TestUtils.AD: run_ad

println("Loading Hillipop data and Capse Lux emulators...")
const h = load_hillipop()
const lmax_h = h.lmax

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

# Define the Turing Model
@model function hillipop_full_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac)
    # 1. Cosmological Priors
    ln10As    ~ Uniform(2.5, 3.5)
    ns        ~ Uniform(0.88, 1.06)
    H0        ~ Uniform(60.0, 80.0)
    omega_b   ~ Uniform(0.019, 0.025)
    omega_cdm ~ Uniform(0.08, 0.20)
    tau_reio  ~ Normal(0.0506, 0.0086)

    # 2. Nuisance Parameters (Subset for realistic full-parameter benchmark)
    A_planck   ~ Normal(1.0, 0.0025)
    AdustT     ~ Uniform(0.0, 2.0)
    AdustP     ~ Uniform(0.0, 2.0)
    Atsz       ~ Uniform(0.0, 2.0)
    Aksz       ~ Uniform(0.0, 2.0)
    Acib       ~ Uniform(0.0, 2.0)
    
    nuisance = (
        A_planck = A_planck,
        AdustT   = AdustT,
        AdustP   = AdustP,
        Atsz     = Atsz,
        Aksz     = Aksz,
        Acib     = Acib
    )

    # 3. Theory
    x = [ln10As, ns, H0, omega_b, omega_cdm, tau_reio]
    raw_TT = Capse.get_Cℓ(x, emu_TT)
    raw_TE = Capse.get_Cℓ(x, emu_TE)
    raw_EE = Capse.get_Cℓ(x, emu_EE)

    Cl_TT = raw_TT[1:idx_lmax] .* fac
    Cl_TE = raw_TE[1:idx_lmax] .* fac
    Cl_EE = raw_EE[1:idx_lmax] .* fac

    # 4. Likelihood
    Turing.@addlogprob! compute_loglike(Cl_TT, Cl_TE, Cl_EE, nuisance, h)
end

println("Initializing model...")
model = hillipop_full_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac)

# Benchmark backends
backends = [
    ("ForwardDiff", AutoForwardDiff()),
    ("Mooncake", AutoMooncake())
]

println("\n" * "="^60)
println("TURING AD BENCHMARK (LogDensityProblems Interface)")
println("="^60)

for (name, adtype) in backends
    println("\nRunning benchmark for: $name")
    try
        # warmup
        run_ad(model, adtype)
        # actual benchmark
        result = run_ad(model, adtype; benchmark=true)
        
        println("  Primal Time:   ", round(result.primal_time * 1000, digits=2), " ms")
        println("  Gradient Time: ", round(result.grad_time * 1000, digits=2), " ms")
        println("  Ratio (G/P):   ", round(result.grad_time / result.primal_time, digits=2))
    catch e
        @warn "Backend $name failed: $e"
    end
end
