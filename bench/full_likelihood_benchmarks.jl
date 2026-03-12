using Hillipop
using Capse
using DifferentiationInterface
import ForwardDiff, Mooncake
using ADTypes
using BenchmarkTools
using LinearAlgebra
using DataFrames
using AbstractCosmologicalEmulators
using Artifacts

println("Loading Hillipop data and Capse Lux emulators...")
const h = load_hillipop(lmax=2500)
const lmax_h = h.lmax

# Load Lux emulators at top level
const artifacts_toml = joinpath(dirname(pathof(Capse)), "..", "Artifacts.toml")
const path = artifact_path(artifact_hash("CAMB_LCDM", artifacts_toml))

const emu_TT = Capse.load_emulator(joinpath(path, "TT/"), emu=LuxEmulator)
const emu_TE = Capse.load_emulator(joinpath(path, "TE/"), emu=LuxEmulator)
const emu_EE = Capse.load_emulator(joinpath(path, "EE/"), emu=LuxEmulator)

# Pre-compute constants
const ell_grid = emu_TT.ℓgrid[3:5001]
const idx_lmax_val = findfirst(==(lmax_h), ell_grid)
const fac_vec = @. 2π / (ell_grid[1:idx_lmax_val] * (ell_grid[1:idx_lmax_val] + 1)) * 1e-12

# Define the full 37-parameter mapping
# 1-6: Cosmo (ln10As, ns, H0, omb, omc, tau)
# 7-37: Nuisance (standard Hillipop order)
const NUISANCE_KEYS = [
    :A_planck, :cal100A, :cal100B, :cal143A, :cal143B, :cal217A, :cal217B,
    :pe100A, :pe100B, :pe143A, :pe143B, :pe217A, :pe217B,
    :AdustT, :AdustP, :beta_dustT, :beta_dustP,
    :Atsz, :Aksz,
    :Acib, :beta_cib, :xi,
    :Aradio, :beta_radio, :Adusty,
    :Asbpx_100x100, :Asbpx_100x143, :Asbpx_100x217, :Asbpx_143x143, :Asbpx_143x217, :Asbpx_217x217
]

function benchmark_full_likelihood()
    # Fiducial 37-parameter vector
    x0_cosmo = [3.044, 0.9649, 67.36, 0.02237, 0.1200, 0.0544]
    x0_nuis  = ones(length(NUISANCE_KEYS)) # Simplify for benchmark, most defaults are 1.0 or 0.0
    # Adjust some specific defaults
    x0_nuis[16:17] .= 1.5  # beta_dust
    x0_nuis[21] = 1.75     # beta_cib
    x0_nuis[24] = -0.7     # beta_radio
    x0_nuis[26:31] .= 0.0  # subpixel
    
    x0_full = vcat(x0_cosmo, x0_nuis)

    function target_function_full(x)
        cosmo_vec = x[1:6]
        # Build nuisance NamedTuple for AD compatibility
        nuis_vals = x[7:37]
        # Use a NamedTuple to avoid Dict-related AD failures
        nuisance_nt = NamedTuple{Tuple(NUISANCE_KEYS)}(Tuple(nuis_vals))

        # Predict Theory
        raw_TT = Capse.get_Cℓ(cosmo_vec, emu_TT)
        raw_TE = Capse.get_Cℓ(cosmo_vec, emu_TE)
        raw_EE = Capse.get_Cℓ(cosmo_vec, emu_EE)

        # Scale and Trim
        Cl_TT = raw_TT[1:idx_lmax_val] .* fac_vec
        Cl_TE = raw_TE[1:idx_lmax_val] .* fac_vec
        Cl_EE = raw_EE[1:idx_lmax_val] .* fac_vec

        return compute_loglike(Cl_TT, Cl_TE, Cl_EE, nuisance_nt, h)
    end

    results = []

    println("\nBenchmarking Full Primal Evaluation (37 params)...")
    t_primal = @benchmark $target_function_full($x0_full)
    push!(results, (backend="Primal", prepared=false, time=median(t_primal).time / 1e6, memory=t_primal.memory / 1024))

    # --- ForwardDiff ---
    println("Benchmarking Full ForwardDiff (Prepared)...")
    backend_fd = AutoForwardDiff()
    prep_fd = prepare_gradient(target_function_full, backend_fd, x0_full)
    buffer_fd = similar(x0_full)
    grad_fd = gradient(target_function_full, backend_fd, x0_full)
    t_fd = @benchmark gradient!($target_function_full, $buffer_fd, $prep_fd, $backend_fd, $x0_full)
    push!(results, (backend="ForwardDiff", prepared=true, time=median(t_fd).time / 1e6, memory=t_fd.memory / 1024))

    # --- Mooncake ---
    println("Benchmarking Full Mooncake (Prepared)...")
    backend_mc = AutoMooncake()
    try
        println("Preparing Mooncake gradient operator (this may take a minute)...")
        prep_mc = prepare_gradient(target_function_full, backend_mc, x0_full)
        buffer_mc = similar(x0_full)
        
        # Warmup and verify
        gradient!(target_function_full, buffer_mc, prep_mc, backend_mc, x0_full)
        println("Mooncake vs ForwardDiff norm diff: ", norm(grad_fd - buffer_mc))
        
        t_mc = @benchmark gradient!($target_function_full, $buffer_mc, $prep_mc, $backend_mc, $x0_full)
        push!(results, (backend="Mooncake", prepared=true, time=median(t_mc).time / 1e6, memory=t_mc.memory / 1024))
    catch e
        println("Mooncake full benchmark failed: $e")
        push!(results, (backend="Mooncake", prepared=true, time=NaN, memory=0.0))
    end

    println("\nFull Likelihood Benchmark Results (Median Time in ms):")
    df = DataFrame(results)
    println(df)
    
    return df
end

benchmark_full_likelihood()
