using Hillipop
using Capse
using DifferentiationInterface
import ForwardDiff, Zygote, Mooncake
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

function benchmark_lux_mooncake()
    ell = emu_TT.ℓgrid[3:5001]
    idx_lmax = findfirst(==(lmax_h), ell)
    fac = @. 2π / (ell[1:idx_lmax] * (ell[1:idx_lmax] + 1)) * 1e-12

    # Fixed nuisance for benchmark
    nuisance = (
        A_planck = 1.0,
        AdustT   = 1.0,
        AdustP   = 1.0,
        Atsz     = 1.0,
        Aksz     = 1.0,
        Acib     = 1.0
    )

    x0 = [3.044, 0.9649, 67.36, 0.02237, 0.1200, 0.0544]

    function target_function(x)
        # Predict Theory
        raw_TT = Capse.get_Cℓ(x, emu_TT)
        raw_TE = Capse.get_Cℓ(x, emu_TE)
        raw_EE = Capse.get_Cℓ(x, emu_EE)

        # Scale and Trim
        Cl_TT = raw_TT[1:idx_lmax] .* fac
        Cl_TE = raw_TE[1:idx_lmax] .* fac
        Cl_EE = raw_EE[1:idx_lmax] .* fac

        return compute_loglike(Cl_TT, Cl_TE, Cl_EE, nuisance, h)
    end

    results = []

    println("\nBenchmarking Primal Evaluation (Lux)...")
    t_primal = @benchmark $target_function($x0)
    push!(results, (backend="Primal (Lux)", prepared=false, time=median(t_primal).time / 1e6, memory=t_primal.memory / 1024))

    # --- ForwardDiff ---
    println("\nBenchmarking ForwardDiff (Lux)...")
    backend_fd = AutoForwardDiff()
    grad_fd = gradient(target_function, backend_fd, x0)[1]
    
    # One-shot
    t_fd = @benchmark gradient($target_function, $backend_fd, $x0)
    push!(results, (backend="ForwardDiff", prepared=false, time=median(t_fd).time / 1e6, memory=t_fd.memory / 1024))
    
    # Prepared
    prep_fd = prepare_gradient(target_function, backend_fd, x0)
    buffer_fd = similar(x0)
    t_fd_p = @benchmark gradient!($target_function, $buffer_fd, $prep_fd, $backend_fd, $x0)
    push!(results, (backend="ForwardDiff", prepared=true, time=median(t_fd_p).time / 1e6, memory=t_fd_p.memory / 1024))

    # --- Mooncake ---
    println("\nBenchmarking Mooncake (Lux)...")
    backend_mc = AutoMooncake()
    try
        # One-shot
        println("Pre-compiling Mooncake rule (one-shot)...")
        grad_mc = gradient(target_function, backend_mc, x0)[1]
        println("Mooncake rule compiled successfully.")
        
        diff_mc = norm(grad_fd - grad_mc)
        println("Mooncake vs ForwardDiff norm diff: ", diff_mc)
        
        t_mc = @benchmark gradient($target_function, $backend_mc, $x0)
        push!(results, (backend="Mooncake", prepared=false, time=median(t_mc).time / 1e6, memory=t_mc.memory / 1024))
        
        # Prepared
        println("Preparing Mooncake gradient operator...")
        prep_mc = prepare_gradient(target_function, backend_mc, x0)
        buffer_mc = similar(x0)
        t_mc_p = @benchmark gradient!($target_function, $buffer_mc, $prep_mc, $backend_mc, $x0)
        push!(results, (backend="Mooncake", prepared=true, time=median(t_mc_p).time / 1e6, memory=t_mc_p.memory / 1024))
        
    catch e
        println("Mooncake benchmark failed (Lux): $e")
        push!(results, (backend="Mooncake", prepared=false, time=NaN, memory=0.0))
    end

    # --- Zygote ---
    println("\nBenchmarking Zygote (Lux)...")
    backend_zg = AutoZygote()
    try
        # One-shot
        println("Pre-compiling Zygote pullback (one-shot)...")
        grad_zg = gradient(target_function, backend_zg, x0)[1]
        println("Zygote pullback compiled successfully.")
        
        diff_zg = norm(grad_fd - grad_zg)
        println("Zygote vs ForwardDiff norm diff: ", diff_zg)
        
        t_zg = @benchmark gradient($target_function, $backend_zg, $x0)
        push!(results, (backend="Zygote", prepared=false, time=median(t_zg).time / 1e6, memory=t_zg.memory / 1024))
        
        # Prepared
        println("Preparing Zygote gradient operator...")
        prep_zg = prepare_gradient(target_function, backend_zg, x0)
        buffer_zg = similar(x0)
        t_zg_p = @benchmark gradient!($target_function, $buffer_zg, $prep_zg, $backend_zg, $x0)
        push!(results, (backend="Zygote", prepared=true, time=median(t_zg_p).time / 1e6, memory=t_zg_p.memory / 1024))
        
    catch e
        println("Zygote benchmark failed (Lux): $e")
        push!(results, (backend="Zygote", prepared=false, time=NaN, memory=0.0))
    end

    println("\nBenchmark Results (Median Time in ms):")
    df = DataFrame(results)
    println(df)
    
    return df
end

benchmark_lux_mooncake()
