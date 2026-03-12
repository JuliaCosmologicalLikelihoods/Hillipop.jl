using Hillipop
using Capse
using DifferentiationInterface
import ForwardDiff, Zygote
using ADTypes
using BenchmarkTools
using LinearAlgebra
using DataFrames

function benchmark_ad_backends()
    println("Loading Hillipop data and Capse emulators...")
    h = load_hillipop(lmax=2500)
    lmax_h = h.lmax

    emu_TT = Capse.trained_emulators["CAMB_LCDM"]["TT"]
    emu_TE = Capse.trained_emulators["CAMB_LCDM"]["TE"]
    emu_EE = Capse.trained_emulators["CAMB_LCDM"]["EE"]

    ell = emu_TT.ℓgrid[3:5001]
    idx_lmax = findfirst(==(lmax_h), ell)
    fac = @. 2π / (ell[1:idx_lmax] * (ell[1:idx_lmax] + 1)) * 1e-12

    # Fixed nuisance for benchmark (using NamedTuple for better AD)
    nuisance = (
        A_planck = 1.0,
        AdustT   = 1.0,
        AdustP   = 1.0,
        Atsz     = 1.0,
        Aksz     = 1.0,
        Acib     = 1.0
    )

    # Benchmark with respect to 6 cosmological parameters
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

    println("\nBenchmarking Primal Evaluation...")
    t_primal = @benchmark $target_function($x0)
    push!(results, (backend="Primal (No Grad)", time=median(t_primal).time / 1e6, memory=t_primal.memory / 1024))

    println("Benchmarking ForwardDiff...")
    backend_fd = AutoForwardDiff()
    t_fd = @benchmark gradient($target_function, $backend_fd, $x0)
    push!(results, (backend="ForwardDiff", time=median(t_fd).time / 1e6, memory=t_fd.memory / 1024))

    println("Benchmarking Zygote...")
    backend_zg = AutoZygote()
    try
        # Pre-compile/warmup
        gradient(target_function, backend_zg, x0)
        t_zg = @benchmark gradient($target_function, $backend_zg, $x0)
        push!(results, (backend="Zygote", time=median(t_zg).time / 1e6, memory=t_zg.memory / 1024))
    catch e
        println("Zygote benchmark failed: $e")
        push!(results, (backend="Zygote", time=NaN, memory=0.0))
    end

    println("\nBenchmark Results (Median Time in ms):")
    df = DataFrame(results)
    println(df)
    
    return df
end

benchmark_ad_backends()
