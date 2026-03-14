using Hillipop
using DifferentiationInterface
using Mooncake
using BenchmarkTools
using LinearAlgebra
using Statistics
using ADTypes

const DI = DifferentiationInterface
const backend = AutoMooncake(; config=nothing)

# 1. Setup realistic inputs
data_path = "/home/marcobonici/Desktop/work/CosmologicalEmulators/jax-loglike/data/planck_pr4_hillipop"
h = load_hillipop(data_path)

# Fiducial theory spectra (ℓ=2..lmax)
lmax = h.lmax
ClTT = ones(lmax-1) * 1e-12
ClEE = ones(lmax-1) * 1e-14
ClTE = ones(lmax-1) * 1e-13

# Fiducial nuisance parameters
pars_nt = (
    cal100A=1.0, cal100B=1.0, cal143A=1.0, cal143B=1.0, cal217A=1.0, cal217B=1.0,
    pe100A=1.0, pe100B=1.0, pe143A=1.0, pe143B=1.0, pe217A=1.0, pe217B=1.0,
    A_planck=1.0,
    AdustT=1.0, AdustP=1.0, beta_dustT=1.5, beta_dustP=1.5,
    Atsz=1.0, Aksz=1.0,
    Acib=1.0, beta_cib=2.5, xi=0.1,
    Aradio=1.0, beta_radio=-0.7,
    Adusty=1.0
)
pars = HillipopNuisance(pars_nt)

println("| Function | Output wrapped? | Prep time (s) | Median (ms) | Min (ms) | Allocs | Memory (KiB) |")
println("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

function benchmark_func(name, f_scalar, x0)
    # println("Profiling $name...")
    t_prep = 0.0
    try
        t_prep = @elapsed begin
            prep = DI.prepare_gradient(f_scalar, backend, x0)
        end
        
        suite = @benchmarkable DI.gradient($f_scalar, $prep, $backend, x) setup=(x = copy($x0))
        res = run(suite; samples=10, evals=1, seconds=30)
        
        println("| $name | yes | $(round(t_prep; digits=4)) | $(round(median(res).time/1e6; digits=4)) | $(round(minimum(res).time/1e6; digits=4)) | $(res.allocs) | $(round(res.memory/1024; digits=2)) |")
    catch e
        println("| $name | FAILED | - | - | - | - | - |")
        # @warn "Benchmark for $name failed" exception=e
    end
end

# 1. _cl_to_dl
benchmark_func("_cl_to_dl", x -> sum(Hillipop._cl_to_dl(x, lmax)), ClTT)

# 2. compute_foreground_dl
benchmark_func("compute_foreground_dl", 
    p -> sum(Hillipop.compute_foreground_dl("TT", 100, 143, 0:lmax, HillipopNuisance((cal100A=p[1], cal100B=1.0, cal143A=1.0, cal143B=1.0, cal217A=1.0, cal217B=1.0, pe100A=1.0, pe100B=1.0, pe143A=1.0, pe143B=1.0, pe217A=1.0, pe217B=1.0, A_planck=1.0, AdustT=p[2], AdustP=1.0, beta_dustT=p[3], beta_dustP=1.5, Atsz=p[4], Aksz=p[5], Acib=p[6], beta_cib=p[7], xi=p[8], Aradio=p[9], beta_radio=p[10], Adusty=p[11])), h)),
    [1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 2.5, 0.1, 1.0, -0.7, 1.0])

# 3. compute_residuals
benchmark_func("compute_residuals",
    x -> sum(Hillipop.compute_residuals("TT", x, pars, h)),
    Hillipop._cl_to_dl(ClTT, lmax))

# 4. xspectra_to_xfreq
R_tt = Hillipop.compute_residuals("TT", Hillipop._cl_to_dl(ClTT, lmax), pars, h)
benchmark_func("xspectra_to_xfreq",
    x -> sum(Hillipop.xspectra_to_xfreq(x, h.dlweight["TT"], h.xspec2xfreq, 6)),
    R_tt)

# 5. select_spectra
Rl_tt = Hillipop.xspectra_to_xfreq(R_tt, h.dlweight["TT"], h.xspec2xfreq, 6)
benchmark_func("select_spectra",
    x -> sum(Hillipop.select_spectra(x, h.lmins["TT"], h.lmaxs["TT"], 6, h.xspec2xfreq)),
    Rl_tt)

# 6. build_residual_vector
x_spectra = vcat(ClTT, ClTE, ClEE)
benchmark_func("build_residual_vector",
    x -> sum(Hillipop.build_residual_vector(x[1:lmax-1], x[lmax:2*lmax-2], x[2*lmax-1:3*lmax-3], pars, h)),
    x_spectra)

# 7. compute_chi2
Xl = Hillipop.build_residual_vector(ClTT, ClTE, ClEE, pars, h)
benchmark_func("compute_chi2",
    x -> Hillipop.compute_chi2(x, h.binning_matrix, h.binned_invkll),
    Xl)

# 8. compute_loglike
benchmark_func("compute_loglike",
    x -> compute_loglike(x[1:lmax-1], x[lmax:2*lmax-2], x[2*lmax-1:3*lmax-3], pars, h),
    x_spectra)
