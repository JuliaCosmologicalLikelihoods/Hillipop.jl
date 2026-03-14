using Hillipop
using DifferentiationInterface
using Mooncake
using BenchmarkTools
using LinearAlgebra
using Statistics
using ADTypes
using ForwardDiff

const DI = DifferentiationInterface
const backend = AutoMooncake(; config=nothing)

# 1. Setup realistic inputs
data_path = "/home/marcobonici/Desktop/work/CosmologicalEmulators/jax-loglike/data/planck_pr4_hillipop"
h = load_hillipop(data_path)

lmax = h.lmax
ClTT = ones(lmax-1) * 1e-12
ClEE = ones(lmax-1) * 1e-14
ClTE = ones(lmax-1) * 1e-13

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

R_tt = Hillipop.compute_residuals("TT", Hillipop._cl_to_dl(ClTT, lmax), pars, h)
Rl_tt = Hillipop.xspectra_to_xfreq(R_tt, h.dlweight["TT"], h.xspec2xfreq, 6)

f_xspectra(x) = sum(Hillipop.xspectra_to_xfreq_unnormed(x, h.dlweight["TT"], h.xspec2xfreq, 6)[1])
f_select(x) = sum(Hillipop.select_spectra(x, h.lmins["TT"], h.lmaxs["TT"], 6, h.xspec2xfreq))

# Verify correctness vs ForwardDiff
println("Verifying correctness...")
grad_mc_xspectra = DI.gradient(f_xspectra, AutoMooncake(config=nothing), R_tt)
grad_fd_xspectra = DI.gradient(f_xspectra, AutoForwardDiff(), R_tt)
println("xspectra_to_xfreq error: ", maximum(abs, grad_mc_xspectra .- grad_fd_xspectra))

grad_mc_select = DI.gradient(f_select, AutoMooncake(config=nothing), Rl_tt)
grad_fd_select = DI.gradient(f_select, AutoForwardDiff(), Rl_tt)
println("select_spectra error: ", maximum(abs, grad_mc_select .- grad_fd_select))

# Benchmark
function run_bench(name, f_scalar, x0)
    prep = DI.prepare_gradient(f_scalar, backend, x0)
    suite = @benchmarkable DI.gradient($f_scalar, $prep, $backend, x) setup=(x = copy($x0))
    res = run(suite; samples=200, evals=1, seconds=60)
    println("| $name | $(round(median(res).time/1e6; digits=4)) | $(round(minimum(res).time/1e6; digits=4)) | $(res.allocs) | $(round(res.memory/1024; digits=2)) |")
end

println("| Function | Median (ms) | Min (ms) | Allocs | Memory (KiB) |")
println("| :--- | :--- | :--- | :--- | :--- |")
run_bench("xspectra_to_xfreq_unnormed", f_xspectra, R_tt)
run_bench("select_spectra", f_select, Rl_tt)

x_spectra = vcat(ClTT, ClTE, ClEE)
f_loglike(x) = compute_loglike(x[1:lmax-1], x[lmax:2*lmax-2], x[2*lmax-1:3*lmax-3], pars, h)
run_bench("compute_loglike", f_loglike, x_spectra)
