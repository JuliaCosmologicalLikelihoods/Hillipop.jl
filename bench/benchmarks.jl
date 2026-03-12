using BenchmarkTools
using JET
using Hillipop

println("Loading Data...")
h = load_hillipop()

# Minimal parameter set (all nuisances at fiducial/unit values)
nt_base = (
    cal100A=1.0, cal100B=1.0, cal143A=1.0, cal143B=1.0, cal217A=1.0, cal217B=1.0,
    pe100A=1.0, pe100B=1.0, pe143A=1.0, pe143B=1.0, pe217A=1.0, pe217B=1.0,
    A_planck=1.0,
    AdustT=1.0, AdustP=1.0, beta_dustT=1.5, beta_dustP=1.5,
    Atsz=1.0, Aksz=1.0, Acib=1.0, beta_cib=1.75, xi=0.1,
    Aradio=0.0, beta_radio=-0.7, Adusty=0.0,
    Asbpx_100x100=0.0, Asbpx_100x143=0.0, Asbpx_100x217=0.0,
    Asbpx_143x143=0.0, Asbpx_143x217=0.0, Asbpx_217x217=0.0
)
pars = HillipopNuisance(nt_base)

# Flat (Zel'dovich) Cl spectra as a placeholder
lmax_in = h.lmax
ells = 2:lmax_in
ClTT = @. 6000.0 / (ells * (ells + 1)) * 1e-12
ClTE = fill(0.0, length(ells))
ClEE = fill(0.0, length(ells))

println("--- TOP LEVEL BENCHMARK ---")
b_total = @benchmark compute_loglike($ClTT, $ClTE, $ClEE, $pars, $h)
display(b_total)

println("\n\n--- PER-FUNCTION BENCHMARKS ---")

println("\n[1] _cl_to_dl")
b_cl_to_dl = @benchmark Hillipop._cl_to_dl($ClTT, $h.lmax)
display(b_cl_to_dl)

println("\n[2] compute_foreground_dl (TT, 100x100)")
ell_vec = collect(0:h.lmax)
b_fg_dl = @benchmark Hillipop.compute_foreground_dl("TT", 100, 100, $ell_vec, $pars, $h)
display(b_fg_dl)

println("\n[3] compute_fg_model (TT)")
mapnames = h.mapnames
frequencies = h.frequencies
pairs = Tuple{String,String,Int,Int}[]
n = length(mapnames)
for i in 1:n, j in i+1:n
    push!(pairs, (mapnames[i], mapnames[j], frequencies[i], frequencies[j]))
end
pair_freqs = [(f1, f2) for (_, _, f1, f2) in pairs]
b_fg_model = @benchmark Hillipop.compute_fg_model("TT", $pair_freqs, $ell_vec, $pars, $h)
display(b_fg_model)

println("\n[4] compute_residuals (TT)")
dlth = Hillipop._cl_to_dl(ClTT, h.lmax)
b_resid = @benchmark Hillipop.compute_residuals("TT", $dlth, $pars, $h)
display(b_resid)

println("\n[5] xspectra_to_xfreq (TT)")
R = Hillipop.compute_residuals("TT", dlth, pars, h)
nxfreq = length(unique(h.frequencies)) * (length(unique(h.frequencies)) + 1) ÷ 2
b_xspec = @benchmark Hillipop.xspectra_to_xfreq($R, $h.dlweight["TT"], $h.xspec2xfreq, $nxfreq)
display(b_xspec)

println("\n[6] select_spectra (TT)")
Rl = Hillipop.xspectra_to_xfreq(R, h.dlweight["TT"], h.xspec2xfreq, nxfreq)
b_select = @benchmark Hillipop.select_spectra($Rl, $h.lmins["TT"], $h.lmaxs["TT"], $nxfreq, $h.xspec2xfreq)
display(b_select)

println("\n[7] build_residual_vector")
b_build_vec = @benchmark Hillipop.build_residual_vector($ClTT, $ClTE, $ClEE, $pars, $h)
display(b_build_vec)

println("\n[8] compute_chi2")
Xl = Hillipop.build_residual_vector(ClTT, ClTE, ClEE, pars, h)
b_chi2 = @benchmark Hillipop.compute_chi2($Xl, $h.binning_matrix, $h.binned_invkll)
display(b_chi2)

println("\n\n--- TYPE STABILITY AUDIT ---")
using InteractiveUtils
println("\n[1] @code_warntype compute_loglike")
@code_warntype compute_loglike(ClTT, ClTE, ClEE, pars, h)

println("\n[2] JET @report_opt compute_loglike")
@report_opt compute_loglike(ClTT, ClTE, ClEE, pars, h)

println("\n[3] @code_warntype compute_residuals")
@code_warntype Hillipop.compute_residuals("TT", dlth, pars, h)

println("\n[4] JET @report_opt compute_residuals")
@report_opt Hillipop.compute_residuals("TT", dlth, pars, h)

println("\n[5] JET @report_opt compute_foreground_dl")
@report_opt Hillipop.compute_foreground_dl("TT", 100, 100, ell_vec, pars, h)

println("Done.")
