using Profile
using Hillipop

h = load_hillipop()

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

lmax_in = h.lmax
ells = 2:lmax_in
ClTT = @. 6000.0 / (ells * (ells + 1)) * 1e-12
ClTE = fill(0.0, length(ells))
ClEE = fill(0.0, length(ells))

# Warmup
compute_loglike(ClTT, ClTE, ClEE, pars, h)

Profile.clear_malloc_data()
compute_loglike(ClTT, ClTE, ClEE, pars, h)
