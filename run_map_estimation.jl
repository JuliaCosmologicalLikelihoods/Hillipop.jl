using Hillipop
using Capse
using AbstractCosmologicalEmulators
using Turing
using Optim
using Artifacts
using LinearAlgebra
using ForwardDiff
using Printf
using ADTypes
using LogDensityProblems
using LogDensityProblemsAD
using DynamicPPL

# 1. Setup
println("Loading data and emulators...")
const h = load_hillipop(lmax=2500)
const lmax_h = h.lmax
const artifacts_toml = joinpath(dirname(pathof(Capse)), "..", "Artifacts.toml")
const path = artifact_path(artifact_hash("CAMB_LCDM", artifacts_toml))
const emu_TT = Capse.load_emulator(joinpath(path, "TT/"), emu=LuxEmulator)
const emu_TE = Capse.load_emulator(joinpath(path, "TE/"), emu=LuxEmulator)
const emu_EE = Capse.load_emulator(joinpath(path, "EE/"), emu=LuxEmulator)
const ell = emu_TT.ℓgrid[3:5001]
const idx_lmax = findfirst(==(lmax_h), ell)
const fac = @. 2π / (ell[1:idx_lmax] * (ell[1:idx_lmax] + 1)) * 1e-12

@model function full_hillipop_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac)
    ln10As ~ Uniform(2.0, 4.0); ns ~ Uniform(0.84, 1.1); H0 ~ Uniform(60.0, 82.0)
    omega_b ~ Uniform(0.019, 0.026); omega_cdm ~ Uniform(0.05, 0.255); tau_reio ~ Uniform(0.02, 0.08)
    A_planck ~ Truncated(Normal(1.0, 0.0025), 0.9, 1.1)
    cal100A ~ Uniform(0.9, 1.1); cal100B ~ Uniform(0.9, 1.1); cal143B ~ Uniform(0.9, 1.1)
    cal217A ~ Uniform(0.9, 1.1); cal217B ~ Uniform(0.9, 1.1)
    AdustT ~ Truncated(Normal(1.0, 0.1), 0.5, 1.5); AdustP ~ Truncated(Normal(1.0, 0.1), 0.7, 1.3)
    beta_dustT ~ Truncated(Normal(1.51, 0.01), 1.4, 1.6); beta_dustP ~ Truncated(Normal(1.59, 0.01), 1.5, 1.7)
    Atsz ~ Uniform(0.0, 50.0); Aksz ~ Uniform(0.0, 50.0)
    Acib ~ Uniform(0.0, 20.0); beta_cib ~ Truncated(Normal(1.75, 0.06), 1.6, 1.9)
    xi ~ Uniform(-1.0, 1.0); Aradio ~ Uniform(0.0, 150.0); Adusty ~ Uniform(0.0, 150.0)

    x_cosmo = [ln10As, ns, H0, omega_b, omega_cdm, tau_reio]
    Cl_TT = Capse.get_Cℓ(x_cosmo, emu_TT)[1:idx_lmax] .* fac
    Cl_TE = Capse.get_Cℓ(x_cosmo, emu_TE)[1:idx_lmax] .* fac
    Cl_EE = Capse.get_Cℓ(x_cosmo, emu_EE)[1:idx_lmax] .* fac

    pars = HillipopNuisance((
        A_planck=A_planck, cal100A=cal100A, cal100B=cal100B, cal143B=cal143B, 
        cal217A=cal217A, cal217B=cal217B, AdustT=AdustT, AdustP=AdustP, 
        beta_dustT=beta_dustT, beta_dustP=beta_dustP, Atsz=Atsz, Aksz=Aksz, 
        Acib=Acib, beta_cib=beta_cib, xi=xi, Aradio=Aradio, Adusty=Adusty
    ))
    Turing.@addlogprob! compute_loglike(Cl_TT, Cl_TE, Cl_EE, pars, h)
end

model = full_hillipop_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac)

# LogDensityProblems interface
t_problem = Turing.LogDensityFunction(model)
ad_problem = ADgradient(AutoForwardDiff(), t_problem)

function logp_func(x)
    lp = LogDensityProblems.logdensity(ad_problem, x)
    return -lp
end

function grad_func!(g, x)
    lp, grad = LogDensityProblems.logdensity_and_gradient(ad_problem, x)
    g .= -grad
    return -lp
end

results = []
println("\nStarting 10 MAP estimations using direct Optim + LogDensityProblems...")

for i in 1:10
    try
        # Start from prior draw
        vi = DynamicPPL.VarInfo(model)
        DynamicPPL.link!(vi, DynamicPPL.SampleFromPrior())
        model(vi, DynamicPPL.SampleFromPrior())
        init_theta = vi[DynamicPPL.SampleFromPrior()]
        
        res = Optim.optimize(logp_func, grad_func!, init_theta, LBFGS(m=50), Optim.Options(iterations=200))
        push!(results, res)
        @printf("Run %2d/10: Final logp = %.4f (Converged: %s)\n", i, -res.minimum, Optim.converged(res))
    catch e
        println("Run $i failed.")
    end
end

if !isempty(results)
    lps = [-r.minimum for r in results]
    best_idx = argmax(lps)
    best_theta = results[best_idx].minimizer
    
    vi = DynamicPPL.VarInfo(model)
    vi[DynamicPPL.SampleFromPrior()] = best_theta
    DynamicPPL.invlink!(vi, DynamicPPL.SampleFromPrior())
    vals = vi[DynamicPPL.SampleFromPrior()]
    
    # Get parameter names from a dummy sample
    ch = sample(model, Prior(), 1)
    p_names = names(ch, :parameters)

    println("\n" * "="^50)
    @printf("Best log-posterior: %.4f\n", -results[best_idx].minimum)
    println("="^50)
    println("\nBest Parameters:")
    for (j, name) in enumerate(p_names)
        @printf("  %-12s: %.6f\n", string(name), vals[j])
    end
    
    @printf("\nSpread in logp across runs: %.4f\n", maximum(lps) - minimum(lps))
end
