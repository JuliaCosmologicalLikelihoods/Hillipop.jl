"""
    HillipopTuringExt

Turing.jl extension for the Hillipop PR4 likelihood.

This extension is automatically loaded when both `Hillipop` and `Turing` are
loaded in the same Julia session. It provides two `@model` variants:

- `hillipop_model(h, ClTT, ClTE, ClEE)` — Variant A: uses `Turing.@addlogprob!`
  to inject the Hillipop log-likelihood directly.

- `hillipop_model_mvnormal(h, ClTT, ClEE, ClEE)` — Variant B: expresses the
  likelihood natively as `Xl_binned ~ MvNormal(0, C)`.

!!! note "Priors"
    The priors defined here match the Planck PR4 HiLLiPoP configuration
    used in the original JAX-based implementation.
"""
module HillipopTuringExt

using Hillipop
using Turing
using LinearAlgebra


# ============================================================================
# Shared prior helper — build a named tuple of all Hillipop priors.
# These match the JAX yaml configuration exactly.
# ============================================================================
function _hillipop_priors()
    return (
        # Global calibration
        A_planck   = Truncated(Normal(1.0, 0.0025), 0.9, 1.1),
        # Per-map absolute calibration (Flat priors)
        cal100A    = Uniform(0.9, 1.1),
        cal100B    = Uniform(0.9, 1.1),
        cal143B    = Uniform(0.9, 1.1),
        cal217A    = Uniform(0.9, 1.1),
        cal217B    = Uniform(0.9, 1.1),
        # Galactic dust (Gaussian + Bounds)
        AdustT     = Truncated(Normal(1.0, 0.1), 0.5, 1.5),
        AdustP     = Truncated(Normal(1.0, 0.1), 0.7, 1.3),
        beta_dustT = Truncated(Normal(1.51, 0.01), 1.4, 1.6),
        beta_dustP = Truncated(Normal(1.59, 0.01), 1.5, 1.7),
        # tSZ / kSZ
        Atsz       = Uniform(0.0, 50.0),
        Aksz       = Uniform(0.0, 50.0),
        # CIB
        Acib       = Uniform(0.0, 20.0),
        beta_cib   = Truncated(Normal(1.75, 0.06), 1.6, 1.9),
        # SZ × CIB
        xi         = Uniform(-1.0, 1.0),
        # Radio PS
        Aradio     = Uniform(0.0, 150.0),
        # Dusty PS
        Adusty     = Uniform(0.0, 150.0),
    )
end


# ============================================================================
# Variant A — @addlogprob! (direct, low-allocation)
# ============================================================================

"""
    hillipop_model(h, ClTT, ClTE, ClEE; modes=("TT","EE","TE"))

Return a Turing `@model` that places priors over the Hillipop nuisance
parameters and injects the log-likelihood via `Turing.@addlogprob!`.
"""
function hillipop_model(h::HillipopData, ClTT, ClTE, ClEE;
                         modes::Tuple=("TT", "EE", "TE"))
    return _hillipop_model_A(h, ClTT, ClTE, ClEE, modes)
end

@model function hillipop_nuisance_priors()
    pr = _hillipop_priors()

    # Variable parameters (Matching JAX)
    A_planck   ~ pr.A_planck
    cal100A    ~ pr.cal100A;  cal100B  ~ pr.cal100B
    cal143B    ~ pr.cal143B
    cal217A    ~ pr.cal217A;  cal217B  ~ pr.cal217B
    
    AdustT     ~ pr.AdustT;   AdustP   ~ pr.AdustP
    beta_dustT ~ pr.beta_dustT; beta_dustP ~ pr.beta_dustP
    
    Atsz       ~ pr.Atsz
    Aksz       ~ pr.Aksz
    
    Acib       ~ pr.Acib;     beta_cib ~ pr.beta_cib
    xi         ~ pr.xi
    
    Aradio     ~ pr.Aradio
    Adusty     ~ pr.Adusty

    # Note: cal143A, peXXX, and beta_radio are not sampled and 
    # will fall back to struct defaults (1.0, 0.975, -0.8) 
    # in the HillipopNuisance constructor.

    return (; A_planck, cal100A, cal100B, cal143B, cal217A, cal217B, AdustT, AdustP, beta_dustT, beta_dustP, Atsz, Aksz, Acib, beta_cib, xi, Aradio, Adusty)
end

@model function _hillipop_model_A(h, ClTT, ClTE, ClEE, modes)
    nuis ~ to_submodel(hillipop_nuisance_priors(), false)

    # Assemble directly into HillipopNuisance (defaults handle fixed pars)
    pars = HillipopNuisance(nuis)

    Turing.@addlogprob! compute_loglike(ClTT, ClTE, ClEE, pars, h; modes=modes)
end


# ============================================================================
# Variant B — data ~ MvNormal (native Turing style)
# ============================================================================

"""
    hillipop_model_mvnormal(h, ClTT, ClTE, ClEE; modes=("TT","EE","TE"))

Return a Turing `@model` that uses native `data ~ MvNormal(μ, C)` syntax.
"""
function hillipop_model_mvnormal(h::HillipopData, ClTT, ClTE, ClEE;
                                   modes::Tuple=("TT", "EE", "TE"))
    # Invert once here — this is O(nbins³) and only happens at construction
    C = Symmetric(inv(h.binned_invkll))
    return _hillipop_model_B(h, C, ClTT, ClTE, ClEE, modes)
end

@model function _hillipop_model_B(h, C, ClTT, ClTE, ClEE, modes)
    nuis ~ to_submodel(hillipop_nuisance_priors(), false)

    # Convert to HillipopNuisance for the build_residual_vector interface
    pars = HillipopNuisance(nuis)

    # Build the unbinned residual vector and project into binned space
    Xl        = build_residual_vector(ClTT, ClTE, ClEE, pars, h; modes=modes)
    Xl_binned = h.binning_matrix * Xl

    # Score against the pre-computed covariance
    Xl_binned ~ MvNormal(zeros(eltype(Xl_binned), length(Xl_binned)), C)
end


# ============================================================================
# Debugging / verification utilities
# ============================================================================

function hillipop_loglike_check(pars::HillipopNuisance, h::HillipopData,
                                  ClTT, ClTE, ClEE;
                                  modes::Tuple=("TT", "EE", "TE"))
    logL_direct = compute_loglike(ClTT, ClTE, ClEE, pars, h; modes=modes)

    model = hillipop_model(h, ClTT, ClTE, ClEE; modes=modes)
    # Evaluate only the likelihood contribution (no prior)
    ctx   = Turing.DynamicPPL.LikelihoodContext()
    _, vi = Turing.DynamicPPL.evaluate!!(model, Turing.DynamicPPL.VarInfo(model), ctx)
    logL_turing = Turing.DynamicPPL.getlogp(vi)

    return (direct = logL_direct,
            turing = logL_turing,
            diff   = abs(logL_direct - logL_turing))
end

function hillipop_loglike_check_mvnormal(pars::HillipopNuisance, h::HillipopData,
                                           ClTT, ClTE, ClEE;
                                           modes::Tuple=("TT", "EE", "TE"))
    logL_direct = compute_loglike(ClTT, ClTE, ClEE, pars, h; modes=modes)

    model = hillipop_model_mvnormal(h, ClTT, ClTE, ClEE; modes=modes)
    ctx   = Turing.DynamicPPL.LikelihoodContext()
    _, vi = Turing.DynamicPPL.evaluate!!(model, Turing.DynamicPPL.VarInfo(model), ctx)
    logL_turing = Turing.DynamicPPL.getlogp(vi)

    return (direct = logL_direct,
            turing = logL_turing,
            diff   = abs(logL_direct - logL_turing))
end

end # module HillipopTuringExt
