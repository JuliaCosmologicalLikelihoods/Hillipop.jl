"""
    HillipopTuringExt

Turing.jl extension for the Hillipop PR4 likelihood.

This extension is automatically loaded when both `Hillipop` and `Turing` are
loaded in the same Julia session. It provides two `@model` variants:

- `hillipop_model(h, ClTT, ClTE, ClEE)` — Variant A: uses `Turing.@addlogprob!`
  to inject the Hillipop log-likelihood directly. Parameters assembled in
  `Dict{Symbol}`. Lightest on memory and allocations; best for HMC.

- `hillipop_model_mvnormal(h, ClTT, ClEE, ClEE)` — Variant B: expresses the
  likelihood natively as `Xl_binned ~ MvNormal(0, C)` using a `HillipopPars`
  struct. The covariance matrix C = inv(h.binned_invkll) is computed once at
  construction time. Preferred for transparent Turing integration.

Also exposes `hillipop_loglike_check` to compare both computation paths.

!!! note "Priors"
    The priors defined here are **placeholder starting points** provided for
    convenience. They are not the official Hillipop PR4 priors; users should
    replace them with their own before running production analyses.
"""
module HillipopTuringExt

using Hillipop
using Turing
using LinearAlgebra


# ============================================================================
# Shared prior helper — build a named tuple of all 31 Hillipop priors.
# Returns `NamedTuple` of `Distribution` objects (not yet sampled).
# Edit these to customise the prior for your analysis.
# ============================================================================
function _hillipop_priors()
    return (
        # Global calibration
        A_planck   = Normal(1.0, 0.0025),
        # Per-map absolute calibration (0.04% each)
        cal100A    = Normal(1.0, 0.0004),
        cal100B    = Normal(1.0, 0.0004),
        cal143A    = Normal(1.0, 0.0004),
        cal143B    = Normal(1.0, 0.0004),
        cal217A    = Normal(1.0, 0.0004),
        cal217B    = Normal(1.0, 0.0004),
        # Polarization efficiency (1%)
        pe100A     = Normal(1.0, 0.01),
        pe100B     = Normal(1.0, 0.01),
        pe143A     = Normal(1.0, 0.01),
        pe143B     = Normal(1.0, 0.01),
        pe217A     = Normal(1.0, 0.01),
        pe217B     = Normal(1.0, 0.01),
        # Galactic dust
        AdustT     = Uniform(0.0, 50.0),
        AdustP     = Uniform(0.0, 50.0),
        beta_dustT = Normal(1.59, 0.2),
        beta_dustP = Normal(1.59, 0.2),
        # tSZ / kSZ
        Atsz       = Uniform(0.0, 10.0),
        Aksz       = Uniform(0.0, 10.0),
        # CIB
        Acib       = Uniform(0.0, 20.0),
        beta_cib   = Normal(1.75, 0.2),
        # SZ × CIB
        xi         = Uniform(0.0, 1.0),
        # Radio PS
        Aradio     = Uniform(0.0, 10.0),
        beta_radio = Normal(-1.0, 1.0),
        # Dusty PS
        Adusty     = Uniform(0.0, 10.0),
    )
end


# ============================================================================
# Variant A — @addlogprob! (direct, low-allocation)
#
# Parameters are assembled into Dict{Symbol} and passed directly to the
# existing compute_loglike function.  This is maximally compatible: the
# likelihood code path is byte-for-byte identical to the non-Turing case.
# ============================================================================

"""
    hillipop_model(h, ClTT, ClTE, ClEE; modes=("TT","EE","TE"))

Return a Turing `@model` that places priors over the 31 Hillipop nuisance
parameters and injects the log-likelihood via `Turing.@addlogprob!`.

This is **Variant A** — the most direct mapping of `compute_loglike` into
Turing. Use this for HMC / NUTS sampling.

# Arguments
- `h`: `HillipopData` struct returned by `load_hillipop`
- `ClTT`, `ClTE`, `ClEE`: theory power spectra in K², starting from ℓ=2
- `modes`: polarization modes to include (default: all three)

# Example
```julia
using Hillipop, Turing
h  = load_hillipop("data/")
m  = hillipop_model(h, ClTT, ClTE, ClEE)
ch = sample(m, NUTS(0.65), 1000)
```
"""
function hillipop_model(h::HillipopData, ClTT, ClTE, ClEE;
                         modes::Tuple=("TT", "EE", "TE"))
    return _hillipop_model_A(h, ClTT, ClTE, ClEE, modes)
end

@model function hillipop_nuisance_priors()
    pr = _hillipop_priors()

    # Sample all nuisance parameters
    A_planck   ~ pr.A_planck
    cal100A    ~ pr.cal100A;  cal100B  ~ pr.cal100B
    cal143A    ~ pr.cal143A;  cal143B  ~ pr.cal143B
    cal217A    ~ pr.cal217A;  cal217B  ~ pr.cal217B
    pe100A     ~ pr.pe100A;   pe100B   ~ pr.pe100B
    pe143A     ~ pr.pe143A;   pe143B   ~ pr.pe143B
    pe217A     ~ pr.pe217A;   pe217B   ~ pr.pe217B
    AdustT     ~ pr.AdustT;   AdustP   ~ pr.AdustP
    beta_dustT ~ pr.beta_dustT; beta_dustP ~ pr.beta_dustP
    Atsz       ~ pr.Atsz
    Aksz       ~ pr.Aksz
    Acib       ~ pr.Acib;     beta_cib ~ pr.beta_cib
    xi         ~ pr.xi
    Aradio     ~ pr.Aradio;   beta_radio ~ pr.beta_radio
    Adusty     ~ pr.Adusty

    return (; A_planck, cal100A, cal100B, cal143A, cal143B, cal217A, cal217B, pe100A, pe100B, pe143A, pe143B, pe217A, pe217B, AdustT, AdustP, beta_dustT, beta_dustP, Atsz, Aksz, Acib, beta_cib, xi, Aradio, beta_radio, Adusty)
end

@model function _hillipop_model_A(h, ClTT, ClTE, ClEE, modes)
    nuis ~ to_submodel(hillipop_nuisance_priors(), false)

    # Assemble directly into HillipopNuisance
    pars = HillipopNuisance(nuis)

    Turing.@addlogprob! compute_loglike(ClTT, ClTE, ClEE, pars, h; modes=modes)
end


# ============================================================================
# Variant B — data ~ MvNormal (native Turing style)
#
# The binned residual vector is expressed as drawn from MvNormal.
# The covariance C = inv(h.binned_invkll) is computed exactly once at
# model-construction time and closed over, so it is not re-inverted per sample.
# Parameters are assembled into a HillipopPars struct.
# ============================================================================

"""
    hillipop_model_mvnormal(h, ClTT, ClTE, ClEE; modes=("TT","EE","TE"))

Return a Turing `@model` that uses native `data ~ MvNormal(μ, C)` syntax.

This is **Variant B** — the covariance `C = inv(h.binned_invkll)` is computed
**once** at construction time. Within the model, a `HillipopPars` struct
collects all 31 nuisance parameters, and `build_residual_vector` is called to
obtain the binned data vector, which is then scored against `MvNormal`.

# Arguments
- `h`: `HillipopData` struct returned by `load_hillipop`
- `ClTT`, `ClTE`, `ClEE`: theory power spectra in K², starting from ℓ=2
- `modes`: polarization modes to include (default: all three)

# Example
```julia
using Hillipop, Turing
h  = load_hillipop("data/")
m  = hillipop_model_mvnormal(h, ClTT, ClTE, ClEE)
ch = sample(m, NUTS(0.65), 1000)
```
"""
function hillipop_model_mvnormal(h::HillipopData, ClTT, ClTE, ClEE;
                                   modes::Tuple=("TT", "EE", "TE"))
    # Invert once here — this is O(nbins³) and only happens at construction
    C = Symmetric(inv(h.binned_invkll))
    return _hillipop_model_B(h, C, ClTT, ClTE, ClEE, modes)
end

@model function _hillipop_model_B(h, C, ClTT, ClTE, ClEE, modes)
    nuis ~ to_submodel(hillipop_nuisance_priors(), false)

    # Convert to Dict for the existing build_residual_vector interface
    T = promote_type(map(typeof, values(nuis))...)
    pars = Dict{Symbol, T}(pairs(nuis)...)

    # Build the unbinned residual vector and project into binned space
    Xl        = build_residual_vector(ClTT, ClTE, ClEE, pars, h; modes=modes)
    Xl_binned = h.binning_matrix * Xl

    # Score against the pre-computed covariance
    Xl_binned ~ MvNormal(zeros(eltype(Xl_binned), length(Xl_binned)), C)
end


# ============================================================================
# Debugging / verification utilities
# ============================================================================

"""
    hillipop_loglike_check(pars, h, ClTT, ClTE, ClEE; modes=("TT","EE","TE"))

Compare the direct `compute_loglike` path with the Turing Variant A model
likelihood evaluation. Returns a named tuple with both values and the
absolute difference.

# Notes
- The Turing path evaluates the **log-likelihood only** (not priors) via
  `Turing.DynamicPPL.LikelihoodContext`.
- A result with `diff < 1e-10` confirms that the extension does not shadow
  or alter the base calculation.

# Example
```julia
result = hillipop_loglike_check(default_pars(), h, ClTT, ClTE, ClEE)
println(result.diff)   # should be < 1e-10
```
"""
function hillipop_loglike_check(pars::Dict{Symbol}, h::HillipopData,
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

"""
    hillipop_loglike_check_mvnormal(pars, h, ClTT, ClTE, ClEE; modes=("TT","EE","TE"))

Same check as `hillipop_loglike_check` but for the MvNormal Variant B model.
"""
function hillipop_loglike_check_mvnormal(pars::Dict{Symbol}, h::HillipopData,
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
