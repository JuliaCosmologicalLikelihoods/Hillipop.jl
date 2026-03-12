"""
    nuisance.jl

Defines the typed hierarchy of nuisance parameters used by the Hillipop likelihood.
"""

Base.@kwdef struct HillipopCalibration{T<:Real}
    A_planck::T
    cal100A::T; cal100B::T
    cal143A::T; cal143B::T
    cal217A::T; cal217B::T
    pe100A::T;  pe100B::T
    pe143A::T;  pe143B::T
    pe217A::T;  pe217B::T
end

Base.@kwdef struct HillipopDust{T<:Real}
    AdustT::T;     AdustP::T
    beta_dustT::T; beta_dustP::T
end

Base.@kwdef struct HillipopSZ{T<:Real}
    Atsz::T
    Aksz::T
end

Base.@kwdef struct HillipopCIB{T<:Real}
    Acib::T
    beta_cib::T
    xi::T
end

Base.@kwdef struct HillipopPointSources{T<:Real}
    Aradio::T; beta_radio::T
    Adusty::T
end

Base.@kwdef struct HillipopSubPixel{T<:Real}
    Asbpx_100x100::T
    Asbpx_100x143::T
    Asbpx_100x217::T
    Asbpx_143x143::T
    Asbpx_143x217::T
    Asbpx_217x217::T
end

"""
    HillipopNuisance{T}

Statically-typed container for all Hillipop PR4 nuisance parameters.
Grouped into functional categories to aid clarity.
"""
Base.@kwdef struct HillipopNuisance{T<:Real}
    cal::HillipopCalibration{T}
    dust::HillipopDust{T}
    sz::HillipopSZ{T}
    cib::HillipopCIB{T}
    ps::HillipopPointSources{T}
    subpixel::HillipopSubPixel{T}
end

# Constructor that accepts a Dict or NamedTuple with flat parameters
function HillipopNuisance(d::Dict{Symbol, T}) where T<:Real
    cal = HillipopCalibration{T}(
        get(d, :A_planck, one(T)),
        get(d, :cal100A, one(T)), get(d, :cal100B, one(T)),
        get(d, :cal143A, one(T)), get(d, :cal143B, one(T)),
        get(d, :cal217A, one(T)), get(d, :cal217B, one(T)),
        get(d, :pe100A, one(T)),  get(d, :pe100B, one(T)),
        get(d, :pe143A, one(T)),  get(d, :pe143B, one(T)),
        get(d, :pe217A, one(T)),  get(d, :pe217B, one(T))
    )
    dust = HillipopDust{T}(
        get(d, :AdustT, zero(T)),     get(d, :AdustP, zero(T)),
        get(d, :beta_dustT, T(1.5)), get(d, :beta_dustP, T(1.5))
    )
    sz = HillipopSZ{T}(
        get(d, :Atsz, zero(T)),
        get(d, :Aksz, zero(T))
    )
    cib = HillipopCIB{T}(
        get(d, :Acib, zero(T)),
        get(d, :beta_cib, T(1.75)),
        get(d, :xi, zero(T))
    )
    ps = HillipopPointSources{T}(
        get(d, :Aradio, zero(T)), get(d, :beta_radio, T(-0.7)),
        get(d, :Adusty, zero(T))
    )
    subpixel = HillipopSubPixel{T}(
        get(d, :Asbpx_100x100, zero(T)),
        get(d, :Asbpx_100x143, zero(T)),
        get(d, :Asbpx_100x217, zero(T)),
        get(d, :Asbpx_143x143, zero(T)),
        get(d, :Asbpx_143x217, zero(T)),
        get(d, :Asbpx_217x217, zero(T))
    )
    return HillipopNuisance{T}(cal, dust, sz, cib, ps, subpixel)
end

# Support passing a Dict of mixed types (e.g. Dict{Symbol, Float64} mixed with Float32)
function HillipopNuisance(d::Dict{Symbol, <:Any})
    # Find a common type for promotion
    T = promote_type(typeof.(values(d))...)
    d_typed = Dict{Symbol, T}(k => T(v) for (k, v) in d)
    return HillipopNuisance(d_typed)
end

function HillipopNuisance(nt::NamedTuple)
    return HillipopNuisance(Dict{Symbol, Any}(pairs(nt)...))
end
