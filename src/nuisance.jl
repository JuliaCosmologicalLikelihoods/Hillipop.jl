"""
    nuisance.jl

Defines the typed hierarchy of nuisance parameters used by the Hillipop likelihood.
"""

Base.@kwdef struct HillipopCalibration{T<:Real}
    A_planck::T = 1.0
    cal100A::T = 1.0; cal100B::T = 1.0
    cal143A::T = 1.0; cal143B::T = 1.0
    cal217A::T = 1.0; cal217B::T = 1.0
    pe100A::T = 1.0;  pe100B::T = 1.0
    pe143A::T = 1.0;  pe143B::T = 1.0
    pe217A::T = 1.0;  pe217B::T = 1.0
end

Base.@kwdef struct HillipopDust{T<:Real}
    AdustT::T = 0.0;     AdustP::T = 0.0
    beta_dustT::T = 1.5; beta_dustP::T = 1.5
end

Base.@kwdef struct HillipopSZ{T<:Real}
    Atsz::T = 0.0
    Aksz::T = 0.0
end

Base.@kwdef struct HillipopCIB{T<:Real}
    Acib::T = 0.0
    beta_cib::T = 1.75
    xi::T = 0.0
end

Base.@kwdef struct HillipopPointSources{T<:Real}
    Aradio::T = 0.0; beta_radio::T = -0.7
    Adusty::T = 0.0
end

Base.@kwdef struct HillipopSubPixel{T<:Real}
    Asbpx_100x100::T = 0.0
    Asbpx_100x143::T = 0.0
    Asbpx_100x217::T = 0.0
    Asbpx_143x143::T = 0.0
    Asbpx_143x217::T = 0.0
    Asbpx_217x217::T = 0.0
end

"""
    HillipopNuisance{T}

Statically-typed container for all Hillipop PR4 nuisance parameters.
Grouped into functional categories to aid clarity.
"""
struct HillipopNuisance{T<:Real}
    cal::HillipopCalibration{T}
    dust::HillipopDust{T}
    sz::HillipopSZ{T}
    cib::HillipopCIB{T}
    ps::HillipopPointSources{T}
    subpixel::HillipopSubPixel{T}
end

# Inner-struct default constructor helper
function HillipopNuisance{T}() where T<:Real
    return HillipopNuisance{T}(
        HillipopCalibration{T}(),
        HillipopDust{T}(),
        HillipopSZ{T}(),
        HillipopCIB{T}(),
        HillipopPointSources{T}(),
        HillipopSubPixel{T}()
    )
end

function HillipopNuisance()
    return HillipopNuisance{Float64}()
end

# Unified constructor from any flat collection (Dict, NamedTuple, Pairs)
# This handles promotion and mapping to sub-structs
function HillipopNuisance(collection::Union{Dict{Symbol, <:Any}, NamedTuple, Base.Iterators.Pairs})
    # Find common type for promotion
    vals = values(collection)
    T = isempty(vals) ? Float64 : promote_type(typeof.(vals)...)
    
    # Calibration
    cal = HillipopCalibration{T}(
        T(get(collection, :A_planck, 1.0)),
        T(get(collection, :cal100A, 1.0)), T(get(collection, :cal100B, 1.0)),
        T(get(collection, :cal143A, 1.0)), T(get(collection, :cal143B, 1.0)),
        T(get(collection, :cal217A, 1.0)), T(get(collection, :cal217B, 1.0)),
        T(get(collection, :pe100A, 1.0)),  T(get(collection, :pe100B, 1.0)),
        T(get(collection, :pe143A, 1.0)),  T(get(collection, :pe143B, 1.0)),
        T(get(collection, :pe217A, 1.0)),  T(get(collection, :pe217B, 1.0))
    )
    
    # Dust
    dust = HillipopDust{T}(
        T(get(collection, :AdustT, 0.0)),     T(get(collection, :AdustP, 0.0)),
        T(get(collection, :beta_dustT, 1.5)), T(get(collection, :beta_dustP, 1.5))
    )
    
    # SZ
    sz = HillipopSZ{T}(
        T(get(collection, :Atsz, 0.0)),
        T(get(collection, :Aksz, 0.0))
    )
    
    # CIB
    cib = HillipopCIB{T}(
        T(get(collection, :Acib, 0.0)),
        T(get(collection, :beta_cib, 1.75)),
        T(get(collection, :xi, 0.0))
    )
    
    # PS
    ps = HillipopPointSources{T}(
        T(get(collection, :Aradio, 0.0)), T(get(collection, :beta_radio, -0.7)),
        T(get(collection, :Adusty, 0.0))
    )
    
    # Subpixel
    subpixel = HillipopSubPixel{T}(
        T(get(collection, :Asbpx_100x100, 0.0)),
        T(get(collection, :Asbpx_100x143, 0.0)),
        T(get(collection, :Asbpx_100x217, 0.0)),
        T(get(collection, :Asbpx_143x143, 0.0)),
        T(get(collection, :Asbpx_143x217, 0.0)),
        T(get(collection, :Asbpx_217x217, 0.0))
    )
    
    return HillipopNuisance{T}(cal, dust, sz, cib, ps, subpixel)
end
