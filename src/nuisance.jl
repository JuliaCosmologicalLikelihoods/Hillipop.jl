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
    pe217A::T = 0.975; pe217B::T = 0.975
end

Base.@kwdef struct HillipopDust{T<:Real}
    AdustT::T = 1.0;     AdustP::T = 1.0
    beta_dustT::T = 1.51; beta_dustP::T = 1.59
end

Base.@kwdef struct HillipopSZ{T<:Real}
    Atsz::T = 1.0
    Aksz::T = 1.0
end

Base.@kwdef struct HillipopCIB{T<:Real}
    Acib::T = 1.0
    beta_cib::T = 1.75
    xi::T = 0.0
end

Base.@kwdef struct HillipopPointSources{T<:Real}
    Aradio::T = 1.0; beta_radio::T = -0.8
    Adusty::T = 1.0
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

# Unified constructor from any flat collection (NamedTuple, Pairs)
# This handles promotion and mapping to sub-structs
function HillipopNuisance(collection::Union{NamedTuple, Base.Iterators.Pairs})
    # Find common type for promotion
    vals = values(collection)
    T = isempty(vals) ? Float64 : promote_type(typeof.(vals)...)
    
    # Pre-generate defaults to avoid repeated allocation
    d_cal = HillipopCalibration{T}()
    d_dust = HillipopDust{T}()
    d_sz = HillipopSZ{T}()
    d_cib = HillipopCIB{T}()
    d_ps = HillipopPointSources{T}()
    d_sbpx = HillipopSubPixel{T}()

    # Calibration
    cal = HillipopCalibration{T}(
        T(get(collection, :A_planck, d_cal.A_planck)),
        T(get(collection, :cal100A, d_cal.cal100A)), T(get(collection, :cal100B, d_cal.cal100B)),
        T(get(collection, :cal143A, d_cal.cal143A)), T(get(collection, :cal143B, d_cal.cal143B)),
        T(get(collection, :cal217A, d_cal.cal217A)), T(get(collection, :cal217B, d_cal.cal217B)),
        T(get(collection, :pe100A, d_cal.pe100A)),  T(get(collection, :pe100B, d_cal.pe100B)),
        T(get(collection, :pe143A, d_cal.pe143A)),  T(get(collection, :pe143B, d_cal.pe143B)),
        T(get(collection, :pe217A, d_cal.pe217A)),  T(get(collection, :pe217B, d_cal.pe217B))
    )
    
    # Dust
    dust = HillipopDust{T}(
        T(get(collection, :AdustT, d_dust.AdustT)),     T(get(collection, :AdustP, d_dust.AdustP)),
        T(get(collection, :beta_dustT, d_dust.beta_dustT)), T(get(collection, :beta_dustP, d_dust.beta_dustP))
    )
    
    # SZ
    sz = HillipopSZ{T}(
        T(get(collection, :Atsz, d_sz.Atsz)),
        T(get(collection, :Aksz, d_sz.Aksz))
    )
    
    # CIB
    cib = HillipopCIB{T}(
        T(get(collection, :Acib, d_cib.Acib)),
        T(get(collection, :beta_cib, d_cib.beta_cib)),
        T(get(collection, :xi, d_cib.xi))
    )
    
    # PS
    ps = HillipopPointSources{T}(
        T(get(collection, :Aradio, d_ps.Aradio)), T(get(collection, :beta_radio, d_ps.beta_radio)),
        T(get(collection, :Adusty, d_ps.Adusty))
    )
    
    # Subpixel
    subpixel = HillipopSubPixel{T}(
        T(get(collection, :Asbpx_100x100, d_sbpx.Asbpx_100x100)),
        T(get(collection, :Asbpx_100x143, d_sbpx.Asbpx_100x143)),
        T(get(collection, :Asbpx_100x217, d_sbpx.Asbpx_100x217)),
        T(get(collection, :Asbpx_143x143, d_sbpx.Asbpx_143x143)),
        T(get(collection, :Asbpx_143x217, d_sbpx.Asbpx_143x217)),
        T(get(collection, :Asbpx_217x217, d_sbpx.Asbpx_217x217))
    )
    
    return HillipopNuisance{T}(cal, dust, sz, cib, ps, subpixel)
end
