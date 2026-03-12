"""
    foreground_dispatch.jl

Maps Hillipop nuisance parameters to CMBForegrounds.jl function calls.

This file handles:
1. Effective frequency lookup per foreground component
2. Mode→T/P routing for dust amplitude and spectral index
3. Cross-frequency pair → effective frequency mapping
4. Accumulation of all foreground D_ℓ contributions
"""

using CMBForegrounds
using DelimitedFiles

# ---------------------------------------------------------------------------
# Effective frequencies per foreground component (matching JAX fgmodel class)
# These map nominal frequencies (100, 143, 217 GHz) to effective band centers.
# ---------------------------------------------------------------------------
const EFF_FREQ_SZ    = Dict(100 => 100.24,  143 => 143.0,   217 => 222.044)
const EFF_FREQ_DUST  = Dict(100 => 105.2,   143 => 147.5,   217 => 228.1,  353 => 370.5)
const EFF_FREQ_CIB   = EFF_FREQ_DUST   # same effective frequencies
const EFF_FREQ_RADIO = Dict(100 => 100.4,   143 => 140.5,   217 => 218.6)

# Reference frequency for all foreground components (matches JAX f0 = 143)
const FG_FREQ_REF = 143.0

# FWHM per nominal frequency, used by sub-pixel model (arcmin)
const FWHM_ARCMIN = Dict(100 => 9.68, 143 => 7.30, 217 => 5.02)

# Dust reference frequency (353 GHz effective)
const DUST_FREQ_REF = EFF_FREQ_DUST[353]  # 370.5 GHz

# ll2pi shape: ℓ(ℓ+1)/(3000·3001), length lmax+1
function _ll2pi(lmax::Int)
    ells = 0:lmax
    return @. ells * (ells + 1) / (3000.0 * 3001.0)
end


# ---------------------------------------------------------------------------
# Cross-frequency pair helper
# Given the 6 Hillipop cross-frequency pairs (f1,f2), return the indices in
# the order they appear in the dust template columns.
# Dust template columns: 100x100, 100x143, 100x217, 143x143, 143x217, 217x217
# ---------------------------------------------------------------------------
const _DUST_PAIR_IDX = Dict(
    (100, 100) => 1,
    (100, 143) => 2,
    (100, 217) => 3,
    (143, 143) => 4,
    (143, 217) => 5,
    (217, 217) => 6,
)

function _dust_template_idx(f1::Int, f2::Int)
    k = f1 <= f2 ? (f1, f2) : (f2, f1)
    return _DUST_PAIR_IDX[k]
end


# ---------------------------------------------------------------------------
# Dust T/P amplitude dispatch
# ---------------------------------------------------------------------------
function _dust_AB(mode::String, pars::HillipopNuisance)
    if mode == "TT"
        return pars.dust.AdustT, pars.dust.AdustT, pars.dust.beta_dustT, pars.dust.beta_dustT
    elseif mode == "EE"
        return pars.dust.AdustP, pars.dust.AdustP, pars.dust.beta_dustP, pars.dust.beta_dustP
    elseif mode == "TE"
        return pars.dust.AdustT, pars.dust.AdustP, pars.dust.beta_dustT, pars.dust.beta_dustP
    else  # ET
        return pars.dust.AdustP, pars.dust.AdustT, pars.dust.beta_dustP, pars.dust.beta_dustT
    end
end


function _get_subpixel(subpixel::HillipopSubPixel, f1::Int, f2::Int)
    if f1 == 100 && f2 == 100 return subpixel.Asbpx_100x100
    elseif (f1 == 100 && f2 == 143) || (f1 == 143 && f2 == 100) return subpixel.Asbpx_100x143
    elseif (f1 == 100 && f2 == 217) || (f1 == 217 && f2 == 100) return subpixel.Asbpx_100x217
    elseif f1 == 143 && f2 == 143 return subpixel.Asbpx_143x143
    elseif (f1 == 143 && f2 == 217) || (f1 == 217 && f2 == 143) return subpixel.Asbpx_143x217
    elseif f1 == 217 && f2 == 217 return subpixel.Asbpx_217x217
    else return zero(subpixel.Asbpx_100x100)
    end
end

"""
    compute_foreground_dl(mode, xs, f1, f2, ell, pars, h)

Compute the total foreground D_ℓ contribution for cross-map-spectrum `xs`
in polarization mode `mode`, for nominal frequencies `f1` and `f2`.

Returns a `Vector{Float64}` of length `lmax+1`.

# Arguments
- `mode`: one of `"TT"`, `"EE"`, `"TE"`, `"ET"`
- `f1`, `f2`: nominal frequencies (Int, e.g. 100, 143, 217)
- `ell`: 0-based multipole vector
- `pars`: `Dict{Symbol,Float64}` of nuisance parameters
- `h`: `HillipopData` struct (for templates and lmax)
"""
function compute_foreground_dl(mode::String, f1::Int, f2::Int,
                               ell::AbstractVector, pars::HillipopNuisance{T_par},
                               h::HillipopData) where {T_par}
    lmax = h.lmax
    T = T_par
    
    # 1. Galactic Dust (template-based)
    tidx = _dust_template_idx(f1, f2)
    tmpl_dust = h.dust_templates[mode][tidx]
    A1, A2, β1, β2 = _dust_AB(mode, pars)
    res = dust_model_template_power(ell, tmpl_dust, A1, A2, β1, β2,
                                          Float64(EFF_FREQ_DUST[f1]),
                                          Float64(EFF_FREQ_DUST[f2]),
                                          DUST_FREQ_REF, 19.6)

    # 2. tSZ (TT only)
    if mode == "TT"
        ef1 = Float64(EFF_FREQ_SZ[f1])
        ef2 = Float64(EFF_FREQ_SZ[f2])
        res = res .+ tsz_cross_power(h.tsz_template, pars.sz.Atsz, ef1, ef2,
                              FG_FREQ_REF, 0.0, 3000.0, ell)
    end

    # 3. kSZ (TT only)
    if mode == "TT"
        res = res .+ ksz_template_scaled(h.ksz_template, pars.sz.Aksz)
    end

    # 4. Clustered CIB (TT only)
    if mode == "TT"
        ef1 = Float64(EFF_FREQ_CIB[f1])
        ef2 = Float64(EFF_FREQ_CIB[f2])
        s1 = cib_mbb_sed_weight(pars.cib.beta_cib, 25.0, FG_FREQ_REF, ef1)
        s2 = cib_mbb_sed_weight(pars.cib.beta_cib, 25.0, FG_FREQ_REF, ef2)
        res = res .+ (pars.cib.Acib * s1 * s2 .* h.cib_template)
    end

    # 5. SZ×CIB (TT only)
    if mode == "TT"
        ef_sz1  = Float64(EFF_FREQ_SZ[f1])
        ef_sz2  = Float64(EFF_FREQ_SZ[f2])
        ef_cib1 = Float64(EFF_FREQ_CIB[f1])
        ef_cib2 = Float64(EFF_FREQ_CIB[f2])
        tr1 = tsz_g_ratio(ef_sz1, FG_FREQ_REF, 2.72548)
        tr2 = tsz_g_ratio(ef_sz2, FG_FREQ_REF, 2.72548)
        cr1 = cib_mbb_sed_weight(pars.cib.beta_cib, 25.0, FG_FREQ_REF, ef_cib1)
        cr2 = cib_mbb_sed_weight(pars.cib.beta_cib, 25.0, FG_FREQ_REF, ef_cib2)
        xi_factor = -pars.cib.xi * sqrt(pars.cib.Acib * pars.sz.Atsz) * (tr2 * cr1 + tr1 * cr2)
        res = res .+ (xi_factor .* h.szxcib_template)
    end

    # 6. Radio Point Sources (TT only)
    if mode == "TT"
        res = res .+ radio_ps_power(ell, pars.ps.Aradio, pars.ps.beta_radio, 
                                   Float64(EFF_FREQ_RADIO[f1]), 
                                   Float64(EFF_FREQ_RADIO[f2]), FG_FREQ_REF)
    end

    # 7. Dusty Point Sources (TT only)
    if mode == "TT"
        res = res .+ dusty_ps_power(ell, pars.ps.Adusty, pars.cib.beta_cib,
                                   Float64(EFF_FREQ_CIB[f1]), 
                                   Float64(EFF_FREQ_CIB[f2]), FG_FREQ_REF, 25.0)
    end

    # 8. Sub-pixel effect (TT only)
    if mode == "TT"
        subpx_val = _get_subpixel(pars.subpixel, f1, f2)
        if subpx_val != 0
            res = res .+ sub_pixel_power(ell, subpx_val,
                                        Float64(FWHM_ARCMIN[f1]),
                                        Float64(FWHM_ARCMIN[f2]))
        end
    end

    return res
end

# Keep compute_foreground_dl! for backward compatibility or ForwardDiff performance,
# but internally use the non-mutating one if we want Zygote.
# Actually, let's just use the non-mutating one everywhere to be safe.
function compute_foreground_dl!(result::AbstractVector, mode::String, f1::Int, f2::Int,
                               ell::AbstractVector, pars::HillipopNuisance{T_par},
                               h::HillipopData) where {T_par}
    res = compute_foreground_dl(mode, f1, f2, ell, pars, h)
    result .= res
    return result
end


"""
    compute_fg_model(mode, pair_freqs, ell, pars, h)

Compute the foreground D_ℓ model for all 15 cross-map-pair spectra in the given mode.

# Arguments
- `mode`: `"TT"`, `"EE"`, `"TE"`, or `"ET"`
- `pair_freqs`: Vector of `(Int,Int)` nominal frequency pairs (length nxspec=15)
- `ell`: multipole vector 0..lmax
- `pars`: nuisance parameter dict
- `h`: HillipopData

# Returns
- `Vector{Vector{Float64}}` of length 15, each of length lmax+1
"""
function compute_fg_model(mode::String, pair_freqs::Vector{Tuple{Int,Int}},
                           ell::AbstractVector, pars::HillipopNuisance,
                           h::HillipopData)
    return [compute_foreground_dl(mode, f1, f2, ell, pars, h)
            for (f1, f2) in pair_freqs]
end
