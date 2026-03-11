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
function _dust_AB(mode::String, pars)
    if mode == "TT"
        return pars[:AdustT], pars[:AdustT], pars[:beta_dustT], pars[:beta_dustT]
    elseif mode == "EE"
        return pars[:AdustP], pars[:AdustP], pars[:beta_dustP], pars[:beta_dustP]
    elseif mode == "TE"
        return pars[:AdustT], pars[:AdustP], pars[:beta_dustT], pars[:beta_dustP]
    else  # ET
        return pars[:AdustP], pars[:AdustT], pars[:beta_dustP], pars[:beta_dustT]
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
                               ell::AbstractVector, pars::Dict{Symbol},
                               h::HillipopData)
    lmax = h.lmax
    T_par = valtype(pars)
    result = zeros(T_par, lmax + 1)
    ll2pi = _ll2pi(lmax)

    # ------------------------------------------------------------------
    # 1. Galactic Dust (template-based)
    # ------------------------------------------------------------------
    tidx = _dust_template_idx(f1, f2)
    tmpl_dust = h.dust_templates[mode][tidx]
    A1, A2, β1, β2 = _dust_AB(mode, pars)
    result .+= dust_model_template_power(ell, tmpl_dust, A1, A2, β1, β2,
                                          Float64(EFF_FREQ_DUST[f1]),
                                          Float64(EFF_FREQ_DUST[f2]),
                                          DUST_FREQ_REF, 19.6)

    # ------------------------------------------------------------------
    # 2. tSZ (TT only)
    # ------------------------------------------------------------------
    if mode == "TT"
        ef1 = Float64(EFF_FREQ_SZ[f1])
        ef2 = Float64(EFF_FREQ_SZ[f2])
        # α_tSZ = 0: template already encodes shape; ℓ_pivot is irrelevant
        sz = tsz_cross_power(h.tsz_template, pars[:Atsz], ef1, ef2,
                              FG_FREQ_REF, 0.0, 3000.0, ell)
        result .+= sz
    end

    # ------------------------------------------------------------------
    # 3. kSZ (TT only, frequency-independent)
    # ------------------------------------------------------------------
    if mode == "TT"
        result .+= ksz_template_scaled(h.ksz_template, pars[:Aksz])
    end

    # ------------------------------------------------------------------
    # 4. Clustered CIB (TT only)
    # ------------------------------------------------------------------
    if mode == "TT"
        ef1 = Float64(EFF_FREQ_CIB[f1])
        ef2 = Float64(EFF_FREQ_CIB[f2])
        # Use pre-loaded template: scale by A_cib × cib_sed(f1) × cib_sed(f2)
        # cib_clustered_power with α=0, z=1 reduces to A_CIB * s1 * s2 * (ℓ/3000)^0 = A_CIB*s1*s2
        # but Hillipop uses a file template, not a pure power law.
        # We therefore apply the SED scaling manually to the template:
        s1 = cib_mbb_sed_weight(pars[:beta_cib], 25.0, FG_FREQ_REF, ef1)
        s2 = cib_mbb_sed_weight(pars[:beta_cib], 25.0, FG_FREQ_REF, ef2)
        result .+= pars[:Acib] * s1 * s2 .* h.cib_template
    end

    # ------------------------------------------------------------------
    # 5. SZ×CIB (TT only)
    # ------------------------------------------------------------------
    if mode == "TT"
        ef_sz1  = Float64(EFF_FREQ_SZ[f1])
        ef_sz2  = Float64(EFF_FREQ_SZ[f2])
        ef_cib1 = Float64(EFF_FREQ_CIB[f1])
        ef_cib2 = Float64(EFF_FREQ_CIB[f2])
        # JAX formula: -xi * sqrt(Acib*Atsz) * [tszRatio(f2)*cibRatio(f1,β) + tszRatio(f1)*cibRatio(f2,β)] * x_tmpl
        # Translated directly for numerical accuracy:
        tr1 = tsz_g_ratio(ef_sz1, FG_FREQ_REF, 2.72548)
        tr2 = tsz_g_ratio(ef_sz2, FG_FREQ_REF, 2.72548)
        cr1 = cib_mbb_sed_weight(pars[:beta_cib], 25.0, FG_FREQ_REF, ef_cib1)
        cr2 = cib_mbb_sed_weight(pars[:beta_cib], 25.0, FG_FREQ_REF, ef_cib2)
        xi_factor = -pars[:xi] * sqrt(pars[:Acib] * pars[:Atsz]) * (tr2 * cr1 + tr1 * cr2)
        result .+= xi_factor .* h.szxcib_template
    end

    # ------------------------------------------------------------------
    # 6. Radio Point Sources (TT only)
    # ------------------------------------------------------------------
    if mode == "TT"
        ef1 = Float64(EFF_FREQ_RADIO[f1])
        ef2 = Float64(EFF_FREQ_RADIO[f2])
        result .+= radio_ps_power(ell, pars[:Aradio], pars[:beta_radio], ef1, ef2, FG_FREQ_REF)
    end

    # ------------------------------------------------------------------
    # 7. Dusty Point Sources (TT only)
    # ------------------------------------------------------------------
    if mode == "TT"
        ef1 = Float64(EFF_FREQ_CIB[f1])
        ef2 = Float64(EFF_FREQ_CIB[f2])
        result .+= dusty_ps_power(ell, pars[:Adusty], pars[:beta_cib],
                                   ef1, ef2, FG_FREQ_REF, 25.0)
    end

    # ------------------------------------------------------------------
    # 8. Sub-pixel effect (TT only)
    # ------------------------------------------------------------------
    if mode == "TT"
        key = Symbol("Asbpx_$(f1)x$(f2)")
        if haskey(pars, key)
            result .+= sub_pixel_power(ell, pars[key],
                                        Float64(FWHM_ARCMIN[f1]),
                                        Float64(FWHM_ARCMIN[f2]))
        end
    end

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
                           ell::AbstractVector, pars::Dict{Symbol},
                           h::HillipopData)
    return [compute_foreground_dl(mode, f1, f2, ell, pars, h)
            for (f1, f2) in pair_freqs]
end
