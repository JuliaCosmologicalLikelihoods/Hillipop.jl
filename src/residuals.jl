"""
    residuals.jl

Compute the D_ℓ residuals between observed data and calibrated theory + foregrounds.

The residual for cross-map-spectrum xs in mode `mode` is:
    R[xs] = D_data[xs] - cal_factor[xs] * D_model[xs]

where D_model = D_theory + Σ_fg D_fg, and cal_factor encodes absolute and
polarization calibration plus the global Planck calibration.

Calibration conventions (matching JAX exactly):
  TT:  cal_factor = cal1 * cal2 / A_planck²
  EE:  cal_factor = (cal1*pe1) * (cal2*pe2) / A_planck²
  TE:  cal_factor = cal1 * (cal2*pe2) / A_planck²
  ET:  cal_factor = (cal1*pe1) * cal2 / A_planck²
"""

"""
    _cal_factor(mode, map1, map2, pars)

Compute the calibration scale factor for cross-map-spectrum (map1 × map2) in `mode`.

# Arguments
- `mode`: `"TT"`, `"EE"`, `"TE"`, or `"ET"`
- `map1`, `map2`: map name strings, e.g. `"100A"`
- `pars`: `Dict{Symbol}` containing `cal100A`, `pe100A`, `A_planck`, etc.

# Returns
- Scalar `Float64` calibration factor
"""
function _cal_factor(mode::String, map1::String, map2::String, pars::HillipopNuisance)
    c1 = getproperty(pars.cal, Symbol("cal$(map1)"))
    c2 = getproperty(pars.cal, Symbol("cal$(map2)"))
    pe1 = getproperty(pars.cal, Symbol("pe$(map1)"))
    pe2 = getproperty(pars.cal, Symbol("pe$(map2)"))
    Apl = pars.cal.A_planck

    if mode == "TT"
        return c1 * c2 / Apl^2
    elseif mode == "EE"
        return (c1 * pe1) * (c2 * pe2) / Apl^2
    elseif mode == "TE"
        return c1 * (c2 * pe2) / Apl^2
    else  # ET
        return (c1 * pe1) * c2 / Apl^2
    end
end


"""
    compute_residuals(mode, dlth, pars, h)

Compute [D_data - cal * D_model] for all 15 cross-map-pair spectra.

# Arguments
- `mode`: `"TT"`, `"EE"`, `"TE"`, or `"ET"`
- `dlth`: `Vector{Float64}` of length lmax+1, the theory D_ℓ for this mode
- `pars`: `HillipopNuisance` of nuisance parameters
- `h`: `HillipopData`

# Returns
- `Matrix{Float64}` of shape `(nxspec, lmax+1)`, the residuals R
"""
function compute_residuals(mode::String, dlth::AbstractVector,
                           pars::HillipopNuisance{T_par}, h::HillipopData) where {T_par}
    mapnames     = h.mapnames
    frequencies  = h.frequencies
    lmax         = h.lmax
    nxspec       = length(h.mapnames) * (length(h.mapnames) - 1) ÷ 2

    # Build list of (map1, map2, f1, f2) for all 15 pairs
    pairs = Tuple{String,String,Int,Int}[]
    n = length(mapnames)
    for i in 1:n, j in i+1:n
        push!(pairs, (mapnames[i], mapnames[j], frequencies[i], frequencies[j]))
    end

    ell = collect(0:lmax)

    # Foreground contributions (list of 15 D_ℓ vectors)
    pair_freqs = [(f1, f2) for (_, _, f1, f2) in pairs]
    dlmodel_fgs = compute_fg_model(mode, pair_freqs, ell, pars, h)

    # Observed data for this mode
    dldata_mode = h.dldata[mode]   # (nxspec, lmax+1)

    # Residuals
    T_val = promote_type(eltype(dlth), T_par)
    R = zeros(T_val, nxspec, lmax + 1)
    for (xs, (map1, map2, f1, f2)) in enumerate(pairs)
        cal = _cal_factor(mode, map1, map2, pars)
        dlmodel_xs = dlth .+ dlmodel_fgs[xs]
        for l in 1:lmax+1
            R[xs, l] = dldata_mode[xs, l] - cal * dlmodel_xs[l]
        end
    end

    return R
end
