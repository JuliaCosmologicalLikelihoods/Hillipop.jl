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
function _get_cal(cal::HillipopCalibration, mapname::String)
    if mapname == "100A" return cal.cal100A
    elseif mapname == "100B" return cal.cal100B
    elseif mapname == "143A" return cal.cal143A
    elseif mapname == "143B" return cal.cal143B
    elseif mapname == "217A" return cal.cal217A
    elseif mapname == "217B" return cal.cal217B
    else throw(ArgumentError("Unknown map $mapname"))
    end
end

function _get_pe(cal::HillipopCalibration, mapname::String)
    if mapname == "100A" return cal.pe100A
    elseif mapname == "100B" return cal.pe100B
    elseif mapname == "143A" return cal.pe143A
    elseif mapname == "143B" return cal.pe143B
    elseif mapname == "217A" return cal.pe217A
    elseif mapname == "217B" return cal.pe217B
    else throw(ArgumentError("Unknown map $mapname"))
    end
end

function _cal_factor(mode::String, map1::String, map2::String, pars::HillipopNuisance)
    c1 = _get_cal(pars.cal, map1)
    c2 = _get_cal(pars.cal, map2)
    pe1 = _get_pe(pars.cal, map1)
    pe2 = _get_pe(pars.cal, map2)
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
    lmax         = h.lmax
    ell = 0:lmax
    dldata_mode = h.dldata[mode]

    T_val = promote_type(eltype(dlth), T_par)
    
    # Non-mutating way to build the residual matrix:
    # R[xs, l] = data - cal * (theory + fg)
    
    # We can use a comprehension to build rows
    R_rows = map(enumerate(h.pairs)) do (xs, (map1, map2, f1, f2))
        cal = _cal_factor(mode, map1, map2, pars)
        dlmodel_fg = compute_foreground_dl(mode, f1, f2, ell, pars, h)
        
        # Row vector for this xs
        return dldata_mode[xs, :] .- cal .* (dlth .+ dlmodel_fg)
    end

    # Stack rows into a matrix using stack (Julia 1.9+)
    return stack(R_rows; dims=1)
end
