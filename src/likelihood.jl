"""
    likelihood.jl

Core Hillipop PR4 log-likelihood computation.

The likelihood is Gaussian in the residual D_ℓ vector:
    -2 ln L = Xl · K⁻¹ · Xl

where Xl = concat(TT, EE, TE) residuals, weighted-averaged across cross-frequency pairs,
and K⁻¹ is the (compressed) inverse covariance matrix.

The TE contribution combines TE and ET residuals jointly (weighted sum over both
before dividing by the total weight), matching the JAX implementation exactly.
"""

"""
    xspectra_to_xfreq(Rspec, weights, xspec2xfreq, nxfreq; normed=true)

Average 15 cross-map-pair spectra into 6 cross-frequency spectra using weighted mean.

Multipoles with zero weight (masked) are excluded from both numerator and denominator.
Returns just the normalized result when `normed=true`, or `(numerator, weight)` when false.

# Arguments
- `Rspec`: `(nxspec, lmax+1)` residual matrix
- `weights`: `(nxspec, lmax+1)` inverse-variance weights
- `xspec2xfreq`: `Vector{Int}` mapping xs → cross-freq index (1-based)
- `nxfreq`: number of cross-frequency bins (6)

# Returns
- `(nxfreq, lmax+1)` weighted-average or `((nxfreq, lmax+1), (nxfreq, lmax+1))` unnormed
"""
function xspectra_to_xfreq_unnormed(Rspec::AbstractMatrix, weights::AbstractMatrix,
                             xspec2xfreq::Vector{Int}, nxfreq::Int)
    nxspec, lmax1 = size(Rspec)
    T   = eltype(Rspec)
    
    # We can use a more functional approach to avoid mutation
    # For each xf, we sum over xs where xspec2xfreq[xs] == xf
    xcl = [sum(weights[xs, l] * Rspec[xs, l] for xs in 1:nxspec if xspec2xfreq[xs] == xf && isfinite(weights[xs, l]) && weights[xs, l] > 0.0; init=zero(T)) for xf in 1:nxfreq, l in 1:lmax1]
    xw8 = [sum(weights[xs, l] for xs in 1:nxspec if xspec2xfreq[xs] == xf && isfinite(weights[xs, l]) && weights[xs, l] > 0.0; init=zero(T)) for xf in 1:nxfreq, l in 1:lmax1]

    return xcl, xw8
end

function xspectra_to_xfreq(Rspec::AbstractMatrix, weights::AbstractMatrix,
                             xspec2xfreq::Vector{Int}, nxfreq::Int)
    xcl, xw8 = xspectra_to_xfreq_unnormed(Rspec, weights, xspec2xfreq, nxfreq)
    
    # Safe division: zero-weight entries stay zero
    return @. ifelse(xw8 > 0.0, xcl / xw8, zero(eltype(xcl)))
end


"""
    select_spectra(xcl, lmins, lmaxs, nxfreq, xspec2xfreq)

Slice each cross-frequency spectrum to its allowed ℓ range and concatenate
into a single flat data vector, matching JAX `_select_spectra`.

# Arguments
- `xcl`: `(nxfreq, lmax+1)` cross-frequency residuals
- `lmins`, `lmaxs`: per-cross-freq ℓ limits (1-indexed arrays, i.e. lmins[xf])
- `nxfreq`: 6

# Returns
- Flat `Vector{Float64}` (the data vector Xl for this mode)
"""
function select_spectra(xcl::AbstractMatrix, lmins::Vector{Int}, lmaxs::Vector{Int},
                         nxfreq::Int, xspec2xfreq::Vector{Int})
    T = eltype(xcl)
    # Using a comprehension to avoid push!
    slices = map(1:nxfreq) do xf
        # Find the representative cross-map-spec index for this cross-freq bin
        xs_rep = findfirst(==(xf), xspec2xfreq)
        lmin = lmins[xs_rep]
        lmax = lmaxs[xs_rep]
        # Julia arrays are 1-indexed; ℓ=lmin is at position lmin+1
        return xcl[xf, lmin+1:lmax+1]
    end
    return vcat(slices...)
end


"""
    compute_chi2(delta_cl, binning_matrix, binned_invkll)

Compute the scalar chi² = (B·Xl)ᵀ · K̃⁻¹ · (B·Xl) using the pre-binned
compressed inverse covariance matrix.

# Arguments
- `delta_cl`: flat data vector Xl of length N
- `binning_matrix`: `(nbins, N)` projection matrix
- `binned_invkll`: `(nbins, nbins)` binned inverse covariance

# Returns
- Scalar chi² value
"""
function compute_chi2(delta_cl::AbstractVector,
                       binning_matrix::AbstractMatrix,
                       binned_invkll::AbstractMatrix)
    proj = binning_matrix * delta_cl         # (nbins,)
    return dot(proj, binned_invkll * proj)   # scalar
end


"""
    _cl_to_dl(Cl, lmax)

Convert a Cl spectrum (K², ℓ starting from ℓ=2 at index 1) to D_ℓ (μK²)
for ℓ in 0..lmax, prepending zeros for ℓ=0,1.

# Returns
- `Vector{Float64}` of length lmax+1
"""
function _cl_to_dl(Cl::AbstractVector, lmax::Int)
    # Using a non-mutating approach: map or comprehension
    # l=0,1 are zero; l>=2 comes from Cl
    dl = map(0:lmax) do l
        if l < 2
            return zero(eltype(Cl))
        else
            cl_idx = l - 1
            if cl_idx <= length(Cl)
                return Cl[cl_idx] * 1e12 * l * (l + 1) / (2π)
            else
                return zero(eltype(Cl))
            end
        end
    end
    return dl
end


"""
    build_residual_vector(ClTT, ClTE, ClEE, pars, h; modes=("TT","EE","TE"))

Assemble the flat concatenated residual data vector `Xl` used by the
Hillipop likelihood.

This function is extracted from `compute_loglike` so it can be called
independently — for example from a Turing.jl `MvNormal` model variant
that constructs the likelihood as `Xl_binned ~ MvNormal(0, C)` directly.

# Arguments
- `ClTT`, `ClTE`, `ClEE`: theory power spectra in K², starting at ℓ=2
- `pars`: nuisance parameters (any type accepted by `compute_residuals`)
- `h`: `HillipopData`

# Returns
- `Vector` of length equal to the total number of selected unbinned multipoles
  across all requested modes. Project by `h.binning_matrix` to get the
  compressed binned vector needed for `h.binned_invkll`.
"""
function build_residual_vector(ClTT::AbstractVector, ClTE::AbstractVector,
                                ClEE::AbstractVector, pars::HillipopNuisance{T_par}, h::HillipopData;
                                modes::Tuple=("TT", "EE", "TE")) where {T_par}
    lmax   = h.lmax
    nxfreq = length(unique(h.frequencies)) * (length(unique(h.frequencies)) + 1) ÷ 2

    # Convert input Cl (K², ℓ≥2) → D_ℓ (μK², index 1 = ℓ=0)
    dlth = (
        TT = _cl_to_dl(ClTT, lmax),
        EE = _cl_to_dl(ClEE, lmax),
        TE = _cl_to_dl(ClTE, lmax),
        ET = _cl_to_dl(ClTE, lmax),
    )

    # TT mode part
    Xl_tt = if "TT" ∈ modes
        R_tt = compute_residuals("TT", dlth.TT, pars, h)
        Rl_tt = xspectra_to_xfreq(R_tt, h.dlweight["TT"], h.xspec2xfreq, nxfreq)
        select_spectra(Rl_tt, h.lmins["TT"], h.lmaxs["TT"], nxfreq, h.xspec2xfreq)
    else
        empty(ClTT)
    end

    # EE mode part
    Xl_ee = if "EE" ∈ modes
        R_ee = compute_residuals("EE", dlth.EE, pars, h)
        Rl_ee = xspectra_to_xfreq(R_ee, h.dlweight["EE"], h.xspec2xfreq, nxfreq)
        select_spectra(Rl_ee, h.lmins["EE"], h.lmaxs["EE"], nxfreq, h.xspec2xfreq)
    else
        empty(ClEE)
    end

    # TE mode part
    Xl_te = if "TE" ∈ modes
        R_te = compute_residuals("TE", dlth.TE, pars, h)
        R_et = compute_residuals("ET", dlth.ET, pars, h)
        Rnum_te, Rw_te = xspectra_to_xfreq_unnormed(R_te, h.dlweight["TE"], h.xspec2xfreq, nxfreq)
        Rnum_et, Rw_et = xspectra_to_xfreq_unnormed(R_et, h.dlweight["ET"], h.xspec2xfreq, nxfreq)
        Rnum  = Rnum_te .+ Rnum_et
        Rw    = Rw_te   .+ Rw_et
        
        Rl_te = @. ifelse(Rw > 0.0, Rnum / Rw, zero(eltype(Rnum)))
        select_spectra(Rl_te, h.lmins["TE"], h.lmaxs["TE"], nxfreq, h.xspec2xfreq)
    else
        empty(ClTE)
    end

    return vcat(Xl_tt, Xl_ee, Xl_te)
end


"""
    compute_loglike(ClTT, ClTE, ClEE, pars, h; modes=("TT","EE","TE"))

Compute the Hillipop PR4 log-likelihood.

# Arguments
- `ClTT`, `ClTE`, `ClEE`: `Vector` of C_ℓ in K², starting at ℓ=2
- `pars`: `HillipopNuisance` (or `Dict{Symbol}`) of nuisance parameters (see parameter list below)
- `h`: `HillipopData` struct from `load_hillipop`

# Keywords
- `modes`: which polarization modes to include (default: all three)

# Required parameter keys
- Calibration: `:cal100A`, `:cal100B`, `:cal143A`, `:cal143B`, `:cal217A`, `:cal217B`
- Pol. efficiency: `:pe100A`, `:pe100B`, `:pe143A`, `:pe143B`, `:pe217A`, `:pe217B`
- Global: `:A_planck`
- Dust: `:AdustT`, `:AdustP`, `:beta_dustT`, `:beta_dustP`
- tSZ: `:Atsz`
- kSZ: `:Aksz`
- CIB: `:Acib`, `:beta_cib`
- SZ×CIB: `:xi`
- Radio PS: `:Aradio`, `:beta_radio`
- Dusty PS: `:Adusty`
- Sub-pixel (optional): `:Asbpx_100x100`, ..., `:Asbpx_217x217`

# Returns
- Scalar log-likelihood value
"""
function compute_loglike(ClTT::AbstractVector, ClTE::AbstractVector, ClEE::AbstractVector,
                          pars::HillipopNuisance, h::HillipopData;
                          modes::Tuple=("TT", "EE", "TE"))
    Xl   = build_residual_vector(ClTT, ClTE, ClEE, pars, h; modes=modes)
    chi2 = compute_chi2(Xl, h.binning_matrix, h.binned_invkll)
    return -0.5 * chi2
end

function compute_loglike(ClTT::AbstractVector, ClTE::AbstractVector, ClEE::AbstractVector,
                          pars::NamedTuple, h::HillipopData;
                          modes::Tuple=("TT", "EE", "TE"))
    return compute_loglike(ClTT, ClTE, ClEE, HillipopNuisance(pars), h; modes=modes)
end

function compute_loglike(ClTT::AbstractVector, ClTE::AbstractVector, ClEE::AbstractVector,
                          pars::Dict{Symbol}, h::HillipopData;
                          modes::Tuple=("TT", "EE", "TE"))
    return compute_loglike(ClTT, ClTE, ClEE, HillipopNuisance(pars), h; modes=modes)
end
