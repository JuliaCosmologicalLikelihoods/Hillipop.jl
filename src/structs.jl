"""
    HillipopData

Pre-loaded data struct for the Hillipop PR4 likelihood.

All data that can be read once at initialization is stored here,
so that `compute_loglike` is allocation-minimal and AD-friendly.
"""
struct HillipopData
    # Multipole ranges per mode, per cross-frequency (6 cross-freqs)
    lmins::Dict{String, Vector{Int}}
    lmaxs::Dict{String, Vector{Int}}

    # Observed D_ℓ cross-spectra: mode => (nxspec=15, ell_max+1) in μK²
    dldata::Dict{String, Matrix{Float64}}

    # Inverse-variance weights: mode => (nxspec=15, ell_max+1)
    dlweight::Dict{String, Matrix{Float64}}

    # Pre-compressed inverse covariance
    binning_matrix::Matrix{Float64}    # (nbins, data_vector_length)
    binned_invkll::Matrix{Float64}     # (nbins, nbins)

    # Map of 15 cross-map-spectra → 6 cross-frequency indices (0-based)
    xspec2xfreq::Vector{Int}

    # Map names and nominal frequencies
    mapnames::Vector{String}
    frequencies::Vector{Int}

    # ℓ_max used for theory spectra
    lmax::Int

    # Pre-loaded foreground templates (D_ℓ, normalized at ℓ=3000 by default)
    # dust: Dict mode => Vector of 6 cross-freq templates
    dust_templates::Dict{String, Vector{Vector{Float64}}}
    tsz_template::Vector{Float64}
    ksz_template::Vector{Float64}
    cib_template::Vector{Float64}
    szxcib_template::Vector{Float64}
end


# ===========================================================================
# HillipopPars — typed nuisance parameter container
# ===========================================================================

"""
    HillipopPars{T}

Statically-typed container for all Hillipop PR4 nuisance parameters.

The type parameter `T` allows this struct to work transparently with
automatic differentiation (e.g. `ForwardDiff.Dual`) as well as plain
`Float64` for direct evaluation.

Sub-pixel amplitudes (`Asbpx_*`) are **not** included; add them to the
`Dict{Symbol}` manually when needed.

| Field | Physical meaning |
|---|---|
| `A_planck` | Global Planck calibration |
| `cal100A/B, cal143A/B, cal217A/B` | Per-map absolute calibration |
| `pe100A/B, pe143A/B, pe217A/B` | Per-map polarization efficiency |
| `AdustT, AdustP` | Galactic dust amplitude (T and P) |
| `beta_dustT, beta_dustP` | Galactic dust spectral index (T and P) |
| `Atsz` | Thermal SZ amplitude |
| `Aksz` | Kinetic SZ amplitude |
| `Acib, beta_cib` | CIB amplitude and spectral index |
| `xi` | SZ×CIB cross-correlation coefficient |
| `Aradio, beta_radio` | Radio PS amplitude and spectral index |
| `Adusty` | Dusty (infrared) PS amplitude |
"""
struct HillipopPars{T}
    A_planck    :: T
    cal100A     :: T; cal100B    :: T
    cal143A     :: T; cal143B    :: T
    cal217A     :: T; cal217B    :: T
    pe100A      :: T; pe100B     :: T
    pe143A      :: T; pe143B     :: T
    pe217A      :: T; pe217B     :: T
    AdustT      :: T; AdustP     :: T
    beta_dustT  :: T; beta_dustP :: T
    Atsz        :: T
    Aksz        :: T
    Acib        :: T; beta_cib   :: T
    xi          :: T
    Aradio      :: T; beta_radio :: T
    Adusty      :: T
end

"""
    Base.convert(::Type{Dict{Symbol,T}}, p::HillipopPars{T}) where T

Convert a `HillipopPars` to a `Dict{Symbol}` for use with `compute_loglike`.
"""
function Base.convert(::Type{Dict{Symbol,T}}, p::HillipopPars{T}) where T
    return Dict{Symbol,T}(
        :A_planck    => p.A_planck,
        :cal100A     => p.cal100A,    :cal100B    => p.cal100B,
        :cal143A     => p.cal143A,    :cal143B    => p.cal143B,
        :cal217A     => p.cal217A,    :cal217B    => p.cal217B,
        :pe100A      => p.pe100A,     :pe100B     => p.pe100B,
        :pe143A      => p.pe143A,     :pe143B     => p.pe143B,
        :pe217A      => p.pe217A,     :pe217B     => p.pe217B,
        :AdustT      => p.AdustT,     :AdustP     => p.AdustP,
        :beta_dustT  => p.beta_dustT, :beta_dustP => p.beta_dustP,
        :Atsz        => p.Atsz,
        :Aksz        => p.Aksz,
        :Acib        => p.Acib,       :beta_cib   => p.beta_cib,
        :xi          => p.xi,
        :Aradio      => p.Aradio,     :beta_radio => p.beta_radio,
        :Adusty      => p.Adusty,
    )
end
