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
    binning_matrix::SparseMatrixCSC{Float64, Int}    # (nbins, data_vector_length)
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
