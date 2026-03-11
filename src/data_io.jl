"""
    read_multipole_ranges(filename)

Read the per-mode, per-cross-frequency multipole ranges from a Hillipop binning FITS file.

Each extension in the FITS file corresponds to a polarization mode (TT, EE, BB, TE).
The returned dictionaries have one entry per mode; each entry is a Vector of length
`nxfreq` (=6 for the default 3-frequency setup).

# Arguments
- `filename`: path to `binning_v4.2.fits`

# Returns
- `(lmins, lmaxs)` — `Dict{String, Vector{Int}}` for each of TT, EE, BB, TE, ET
"""
function read_multipole_ranges(filename::AbstractString)
    lmins = Dict{String, Vector{Int}}()
    lmaxs = Dict{String, Vector{Int}}()

    FITS(filename, "r") do f
        # HDU 1 is the primary (empty); extensions start at 2
        for i in 2:length(f)
            hdu = f[i]
            tag = read_key(hdu, "SPEC")[1]
            lmins[tag] = read(hdu, "LMIN")
            lmaxs[tag] = read(hdu, "LMAX")
        end
    end

    # ET shares TE bounds
    lmins["ET"] = lmins["TE"]
    lmaxs["ET"] = lmaxs["TE"]

    return lmins, lmaxs
end


"""
    _map_combinations(mapnames)

Return all ordered pairs (m1, m2) with m1 < m2 (in index order), matching
`itertools.combinations` from Python.
"""
function _map_combinations(mapnames::Vector{String})
    pairs = Tuple{String,String}[]
    n = length(mapnames)
    for i in 1:n, j in i+1:n
        push!(pairs, (mapnames[i], mapnames[j]))
    end
    return pairs
end


"""
    _build_xspec2xfreq(frequencies)

Build the mapping from cross-map-spectrum index (0..nxspec-1) to
cross-frequency index (0..nxfreq-1), matching the JAX `_xspec2xfreq` method.

# Arguments
- `frequencies`: length-nmap vector of nominal frequencies (e.g. [100,100,143,143,217,217])

# Returns
- `Vector{Int}` of length nxspec, 1-based indices into the 6 cross-freq pairs
"""
function _build_xspec2xfreq(frequencies::Vector{Int})
    ufreqs = sort(unique(frequencies))
    nfreq  = length(ufreqs)

    # Build ordered list of (f1_idx, f2_idx) cross-freq pairs (f1<=f2)
    freq_pairs = Tuple{Int,Int}[]
    for i in 1:nfreq, j in i:nfreq
        push!(freq_pairs, (i, j))
    end

    # For each cross-map pair, find which cross-freq bin it belongs to
    nmap = length(frequencies)
    xspec2xfreq = Int[]
    for m1 in 1:nmap, m2 in m1+1:nmap
        fi1 = findfirst(==(frequencies[m1]), ufreqs)
        fi2 = findfirst(==(frequencies[m2]), ufreqs)
        # Ensure fi1 <= fi2 (matches the freq_pairs ordering)
        pair = fi1 <= fi2 ? (fi1, fi2) : (fi2, fi1)
        push!(xspec2xfreq, findfirst(==(pair), freq_pairs))
    end

    return xspec2xfreq
end


"""
    read_xspectra(basename, mapnames; lmax=2500)

Read all cross-spectra FITS files and return the observed D_ℓ data and 1/σ² weights.

Files are named `{basename}_{m1}x{m2}.fits`. HDU 1 contains D_ℓ [K²], multiplied
by 1e12 to convert to μK². HDU 2 contains σ(D_ℓ); weights = 1/σ².

# Arguments
- `basename`: path prefix, e.g. `"data/dl_PR4_v4.2"`
- `mapnames`: the 6 map labels, e.g. `["100A","100B","143A","143B","217A","217B"]`

# Returns
- `(dldata, dlweight)` — `Dict{String, Matrix{Float64}}` with keys `"TT","EE","TE","ET"`,
  each matrix of shape `(nxspec, lmax+1)`.
"""
function read_xspectra(basename::AbstractString, mapnames::Vector{String}; lmax::Int=2500)
    pairs   = _map_combinations(mapnames)
    nxspec  = length(pairs)
    modes   = ["TT", "EE", "TE", "ET"]

    dldata   = Dict(m => zeros(Float64, nxspec, lmax+1) for m in modes)
    dlweight = Dict(m => zeros(Float64, nxspec, lmax+1) for m in modes)

    for (xs, (m1, m2)) in enumerate(pairs)
        file_fwd = "$(basename)_$(m1)x$(m2).fits"
        file_rev = "$(basename)_$(m2)x$(m1).fits"

        FITS(file_fwd, "r") do f
            nhdu = length(f)
            # Check we have at least 2 HDUs (data + sigma)
            has_sigma = nhdu >= 2

            # HDU 2 = data (1-indexed in Julia FITSIO: primary is 1, first extension is 2)
            data1 = read(f[2]) .* 1e12   # shape: (4, ell_max_file) in K²→μK²

            if has_sigma
                sigma1 = read(f[3]) .* 1e12
            end

            # TT = col 1, EE = col 2, TE = col 4 (FITS 1-indexed; JAX uses rows 0,1,3)
            nell = min(size(data1, 1), lmax+1)
            dldata["TT"][xs, 1:nell] .= data1[1:nell, 1]
            dldata["EE"][xs, 1:nell] .= data1[1:nell, 2]
            dldata["TE"][xs, 1:nell] .= data1[1:nell, 4]

            if has_sigma
                nell_s = min(size(sigma1, 1), lmax+1)
                for mode in ("TT", "EE", "TE")
                    col = mode == "TT" ? 1 : mode == "EE" ? 2 : 4
                    for ell in 1:nell_s
                        s = sigma1[ell, col]
                        dlweight[mode][xs, ell] = s == 0.0 ? 0.0 : 1.0 / s^2
                    end
                end
            else
                # Uniform weight fallback
                for mode in ("TT", "EE", "TE")
                    dlweight[mode][xs, :] .= 1.0
                end
            end
        end

        # ET: use row 4 from the reversed FITS (m2×m1)
        FITS(file_rev, "r") do f
            nhdu = length(f)
            has_sigma = nhdu >= 2
            data2 = read(f[2]) .* 1e12
            nell = min(size(data2, 1), lmax+1)
            dldata["ET"][xs, 1:nell] .= data2[1:nell, 4]
            if has_sigma
                sigma2 = read(f[3]) .* 1e12
                nell_s = min(size(sigma2, 1), lmax+1)
                for ell in 1:nell_s
                    s = sigma2[ell, 4]
                    dlweight["ET"][xs, ell] = s == 0.0 ? 0.0 : 1.0 / s^2
                end
            else
                dlweight["ET"][xs, :] .= 1.0
            end
        end
    end

    return dldata, dlweight
end


"""
    read_invkll_npy(binning_matrix_file, binned_invkll_file)

Load the pre-binned compressed inverse covariance matrix from NumPy .npy files.

# Arguments
- `binning_matrix_file`: path to `binning_matrix.npy`,  shape `(nbins, data_vector_len)`
- `binned_invkll_file`: path to `binned_invkll.npy`, shape `(nbins, nbins)`

# Returns
- `(B, Kinv)` as `Matrix{Float64}`
"""
function read_invkll_npy(binning_matrix_file::AbstractString, binned_invkll_file::AbstractString)
    B    = npzread(binning_matrix_file)
    Kinv = npzread(binned_invkll_file)
    return Float64.(B), Float64.(Kinv)
end


"""
    read_fg_template(filename; lmax=2500, lnorm=3000)

Read a two-column (ell, Dl) ASCII foreground template, zero-pad to integer ℓ grid,
and normalize at `lnorm`.

# Returns
- `Vector{Float64}` of length `lmax+1` (index i corresponds to ℓ = i-1)
"""
function read_fg_template(filename::AbstractString; lmax::Int=2500, lnorm::Int=3000)
    data = readdlm(filename, comments=true)
    ells  = Int.(data[:, 1])
    vals  = Float64.(data[:, 2])

    max_ell = max(lmax, lnorm, maximum(ells))
    tmpl = zeros(Float64, max_ell + 1)
    for (e, v) in zip(ells, vals)
        tmpl[e+1] = v   # 1-based: index e+1 = ell e
    end

    # Normalize at lnorm
    norm_val = tmpl[lnorm+1]
    if norm_val != 0.0
        tmpl ./= norm_val
    end

    return tmpl[1:lmax+1]
end


"""
    read_dust_templates(basename, mode; lmax=2500, cross_pairs=DEFAULT_CROSS_PAIRS)

Read a 7-column dust template file `{basename}_{mode}.txt` with columns:
`ell, 100x100, 100x143, 100x217, 143x143, 143x217, 217x217`.

Returns a `Vector` of 6 `Vector{Float64}` templates (one per cross-freq pair),
zero-padded to `lmax+1`, **not** renormalized (templates are already in μK²).

# Column ordering matches JAX `dust` and `dust_model` classes.
"""
const _DUST_HDR = ["100x100", "100x143", "100x217", "143x143", "143x217", "217x217"]

function read_dust_templates(basename::AbstractString, mode::AbstractString;
                              lmax::Int=2500)
    filename = "$(basename)_$(mode).txt"
    data = readdlm(filename, comments=true)
    ells = Int.(data[:, 1])

    max_ell = max(lmax, 3000, maximum(ells))
    templates = [zeros(Float64, max_ell+1) for _ in 1:6]

    for (col_idx, col_name) in enumerate(_DUST_HDR)
        vals = Float64.(data[:, col_idx+1])
        for (e, v) in zip(ells, vals)
            templates[col_idx][e+1] = v
        end
        # Normalize at ℓ=3000
        norm_val = templates[col_idx][3001]
        if norm_val != 0.0
            templates[col_idx] ./= norm_val
        end
        templates[col_idx] = templates[col_idx][1:lmax+1]
    end

    return templates
end


"""
    load_hillipop(data_dir=artifact"planck_PR4_hillipop"; lmax=2500)

Load all data needed for the Hillipop PR4 likelihood and return a `HillipopData` struct.
By default, this automatically downloads the `planck_PR4_hillipop` artifact from Zenodo.

# Arguments
- `data_dir`: directory containing the Hillipop data files
  (e.g. `joinpath(@__DIR__, "..", "data")`)

# Returns
- `HillipopData` struct
"""
function load_hillipop(data_dir::AbstractString=artifact"planck_PR4_hillipop"; lmax::Int=2500)
    mapnames  = ["100A", "100B", "143A", "143B", "217A", "217B"]
    frequencies = [100, 100, 143, 143, 217, 217]

    # --- Multipole ranges ---
    binning_file = joinpath(data_dir, "binning_v4.2.fits")
    lmins, lmaxs = read_multipole_ranges(binning_file)

    # --- Cross-spectra data + weights ---
    basename = joinpath(data_dir, "dl_PR4_v4.2")
    dldata, dlweight = read_xspectra(basename, mapnames; lmax=lmax)

    # --- Compressed inverse covariance ---
    bmat_file = joinpath(data_dir, "binning_matrix_dense.npy")
    bkll_file = joinpath(data_dir, "binned_invkll_dense.npy")
    binning_matrix, binned_invkll = read_invkll_npy(bmat_file, bkll_file)

    # --- xspec→xfreq map ---
    xspec2xfreq = _build_xspec2xfreq(frequencies)

    # --- Foreground templates ---
    fg_dir = joinpath(data_dir, "foregrounds")
    dust_basename = joinpath(fg_dir, "DUST_Planck_PR4_model_v4.2")
    dust_templates = Dict{String, Vector{Vector{Float64}}}()
    for mode in ("TT", "EE", "TE", "ET")
        dust_templates[mode] = read_dust_templates(dust_basename, mode; lmax=lmax)
    end

    tsz_template   = read_fg_template(joinpath(fg_dir, "SZ_Planck_PR4_model.txt");   lmax=lmax)
    ksz_template   = read_fg_template(joinpath(fg_dir, "kSZ_Planck_PR4_model.txt");  lmax=lmax)
    cib_template   = read_fg_template(joinpath(fg_dir, "CIB_Planck_PR4_model.txt");  lmax=lmax)
    szxcib_template = read_fg_template(joinpath(fg_dir, "SZxCIB_Planck_PR4_model.txt"); lmax=lmax)

    return HillipopData(
        lmins, lmaxs,
        dldata, dlweight,
        binning_matrix, binned_invkll,
        xspec2xfreq,
        mapnames, frequencies,
        lmax,
        dust_templates,
        tsz_template, ksz_template, cib_template, szxcib_template
    )
end
