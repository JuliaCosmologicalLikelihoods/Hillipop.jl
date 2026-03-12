using Test
using Hillipop
using JSON

@testset "Hillipop.jl" begin
    # ---------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------
    @testset "load_hillipop" begin
        h = load_hillipop()

        @test h.lmax == 2500
        @test length(h.mapnames) == 6
        @test length(h.frequencies) == 6
        @test length(h.xspec2xfreq) == 15   # C(6,2) = 15 map pairs

        # Multipole ranges
        for mode in ("TT", "EE", "TE", "ET")
            @test haskey(h.lmins, mode)
            @test haskey(h.lmaxs, mode)
        end

        # Data matrices
        for mode in ("TT", "EE", "TE", "ET")
            @test size(h.dldata[mode])   == (15, h.lmax+1)
            @test size(h.dlweight[mode]) == (15, h.lmax+1)
        end

        # Covariance
        nbins = size(h.binning_matrix, 1)
        @test size(h.binning_matrix, 2) > 0
        @test size(h.binned_invkll) == (nbins, nbins)

        # Templates
        @test length(h.tsz_template)    == h.lmax + 1
        @test length(h.ksz_template)    == h.lmax + 1
        @test length(h.cib_template)    == h.lmax + 1
        @test length(h.szxcib_template) == h.lmax + 1

        for mode in ("TT", "EE", "TE", "ET")
            @test length(h.dust_templates[mode]) == 6
            @test all(length(t) == h.lmax+1 for t in h.dust_templates[mode])
        end
    end

    # ---------------------------------------------------------------
    # Equivalence with JAX
    # ---------------------------------------------------------------
    @testset "Equivalence with JAX Reference" begin
        ref_json = joinpath(dirname(@__DIR__), "test", "variational_references.json")
        if isfile(ref_json)
            cases = JSON.parsefile(ref_json)
            h = load_hillipop()
            
            for case in cases
                name = case["name"]
                @testset "$name" begin
                    cltt = Float64.(case["cltt"])
                    clte = Float64.(case["clte"])
                    clee = Float64.(case["clee"])
                    
                    pars_raw = case["params"]
                    pars = Dict{Symbol, Float64}(Symbol(k) => Float64(v) for (k, v) in pars_raw)
                    
                    logL_jax = Float64(case["logL"])
                    logL_jl  = compute_loglike(cltt, clte, clee, pars, h)
                    
                    @info "Case: $name"
                    @info "JAX logL:   $logL_jax"
                    @info "Julia logL: $logL_jl"
                    
                    diff = abs(logL_jax - logL_jl)
                    reldiff = diff / max(1.0, abs(logL_jax))
                    @info "Abs Diff:   $diff, Rel Diff: $reldiff"
                    
                    @test isapprox(logL_jax, logL_jl, rtol=1e-3)
                end
            end
        else
            @warn "variational_references.json not found, skipping equivalence test."
        end
    end

    # ---------------------------------------------------------------
    # Likelihood evaluation (smoke test)
    # ---------------------------------------------------------------
    @testset "compute_loglike smoke" begin
        h = load_hillipop()

        # Minimal parameter set (all nuisances at fiducial/unit values)
        pars = HillipopNuisance(;
            cal = HillipopCalibration(
                A_planck = 1.0,
                cal100A = 1.0, cal100B = 1.0,
                cal143A = 1.0, cal143B = 1.0,
                cal217A = 1.0, cal217B = 1.0,
                pe100A  = 1.0, pe100B  = 1.0,
                pe143A  = 1.0, pe143B  = 1.0,
                pe217A  = 1.0, pe217B  = 1.0
            ),
            dust = HillipopDust(
                AdustT  = 1.0, AdustP  = 1.0,
                beta_dustT = 1.5, beta_dustP = 1.5
            ),
            sz = HillipopSZ(
                Atsz    = 1.0,
                Aksz    = 1.0
            ),
            cib = HillipopCIB(
                Acib    = 1.0,
                beta_cib = 1.75,
                xi      = 0.1
            ),
            ps = HillipopPointSources(
                Aradio  = 0.0,
                beta_radio = -0.7,
                Adusty  = 0.0
            ),
            subpixel = HillipopSubPixel(
                Asbpx_100x100 = 0.0, Asbpx_100x143 = 0.0,
                Asbpx_100x217 = 0.0, Asbpx_143x143 = 0.0,
                Asbpx_143x217 = 0.0, Asbpx_217x217 = 0.0
            )
        )

        # Flat (Zel'dovich) Cl spectra as a placeholder
        lmax_in = h.lmax
        ells = 2:lmax_in
        ClTT = @. 6000.0 / (ells * (ells + 1)) * 1e-12   # rough BB-like shape in K²
        ClTE = fill(0.0, length(ells))
        ClEE = fill(0.0, length(ells))

        logL = compute_loglike(ClTT, ClTE, ClEE, pars, h)
        @test isfinite(logL)
        @test logL < 0.0   # likelihood must be ≤ 0 (log of a probability ≤ 1 at maximum)
    end

    # ---------------------------------------------------------------
    # Calibration: A_planck=1, identical maps, zero fg → zero effect
    # ---------------------------------------------------------------
    @testset "calibration sanity" begin
        h = load_hillipop()
        # With cal=1, pe=1, A_planck=1: residual = data - model
        # Changing A_planck shifts the calibration factor
        # We just check that varying A_planck produces a different loglike
        pars_base = Dict{Symbol,Float64}(
            :cal100A=>1.0,:cal100B=>1.0,:cal143A=>1.0,:cal143B=>1.0,:cal217A=>1.0,:cal217B=>1.0,
            :pe100A=>1.0,:pe100B=>1.0,:pe143A=>1.0,:pe143B=>1.0,:pe217A=>1.0,:pe217B=>1.0,
            :A_planck=>1.0,
            :AdustT=>1.0,:AdustP=>1.0,:beta_dustT=>1.5,:beta_dustP=>1.5,
            :Atsz=>1.0,:Aksz=>1.0,:Acib=>1.0,:beta_cib=>1.75,:xi=>0.1,
            :Aradio=>0.0,:beta_radio=>-0.7,:Adusty=>0.0,
        )
        ells = 2:h.lmax
        ClTT = @. 6000.0 / (ells * (ells + 1)) * 1e-12
        ClTE = fill(0.0, length(ells))
        ClEE = fill(0.0, length(ells))

        pars_shifted = copy(pars_base)
        pars_shifted[:A_planck] = 1.01  # 1% shift

        logL0 = compute_loglike(ClTT, ClTE, ClEE, pars_base, h)
        logL1 = compute_loglike(ClTT, ClTE, ClEE, pars_shifted, h)
        @test logL0 != logL1
    end
end


using ForwardDiff

@testset "ForwardDiff compatibility" begin
    h    = load_hillipop(lmax=2500)
    ClTT = ones(2499)
    ClTE = ones(2499)
    ClEE = ones(2499)
    pars = HillipopNuisance(;
        cal = HillipopCalibration(
            A_planck = 1.0,
            cal100A = 1.0, cal100B = 1.0,
            cal143A = 1.0, cal143B = 1.0,
            cal217A = 1.0, cal217B = 1.0,
            pe100A  = 1.0, pe100B  = 1.0,
            pe143A  = 1.0, pe143B  = 1.0,
            pe217A  = 1.0, pe217B  = 1.0
        ),
        dust = HillipopDust(
            AdustT  = 1.0, AdustP  = 1.0,
            beta_dustT = 1.5, beta_dustP = 1.5
        ),
        sz = HillipopSZ(
            Atsz    = 1.0,
            Aksz    = 1.0
        ),
        cib = HillipopCIB(
            Acib    = 1.0,
            beta_cib = 1.75,
            xi      = 0.1
        ),
        ps = HillipopPointSources(
            Aradio  = 0.0,
            beta_radio = -0.7,
            Adusty  = 0.0
        ),
        subpixel = HillipopSubPixel(
            Asbpx_100x100 = 0.0, Asbpx_100x143 = 0.0,
            Asbpx_100x217 = 0.0, Asbpx_143x143 = 0.0,
            Asbpx_143x217 = 0.0, Asbpx_217x217 = 0.0
        )
    )
    base_pars = Dict{Symbol,Float64}(
        :cal100A => 1.0, :cal100B => 1.0,
        :cal143A => 1.0, :cal143B => 1.0,
        :cal217A => 1.0, :cal217B => 1.0,
        :pe100A  => 1.0, :pe100B  => 1.0,
        :pe143A  => 1.0, :pe143B  => 1.0,
        :pe217A  => 1.0, :pe217B  => 1.0,
        :A_planck    => 1.0,
        :AdustT      => 1.0, :AdustP     => 1.0,
        :beta_dustT  => 1.5, :beta_dustP => 1.5,
        :Atsz    => 1.0,  :Aksz    => 1.0,
        :Acib    => 1.0,  :beta_cib => 1.75,
        :xi      => 0.1,
        :Aradio  => 0.1,  :beta_radio => -0.7,
        :Adusty  => 0.1,
    )

    keys_arr = collect(keys(base_pars))
    vals_arr = Float64[base_pars[k] for k in keys_arr]

    function obj_func(v::Vector{T}) where T
        p_dict = Dict{Symbol,T}(keys_arr[i] => v[i] for i in eachindex(keys_arr))
        return compute_loglike(ClTT, ClTE, ClEE, p_dict, h)
    end

    grad = ForwardDiff.gradient(obj_func, vals_arr)
    @test length(grad) == length(vals_arr)
    @test all(isfinite.(grad))
end
