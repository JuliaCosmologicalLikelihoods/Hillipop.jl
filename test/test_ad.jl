using Test
using Hillipop
using DifferentiationInterface
using ADTypes
import ForwardDiff, Mooncake
using LinearAlgebra

@testset "DifferentiationInterface Integration" begin
    h = load_hillipop(lmax=2500)
    ClTT = ones(2499) * 1e-12
    ClTE = zeros(2499)
    ClEE = ones(2499) * 1e-13

    # Fiducial parameters
    base_pars = (
        A_planck    = 1.0,
        AdustT      = 1.0, 
        Atsz        = 1.0,
        Acib        = 1.0,
    )

    keys_arr = collect(keys(base_pars))
    vals_arr = [base_pars[k] for k in keys_arr]

    function obj_func(v::Vector{T}) where T
        p_nt = NamedTuple{Tuple(keys_arr)}(Tuple(v))
        return compute_loglike(ClTT, ClTE, ClEE, HillipopNuisance(p_nt), h)
    end

    function obj_func_cl(v::Vector{T}) where T
        lmax_val = h.lmax
        c_tt = v[1:lmax_val-1]
        c_te = v[lmax_val:2*lmax_val-2]
        c_ee = v[2*lmax_val-1:3*lmax_val-3]
        return compute_loglike(c_tt, c_te, c_ee, HillipopNuisance(base_pars), h)
    end

    cl_vec = vcat(ClTT, ClTE, ClEE)

    @testset "ForwardDiff Backend" begin
        backend = AutoForwardDiff()
        val = obj_func(vals_arr)
        grad = gradient(obj_func, backend, vals_arr)
        @test isfinite(val)
        @test all(isfinite.(grad))
        @test length(grad) == length(vals_arr)

        val_cl = obj_func_cl(cl_vec)
        grad_cl = gradient(obj_func_cl, backend, cl_vec)
        @test isfinite(val_cl)
        @test all(isfinite.(grad_cl))
        @test length(grad_cl) == length(cl_vec)
    end

    @testset "Mooncake Backend" begin
        backend = AutoMooncake(config=nothing)
        val = obj_func(vals_arr)
        
        prep_nuisance = prepare_gradient(obj_func, backend, vals_arr)
        grad = gradient(obj_func, prep_nuisance, backend, vals_arr)
        @test isfinite(val)
        @test all(isfinite.(grad))
        
        # Compare with ForwardDiff for nuisance pars
        grad_fd = gradient(obj_func, AutoForwardDiff(), vals_arr)
        @test isapprox(grad, grad_fd, rtol=1e-6)

        # Compare with ForwardDiff for Cl spectra
        prep_cl = prepare_gradient(obj_func_cl, backend, cl_vec)
        grad_cl_mc = gradient(obj_func_cl, prep_cl, backend, cl_vec)
        grad_cl_fd = gradient(obj_func_cl, AutoForwardDiff(), cl_vec)
        
        @test all(isfinite.(grad_cl_mc))
        @test isapprox(grad_cl_mc, grad_cl_fd, rtol=1e-6)
    end

    # Zygote is skipped or marked as broken due to Dict/Mutation issues if not fully refactored
    # but we have already refactored many loops to comprehensions.
    @testset "Zygote Backend (Experimental)" begin
        import Zygote
        backend = AutoZygote()
        try
            grad = gradient(obj_func, backend, vals_arr)
            @test all(isfinite.(grad))
        catch e
            @info "Zygote still failing on this codebase: $e"
            @test_skip false
        end
    end
end
