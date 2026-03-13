using Test
using Hillipop
using Pkg

# Utility function to check if CLASS is installed on the system path
function is_class_available()
    try
        success(`class --version`) || success(`class`)
        return true
    catch
        return false
    end
end

@testset "CLASS.jl Integration" begin
    # Check if CLASS is in dependencies and binary is available
    has_class_dep = haskey(Pkg.project().dependencies, "CLASS") || haskey(Pkg.project().dependencies, "CLASS_jll")
    
    # In some setups, CLASS might be dynamically loaded.
    # We will conditionally load DataFrames and CLASS if available.
    if !is_class_available()
        @info "CLASS executable not found in PATH, skipping CLASS integration tests."
        @test_skip false
    else
        # If class is available, we assume the environment has CLASS.jl
        # (It should be added as a test dependency)
        using CLASS
        using DataFrames

        @testset "Fiducial Cosmology Evaluation" begin
            # 1. Define Fiducial Cosmology
            cosmo_params = (
                H0        = 67.36,   
                omega_b   = 0.02237, 
                omega_cdm = 0.1200,  
                A_s       = 2.101e-9,
                n_s       = 0.9649,  
                tau_reio  = 0.0544,  
                N_ur      = 2.0328,  
                m_ncdm    = 0.06,    
            )

            # 2. Define Nuisance Parameters
            nuisance = HillipopNuisance(
                HillipopCalibration(
                    A_planck = 1.0, cal100A = 1.0, cal100B = 1.0, cal143A = 1.0, cal143B = 1.0,
                    cal217A = 1.0, cal217B = 1.0, pe100A  = 1.0, pe100B  = 1.0, pe143A  = 1.0,
                    pe143B  = 1.0, pe217A  = 1.0, pe217B  = 1.0
                ),
                HillipopDust(AdustT=1.0, AdustP=1.0, beta_dustT=1.5, beta_dustP=1.5),
                HillipopSZ(Atsz=1.0, Aksz=1.0),
                HillipopCIB(Acib=1.0, beta_cib=1.75, xi=1.0),
                HillipopPointSources(Aradio=1.0, beta_radio=-0.7, Adusty=1.0),
                HillipopSubPixel(
                    Asbpx_100x100 = 0.0, Asbpx_100x143 = 0.0, Asbpx_100x217 = 0.0,
                    Asbpx_143x143 = 0.0, Asbpx_143x217 = 0.0, Asbpx_217x217 = 0.0
                )
            )

            # 3. Run CLASS
            class_input = Dict(
                "H0"            => cosmo_params.H0,
                "omega_b"       => cosmo_params.omega_b,
                "omega_cdm"     => cosmo_params.omega_cdm,
                "A_s"           => cosmo_params.A_s,
                "n_s"           => cosmo_params.n_s,
                "tau_reio"      => cosmo_params.tau_reio,
                "N_ur"          => cosmo_params.N_ur,
                "N_ncdm"        => 1,
                "m_ncdm"        => cosmo_params.m_ncdm,
                "output"        => "tCl, pCl, lCl",
                "lensing"       => "yes",
                "l_max_scalars" => 3000
            )

            prob = CLASSProblem(class_input...)
            sol = CLASS.solve(prob)
            cls_df = sol["lCl"]

            # 4. Post-Process 
            ell = cls_df.l
            T_cmb = 2.7255
            valid_idx = ell .>= 2
            ell_valid = ell[valid_idx]
            conversion = @. (2 * pi / (ell_valid * (ell_valid + 1))) * T_cmb^2

            Cl_TT = cls_df.TT[valid_idx] .* conversion
            Cl_TE = cls_df.TE[valid_idx] .* conversion
            Cl_EE = cls_df.EE[valid_idx] .* conversion

            h = load_hillipop()
            idx_lmax = findfirst(==(h.lmax), ell_valid)
            Cl_TT = Cl_TT[1:idx_lmax]
            Cl_TE = Cl_TE[1:idx_lmax]
            Cl_EE = Cl_EE[1:idx_lmax]

            # 5. Compute Log-Likelihood
            loglike = compute_loglike(Cl_TT, Cl_TE, Cl_EE, nuisance, h)
            chisq   = -2 * loglike
            
            # Assert finite loglikelihood
            @test isfinite(loglike)

            # Assert match with JAX reference within 10%
            reference_loglike = -18693.669380
            reference_chisq = -2 * reference_loglike
            @test isapprox(chisq, reference_chisq, rtol=0.1)

            # 6. Test change in chi-square for higher A_s
            class_input_high_As = copy(class_input)
            class_input_high_As["A_s"] = cosmo_params.A_s * 1.10 # +10% A_s
            
            prob_high = CLASSProblem(class_input_high_As...)
            sol_high = CLASS.solve(prob_high)
            cls_df_high = sol_high["lCl"]

            ell_high = cls_df_high.l
            valid_idx_high = ell_high .>= 2
            ell_valid_high = ell_high[valid_idx_high]
            conversion_high = @. (2 * pi / (ell_valid_high * (ell_valid_high + 1))) * T_cmb^2

            Cl_TT_high = cls_df_high.TT[valid_idx_high] .* conversion_high
            Cl_TE_high = cls_df_high.TE[valid_idx_high] .* conversion_high
            Cl_EE_high = cls_df_high.EE[valid_idx_high] .* conversion_high

            idx_lmax_high = findfirst(==(h.lmax), ell_valid_high)
            Cl_TT_high = Cl_TT_high[1:idx_lmax_high]
            Cl_TE_high = Cl_TE_high[1:idx_lmax_high]
            Cl_EE_high = Cl_EE_high[1:idx_lmax_high]

            loglike_high = compute_loglike(Cl_TT_high, Cl_TE_high, Cl_EE_high, nuisance, h)
            chisq_high = -2 * loglike_high

            # Since the reference is near best-fit, +10% A_s should worsen (increase) the chi-square.
            @test chisq_high > chisq
        end
    end
end
