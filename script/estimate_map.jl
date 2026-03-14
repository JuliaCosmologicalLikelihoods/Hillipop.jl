using Pkg
Pkg.activate(dirname(@__DIR__))

using Hillipop
using Capse
using Turing
using ADTypes
using LinearAlgebra
using AbstractCosmologicalEmulators
using Artifacts
using Mooncake
using Optim

# ── Section 1: Setup ─────────────────────────────────────────────────────────

println("Loading data and emulators...")
const h_data = load_hillipop()
const lmax_h = h_data.lmax

const artifacts_toml = joinpath(dirname(pathof(Capse)), "..", "Artifacts.toml")
const path = artifact_path(artifact_hash("CAMB_LCDM", artifacts_toml))
const emu_TT = Capse.load_emulator(joinpath(path, "TT/"), emu=LuxEmulator)
const emu_TE = Capse.load_emulator(joinpath(path, "TE/"), emu=LuxEmulator)
const emu_EE = Capse.load_emulator(joinpath(path, "EE/"), emu=LuxEmulator)

const ell = emu_TT.ℓgrid[3:5001]
const idx_lmax = findfirst(==(lmax_h), ell)
const fac = @. 2π / (ell[1:idx_lmax] * (ell[1:idx_lmax] + 1)) * 1e-12

# ── Section 2: Model ─────────────────────────────────────────────────────────

@model function hillipop_production_model(h, emu_TT, emu_TE, emu_EE, idx_lmax, fac)
    # --- Cosmological Parameters ---
    ln10As    ~ Uniform(2.5, 3.5)
    ns        ~ Uniform(0.88, 1.06)
    H0        ~ Uniform(60.0, 80.0)
    omega_b   ~ Uniform(0.019, 0.025)
    omega_cdm ~ Uniform(0.08, 0.20)
    tau_reio  ~ Normal(0.0506, 0.0086)

    # --- Nuisance Parameters ---
    A_planck   ~ Normal(1.0, 0.0025)
    cal100A    ~ Uniform(0.9, 1.1)
    cal100B    ~ Uniform(0.9, 1.1)
    cal143B    ~ Uniform(0.9, 1.1)
    cal217A    ~ Uniform(0.9, 1.1)
    cal217B    ~ Uniform(0.9, 1.1)
    AdustT     ~ Normal(1.0, 0.1)
    AdustP     ~ Normal(1.0, 0.1)
    beta_dustT ~ Normal(1.51, 0.01)
    beta_dustP ~ Normal(1.59, 0.02)
    Atsz       ~ Uniform(0.0, 50.0)
    Aksz       ~ Uniform(0.0, 50.0)
    Acib       ~ Uniform(0.0, 20.0)
    beta_cib   ~ Normal(1.75, 0.06)
    xi         ~ Uniform(-1.0, 1.0)
    Aradio     ~ Uniform(0.0, 150.0)
    Adusty     ~ Uniform(0.0, 150.0)

    nuisance = (
        A_planck = A_planck,
        cal100A = cal100A, cal100B = cal100B, cal143B = cal143B,
        cal217A = cal217A, cal217B = cal217B,
        AdustT = AdustT, AdustP = AdustP,
        beta_dustT = beta_dustT, beta_dustP = beta_dustP,
        Atsz = Atsz, Aksz = Aksz,
        Acib = Acib, beta_cib = beta_cib, xi = xi,
        Aradio = Aradio, Adusty = Adusty
    )

    x_cosmo = [ln10As, ns, H0, omega_b, omega_cdm, tau_reio]
    raw_TT = Capse.get_Cℓ(x_cosmo, emu_TT)
    raw_TE = Capse.get_Cℓ(x_cosmo, emu_TE)
    raw_EE = Capse.get_Cℓ(x_cosmo, emu_EE)

    Cl_TT = raw_TT[1:idx_lmax] .* fac
    Cl_TE = raw_TE[1:idx_lmax] .* fac
    Cl_EE = raw_EE[1:idx_lmax] .* fac

    Turing.@addlogprob! compute_loglike(Cl_TT, Cl_TE, Cl_EE, nuisance, h)
end

# ── Section 3: MAP Estimation ───────────────────────────────────────────────

println("Initializing MAP estimation with Mooncake...")
model = hillipop_production_model(h_data, emu_TT, emu_TE, emu_EE, idx_lmax, fac)

# Fiducial initial parameters
init_params = (
    ln10As = 3.044, ns = 0.9649, H0 = 67.36, omega_b = 0.02237, omega_cdm = 0.1200, tau_reio = 0.0544,
    A_planck = 1.0, cal100A = 1.0, cal100B = 1.0, cal143B = 1.0, cal217A = 1.0, cal217B = 1.0,
    AdustT = 1.0, AdustP = 1.0, beta_dustT = 1.51, beta_dustP = 1.59,
    Atsz = 1.0, Aksz = 1.0, Acib = 1.0, beta_cib = 1.75, xi = 0.0,
    Aradio = 1.0, Adusty = 1.0
)

# --- MAP Estimation Loop ---
for run_idx in 1:10
    println("\n" * "="^60)
    println("MAP RUN #$run_idx")
    println("="^60)

    # Define a callback to print parameters at each iteration
    # Note: Optimization happens in unconstrained space
    # The callback signature for the OptimizationBase interface is (state, value)
    function callback(state, value)
        println("Iteration $(state.iter):")
        println("  Objective: $(value)")
        return false
    end

    println("Starting optimization...")
    # Turing passes these down to Optim.
    # Note: Optim.Options has to be passed as a keyword 'options' to maximum_a_posteriori
    # according to typical Turing optimisation interface.
    map_estimate = maximum_a_posteriori(
        model,
        LBFGS(),
        initial_params=InitFromParams(init_params),
        adtype=AutoMooncake(),
        callback=callback,
        iterations=100,
        show_trace=true
    )

    println("\nMAP ESTIMATE RESULTS (RUN $run_idx)")
    display(map_estimate)

    println("\nFinal Parameter values at MAP:")
    # Access values via .params
    for (k, v) in pairs(map_estimate.params)
        println("  $k: $v")
    end
end
