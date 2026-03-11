"""
    Hillipop

Julia implementation of the Planck PR4 Hillipop high-ℓ TTTEEE likelihood.

This package provides a faithful translation of the JAX `HillipopPR4` likelihood,
reusing `CMBForegrounds.jl` for foreground models.

# Main entry points
- `load_hillipop(data_dir)` — load all data files, return a `HillipopData` struct
- `compute_loglike(ClTT, ClTE, ClEE, pars, h::HillipopData)` — evaluate log L
- `build_residual_vector(ClTT, ClTE, ClEE, pars, h)` — flat residual vector (for Turing)
- `HillipopPars{T}` — typed nuisance parameter container (for Turing)

# Units
- Input Cl in K² (ℓ=0 at index 1, starting from ℓ=2)
- Internal working in μK² as D_ℓ = ℓ(ℓ+1)/(2π)·C_ℓ
"""
module Hillipop

using LinearAlgebra
using DelimitedFiles
using CMBForegrounds
using FITSIO
using Artifacts
using NPZ
include("structs.jl")
include("data_io.jl")
include("foreground_dispatch.jl")
include("residuals.jl")
include("likelihood.jl")

export HillipopData, HillipopPars
export load_hillipop, compute_loglike, build_residual_vector

end # module Hillipop
