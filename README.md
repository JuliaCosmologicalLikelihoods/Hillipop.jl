# Hillipop.jl

[![Build Status](https://github.com/JuliaCosmologicalLikelihoods/Hillipop.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaCosmologicalLikelihoods/Hillipop.jl/actions/workflows/CI.yml)

A high-performance, fully differentiable Julia implementation of the Planck PR4 Hillipop high-ℓ TTTEEE likelihood. 

This package enables efficient evaluation of the cosmic microwave background (CMB) high-ℓ polarization likelihood and supports seamless integration with Auto-Differentiation (AD) frameworks (e.g., ForwardDiff, Zygote, Mooncake), making it ideal for Hamiltonian Monte Carlo (HMC) sampling and variational inference in modern cosmological analyses.

## Author
**Marco Bonici**  
Postdoctoral Researcher at the Waterloo Centre for Astrophysics.

## Acknowledgements
This package is a faithful, Julia-native translation of the JAX-based [`jax-loglike`](https://github.com/alexander-reeves/jax-loglike) repository.

## Features
- **Faithful Translation:** Primal (forward) calculations exactly match the reference JAX and original Python implementations.
- **AutoDiff Ready:** Supports end-to-end differentiability through DifferentiationInterface.jl. Highly optimized custom reverse-rules (`rrule!!`) are provided for Mooncake/ChainRulesCore to guarantee exceptional performance and memory efficiency during gradient evaluations.
- **Modular Architecture:** Foreground modeling is seamlessly delegated to `CMBForegrounds.jl`.
- **Turing.jl Compatible:** Typed nuisance parameter containers and straightforward flat residual outputs make it simple to wrap inside probabilistic programming paradigms.

## Usage Overview

```julia
using Hillipop

# 1. Load data (automatically downloads the required artifact if not present)
h = load_hillipop()

# 2. Setup your parameters and theory spectra
pars = HillipopNuisance(A_planck = 1.0, AdustT = 1.0, Atsz = 1.0, Acib = 1.0) # ... other params

# Provide theory Cl vectors (in K², starting at ℓ=2)
ClTT = ... 
ClTE = ...
ClEE = ...

# 3. Compute log-likelihood
logL = compute_loglike(ClTT, ClTE, ClEE, pars, h)
```

## Units and Conventions
- **Theory inputs (`C_ℓ`)** are expected in units of $\text{K}^2$, starting from $\ell=2$.
- **Internal computations** use $D_\ell$ in $\mu\text{K}^2$, converted automatically via $D_\ell = \frac{\ell(\ell+1)}{2\pi} C_\ell \times 10^{12}$.
