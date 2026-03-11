using JSON
using Hillipop

function main()
    h = load_hillipop()

    # Load JAX reference data
    ref_file = joinpath(dirname(@__DIR__), "test", "reference_jax.json")
    if !isfile(ref_file)
        println("Reference file not found!")
        return
    end

    ref = JSON.parsefile(ref_file)
    ClTT = Float64.(ref["ClTT"])
    ClEE = Float64.(ref["ClEE"])
    ClTE = Float64.(ref["ClTE"])
    
    # Convert string keys to Symbol
    pars = Dict{Symbol, Float64}()
    for (k, v) in ref["params"]
        pars[Symbol(k)] = Float64(v)
    end
    
    logL_jax = Float64(ref["logL"])
    
    # Compute in Julia
    logL_jl = compute_loglike(ClTT, ClTE, ClEE, pars, h)
    
    println("JAX logL:   ", logL_jax)
    println("Julia logL: ", logL_jl)
    
    diff = abs(logL_jax - logL_jl)
    println("Abs Diff:   ", diff)
    
    if diff < 1e-2
        println("SUCCESS: Numerical equivalence verified!")
    else
        println("FAILURE: Mismatch!")
    end
end

main()
