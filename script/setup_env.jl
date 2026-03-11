using Pkg

# Activates the local script/ environment
pwd_dir = @__DIR__
Pkg.activate(pwd_dir)

# Add standard dependencies
Pkg.add([
    PackageSpec(name="Turing"),
    PackageSpec(name="AbstractCosmologicalEmulators"),
    PackageSpec(name="Serialization"),
    PackageSpec(url="https://github.com/CosmologicalEmulators/Capse.jl", rev="develop")
])

# Add the local Hillipop.jl parent directory as a development dependency
Pkg.develop(path=joinpath(pwd_dir, ".."))

println("Environment setup complete! You can now run `julia --project=. run_hillipop_chains.jl`")
