using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
using MorphoMol
using JLD2
using LinearAlgebra
using Rotations
using PyCall

function rwm_only_interface_persistence_call(
    config_string::String
    )
    eval(Meta.parse(config_string))

    template_centers = MorphoMol.TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.TEMPLATES[mol_type]["template_radii"]
    x_init = MorphoMol.get_initial_state(n_mol, bnds)
    
    n_atoms_per_mol = length(template_centers) ÷ 3
    template_centers = reshape(template_centers,(3,n_atoms_per_mol))

    β = 1.0 / T
    Σ = vcat([[σ_r, σ_r, σ_r, σ_t, σ_t, σ_t] for _ in 1:n_mol]...)

    energy(x) = MorphoMol.death_by_birth_interface_persistence(x, template_centers, persistence_weights)
    perturbation(x) = MorphoMol.perturb_single_randomly_chosen(x, σ_r, σ_t)

    rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, β)

    input = Dict(
        "template_centers" => template_centers,
        "template_radii" => template_radii,
        "n_mol" => n_mol,
        "σ_r" => σ_r,
        "σ_t" => σ_t,
        "persistence_weights" => persistence_weights,
        "T" => T,
        "mol_type" => mol_type
    )
    
    output = Dict{String, Vector}(
        "states" => Vector{Vector{Float64}}([]),
        "Es" => Vector{Float64}([]), 
        "P0s" => Vector{Float64}([]),
        "P1s" => Vector{Float64}([]),
        "αs" => Vector{Float32}([]),
    )

    MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), simulation_time_minutes, output);
    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" input output
end