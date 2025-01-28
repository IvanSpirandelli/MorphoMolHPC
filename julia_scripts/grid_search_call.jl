using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
using MorphoMol
using Random
using JLD2

function grid_search_call(config_string::String)
    eval(Meta.parse(config_string))
    template_centers = MorphoMol.TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.TEMPLATES[mol_type]["template_radii"];

    x_init = MorphoMol.get_initial_state(n_mol, bnds)
    radii = vcat([template_radii for i in 1:n_mol]...);

    energy(x) = MorphoMol.persistence(x, template_centers, persistence_weights)
    perturbation(x) = MorphoMol.perturb_single_randomly_chosen(x, σ_r, σ_t)

    input = Dict(
        "template_centers" => template_centers,
        "template_radii" => template_radii,
        "n_mol" => n_mol,
        "σ_r" => σ_r,
        "σ_t" => σ_t,
        "persistence_weights" => persistence_weights,
        "T" => T_search,
        "mol_type" => mol_type
    )

    output = Dict{String, Vector}(
        "states" => Vector{Vector{Float64}}([]),
        "Es" => Vector{Float64}([]),
        "αs" => Vector{Float32}([]),
        "P0s" => Vector{Float64}([]),
        "P1s" => Vector{Float64}([]),
        "P2s" => Vector{Float64}([]),
    )

    rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, 1.0 / T_search)
    MorphoMol.Algorithms.simulate!(rwm, x_init, search_time_minutes, output);

    T_sim = MorphoMol.calculate_T0(output["Es"], 0.24)

    input["T"] = T_sim
    output = Dict{String, Vector}(
        "states" => Vector{Vector{Float64}}([]),
        "Es" => Vector{Float64}([]),
        "αs" => Vector{Float32}([]),
        "P0s" => Vector{Float64}([]),
        "P1s" => Vector{Float64}([]),
        "P2s" => Vector{Float64}([]),
    )

    rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, 1.0 / T_sim)
    MorphoMol.Algorithms.simulate!(rwm, x_init, simulation_time_minutes, output);
    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" input output
end