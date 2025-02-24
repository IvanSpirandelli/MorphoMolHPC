using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

using MorphoMol
using JLD2
using LinearAlgebra
using Rotations

function rwm_ma_call(
    config_string::String
    )
    eval(Meta.parse(config_string))
    template_centers = MorphoMol.TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.TEMPLATES[mol_type]["template_radii"]

    x_init = x # For some obscure reason having 'x_init' in the config string throws an  error. Just as does bounds.
    if length(x_init) == 0
        x_init = MorphoMol.get_initial_state(n_mol, bnds)
    end
    println(x_init)
    n_atoms_per_mol = length(template_centers) ÷ 3
    template_centers = reshape(template_centers,(3,n_atoms_per_mol))
    radii = vcat([template_radii for i in 1:n_mol]...);

    β = 1.0 / T
    pf = MorphoMol.Energies.get_prefactors(rs, η)

    energy(x) = MorphoMol.solvation_free_energy_and_measures_in_bounds(x, template_centers, radii, rs, pf, 0.0, overlap_slope, bnds, delaunay_eps)
    perturbation(x) = MorphoMol.perturb_single_randomly_chosen(x, σ_r, σ_t)

    rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, β)

    input = Dict(
        "template_centers" => template_centers,
        "template_radii" => template_radii,
        "n_mol" => n_mol,
        "σ_r" => σ_r,
        "σ_t" => σ_t,
        "rs" => rs,
        "η" => η,
        "prefactors" => pf,
        "overlap_slope" => overlap_slope,
        "T" => T,
        "mol_type" => mol_type,
        "bounds" => bnds,
        "simulation_time_minutes" => simulation_time_minutes,
        "delaunay_eps" => delaunay_eps,
        "comment" => comment,
    )
    
    output = Dict{String, Vector}(
        "states" => Vector{Vector{Float64}}([]),
        "Es" => Vector{Float64}([]), 
        "Vs" => Vector{Float64}([]), 
        "As" => Vector{Float64}([]), 
        "Cs" => Vector{Float64}([]), 
        "Xs" => Vector{Float64}([]),
        "OLs" => Vector{Float64}([]),
        "PDGMs" => Vector{Any}([]),
        "αs" => Vector{Float32}([]),
    )

    MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), simulation_time_minutes, output);

    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" input output
end