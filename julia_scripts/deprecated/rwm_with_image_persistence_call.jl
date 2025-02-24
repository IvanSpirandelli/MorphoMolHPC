using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
using MorphoMol
using JLD2
using LinearAlgebra
using Rotations
using PyCall

function rwm_with_image_persistence_call(
    config_string::String
    )
    eval(Meta.parse(config_string))

    template_centers = MorphoMol.TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.TEMPLATES[mol_type]["template_radii"]
    x_init = MorphoMol.get_initial_state(n_mol, bnds)
    
    n_atoms_per_mol = length(template_centers) ÷ 3
    template_centers = reshape(template_centers,(3,n_atoms_per_mol))
    radii = vcat([template_radii for i in 1:n_mol]...);

    charged_indices, pos_subcomplex_indices, neg_subcomplex_indices = MorphoMol.get_charged_and_subcomplex_indices(mol_type, n_mol)
    subcomplex_indices = Vector{Int}([])
    if subcomplex_selection == "positive_in_all_charged"
        subcomplex_indices = pos_subcomplex_indices
    elseif subcomplex_selection == "negative_in_all_charged"
        subcomplex_indices = neg_subcomplex_indices
    end

    β = 1.0 / T
    pf = MorphoMol.Energies.get_prefactors(rs, η)
    Σ = vcat([[σ_r, σ_r, σ_r, σ_t, σ_t, σ_t] for _ in 1:n_mol]...)

    energy(x) = MorphoMol.solvation_free_energy_with_image_persistence_and_measures(x, template_centers, radii, rs, pf, 0.0, overlap_slope, persistence_weights, charged_indices, subcomplex_indices, delaunay_eps)
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
        "white_bear_prefactors" => pf,
        "overlap_slope" => overlap_slope,
        "persistence_weights" => persistence_weights,
        "charged_indices" => charged_indices,
        "subcomplex_indices" => subcomplex_indices,
        "T" => T,
        "mol_type" => mol_type
    )
    
    output = Dict{String, Vector}(
        "states" => Vector{Vector{Float64}}([]),
        "Es" => Vector{Float64}([]), 
        "Vs" => Vector{Float64}([]), 
        "As" => Vector{Float64}([]), 
        "Cs" => Vector{Float64}([]), 
        "Xs" => Vector{Float64}([]),
        "OLs" => Vector{Float64}([]),
        "i0s" => Vector{Float64}([]),
        "i1s" => Vector{Float64}([]),
        "i2s" => Vector{Float64}([]),
        "αs" => Vector{Float32}([]),
    )

    MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), simulation_time_minutes, output);

    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" input output
end