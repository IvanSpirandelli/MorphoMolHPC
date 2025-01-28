using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

using MorphoMol
using JLD2
using LinearAlgebra
using Rotations

function cc_rwm_ma_call(
    config_string::String
    )
    if occursin("n_mol=2;", config_string)
        cc_for_two(config_string)
    else
        cc_for_more_than_two(config_string)
    end
end

function cc_for_more_than_two(config_string::String)
    eval(Meta.parse(config_string))
    template_centers = MorphoMol.TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.TEMPLATES[mol_type]["template_radii"]

    x_init = x # For some obscure reason having 'x_init' in the config string throws an  error. Just as does bounds.
    if length(x_init) == 0
        x_init = MorphoMol.get_initial_state(n_mol, bnds)
    end
    n_atoms_per_mol = length(template_centers) ÷ 3
    template_centers = reshape(template_centers,(3,n_atoms_per_mol))
    radii = vcat([template_radii for i in 1:n_mol]...);

    β = 1.0 / T
    pf = MorphoMol.Energies.get_prefactors(rs, η)

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
        "comment" => "live action role play",
    )


    output = Dict{String, Vector}(
        "states" => Vector{Vector{Float64}}([]),
        "Es" => Vector{Float64}([]), 
        "Vs" => Vector{Float64}([]), 
        "As" => Vector{Float64}([]), 
        "Cs" => Vector{Float64}([]), 
        "Xs" => Vector{Float64}([]),
        "OLs" => Vector{Float64}([]),
        "αs" => Vector{Float32}([]),
    )

    perturbation_call(x) = MorphoMol.get_index_and_perturb_single_randomly_chosen(x, σ_r, σ_t)

    ssu_energy, ssu_measures = MorphoMol.get_single_subunit_energy_and_measures(mol_type, rs, pf, overlap_jump, overlap_slope, delaunay_eps)
    bol_nmol(x, i, j) = MorphoMol.are_bounding_spheres_overlapping(x, i, j, MorphoMol.get_bounding_radius(mol_type))

    initial_connected_component_call(x) = MorphoMol.get_initial_connected_component_energies(
        x,
        template_centers, 
        template_radii,
        rs, 
        pf, 
        overlap_jump,
        overlap_slope,
        delaunay_eps,
        bol_nmol
    )
        
    energy_call(previous_ccs::Dict{Vector{Int}, Tuple{Float64, Dict{String, Any}}}, i::Int, x::Vector{Float64}) = 
        MorphoMol.connected_component_wise_solvation_free_energy_and_measures_in_bounds!(
            previous_ccs,
            i,
            x,
            template_centers, 
            template_radii,
            rs, 
            pf, 
            overlap_jump,
            overlap_slope,
            bnds,
            delaunay_eps,
            ssu_energy,
            ssu_measures,
            bol_nmol
        )

    rwm = MorphoMol.Algorithms.ConnectedComponentRandomWalkMetropolis(energy_call, perturbation_call, initial_connected_component_call, β)

    MorphoMol.Algorithms.simulate!(rwm, x_init, simulation_time_minutes, output);

    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" input output
end

function cc_for_two(config_string::String)
    eval(Meta.parse(config_string))
    template_centers = MorphoMol.TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.TEMPLATES[mol_type]["template_radii"]

    x_init = x # For some obscure reason having 'x_init' in the config string throws an  error. Just as does bounds.
    if length(x_init) == 0
        x_init = MorphoMol.get_initial_state(n_mol, bnds)
    end
    n_atoms_per_mol = length(template_centers) ÷ 3
    template_centers = reshape(template_centers,(3,n_atoms_per_mol))
    radii = vcat([template_radii for i in 1:n_mol]...);

    β = 1.0 / T
    pf = MorphoMol.Energies.get_prefactors(rs, η)

    ssu_energy, ssu_measures = MorphoMol.get_single_subunit_energy_and_measures(mol_type, rs, pf, overlap_jump, overlap_slope, delaunay_eps)
    bol_nmol(x) = MorphoMol.are_bounding_spheres_overlapping(x, 1, 2, MorphoMol.get_bounding_radius(mol_type))
    energy(x) = MorphoMol.solvation_free_energy_and_measures_in_bounds(
        x,
        template_centers,
        radii,
        rs,
        pf,
        overlap_jump,
        overlap_slope,
        bnds, 
        delaunay_eps,
        ssu_energy,
        ssu_measures,
        bol_nmol
        )

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
        "αs" => Vector{Float32}([]),
    )

    MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), simulation_time_minutes, output);

    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" input output
end