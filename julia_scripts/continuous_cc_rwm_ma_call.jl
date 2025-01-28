using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

using MorphoMol
using JLD2
using LinearAlgebra
using Rotations

function continuous_cc_rwm_ma_call(
    config_string::String
    )
    eval(Meta.parse(config_string))
    @load "$(in_out_folder)$(file)" input output

    mol_type = input["mol_type"]
    template_radii = input["template_radii"]
    template_centers = input["template_centers"]
    n_mol = input["n_mol"]
    rs = input["rs"]
    pf = input["prefactors"]
    T = input["T"]
    bnds = input["bounds"]
    σ_r = input["σ_r"]
    σ_t = input["σ_t"]
    overlap_slope = input["overlap_slope"]
    overlap_jump = 0.0
    delaunay_eps = 100.0

    x_init = deepcopy(output["states"][end])

    radii = vcat([template_radii for i in 1:n_mol]...);

    β = 1.0 / T
    
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
    
    input["simulation_time_minutes"] += simulation_time_minutes

    MorphoMol.Algorithms.simulate!(rwm, x_init, simulation_time_minutes, output);

    @save "$(in_out_folder)$(file)" input output
end