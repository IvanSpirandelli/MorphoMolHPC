using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
using MorphoMol
using JLD2
using LinearAlgebra
using Rotations
using PyCall

function mixed_rwm_call(
    config_string::String
    )

    eval(Meta.parse(config_string))

    template_centers = MorphoMol.TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.TEMPLATES[mol_type]["template_radii"]
    x_init = MorphoMol.get_initial_state(n_mol, bnds)
    
    n_atoms_per_mol = length(template_centers) ÷ 3
    template_centers = reshape(template_centers,(3,n_atoms_per_mol))
    radii = vcat([template_radii for i in 1:n_mol]...);

    β_a = 1.0 / T_a
    β_b = 1.0 / T_b
    pf = MorphoMol.Energies.get_prefactors(rs, η)
    Σ = vcat([[σ_r, σ_r, σ_r, σ_t, σ_t, σ_t] for _ in 1:n_mol]...)

    energy_a(x) = MorphoMol.solvation_free_energy_and_measures(x, template_centers, radii, rs, pf, 0.0, overlap_slope, delaunay_eps)
    energy_b(x) = MorphoMol.persistence(x, template_centers, persistence_weights)
    perturbation(x) = MorphoMol.perturb_single_randomly_chosen(x, σ_r, σ_t)

    mrwm = MorphoMol.Algorithms.MixedEnergyRandomWalkMetropolis(energy_a, energy_b, perturbation, β_a, β_b)

    input = Dict(
        "template_centers" => template_centers,
        "template_radii" => template_radii,
        "n_mol" => n_mol,
        "σ_r" => σ_r,
        "σ_t" => σ_t,
        "rs" => rs,
        "η" => η,
        "white_bear_prefactpors" => pf,
        "overlap_slope" => overlap_slope,
        "T_a" => T_a,
        "T_b" => T_b,
        "mol_type" => mol_type
    )
    
    output = Dict{String, Vector}(
        "states" => Vector{Vector{Float64}}([]),
        "Es_a" => Vector{Float64}([]), 
        "Es_b" => Vector{Float64}([]),
        "Vs" => Vector{Float64}([]), 
        "As" => Vector{Float64}([]), 
        "Cs" => Vector{Float64}([]), 
        "Xs" => Vector{Float64}([]),
        "OLs" => Vector{Float64}([]),
        "αs_a" => Vector{Float32}([]),
        "PDGMs" => Vector{Any}([]),
        "P0" => Vector{Float64}([]),
        "P1" => Vector{Float64}([]),
        "P2" => Vector{Float64}([]),
        "αs_b" => Vector{Float32}([]),
    )

    MorphoMol.Algorithms.simulate!(mrwm, deepcopy(x_init), simulation_time_minutes, output);

    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" input output
end