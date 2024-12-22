using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

using MorphoMol
using JLD2
using LinearAlgebra
using Rotations

function hmc_call(
    config_string::String
    )
    eval(Meta.parse(config_string))

    template_centers = MorphoMol.TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.TEMPLATES[mol_type]["template_radii"]
    x_init = MorphoMol.get_initial_state(n_mol, bnds)

    n_atoms_per_mol = length(template_centers) ÷ 3
    template_centers = reshape(template_centers,(3,n_atoms_per_mol))
    radii = vcat([template_radii for i in 1:n_mol]...);

    β = 1.0 / T
    pf = MorphoMol.Energies.get_prefactors(rs, η)
    Σ = vcat([[σ_r, σ_r, σ_r, σ_t, σ_t, σ_t] for _ in 1:n_mol]...)

    energy(x) = MorphoMol.solvation_free_energy_and_measures_in_bounds(x, template_centers, radii, rs, pf, 0.0, overlap_slope, bnds, 1.0)
    energy_gradient!(∇E, x) = MorphoMol.solvation_free_energy_gradient!(∇E, x, template_centers, radii, rs, pf, overlap_slope)

    hmc = MorphoMol.Algorithms.HamiltonianMonteCarlo(energy, energy_gradient!, MorphoMol.Algorithms.standard_leapfrog!, β, L, ε, Σ)

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
        "T" => T,
        "mol_type" => mol_type,
        "ε" => ε,
        "L" => L
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

    MorphoMol.Algorithms.simulate!(hmc, deepcopy(x_init), simulation_time_minutes, output)

    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" input output
end