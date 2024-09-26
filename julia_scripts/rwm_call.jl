using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

using MorphoMol
using JLD2
using LinearAlgebra
using Rotations

function rwm_call(
    config_string::String
    )

    eval(Meta.parse(config_string))

    template_centers = MorphoMol.Utilities.TMV_TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.Utilities.TMV_TEMPLATES[mol_type]["template_radii"]
    x_init = MorphoMol.Utilities.get_initial_state(n_mol, bnds)
    
    n_atoms_per_mol = length(template_centers) ÷ 3
    template_centers = reshape(template_centers,(3,n_atoms_per_mol))
    radii = vcat([template_radii for i in 1:n_mol]...);

    β = 1.0 / T
    pf = MorphoMol.Energies.get_prefactors(rs, η)
    Σ = vcat([[σ_r, σ_r, σ_r, σ_t, σ_t, σ_t] for _ in 1:n_mol]...)

    energy(x) = solvation_free_energy_and_measures_in_bounds(x, template_centers, radii, rs, pf, 0.0, overlap_slope, bnds, delaunay_eps)
    perturbation(x) = perturb_single_randomly_chosen(x, σ_r, σ_t)
    #perturbation(x) = perturb_all(x, Σ)

    rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, β)

    input = MorphoMol.Algorithms.MorphometricSimulationInput(
        template_centers,
        template_radii,
        n_mol,
        σ_r,
        σ_t,
        rs,
        η,
        pf,
        0.0,
        overlap_slope,
        T,
        0.0,
        0
    )

    output = MorphoMol.Algorithms.MorphometricSimulationOutput(
        Vector{Vector{Float64}}([]),
        Vector{Float64}([]),
        Vector{Float32}([]),
        Vector{Float32}([]),
        Vector{Float32}([]),
        Vector{Float32}([]),
        Vector{Float32}([]),
        Vector{Float32}([])
    )

    MorphoMol.Algorithms.simulate!(rwm, output, deepcopy(x_init), simulation_time_minutes);

    in_out_data = MorphoMol.Algorithms.SimulationData(input, output)
    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" in_out_data
end

perturb_all(x, Σ) = x .+ (randn(length(x)) .* Σ)

function perturb_single_randomly_chosen(x, σ_r, σ_t)
    x_cand = deepcopy(x)
    i  = rand(0:(length(x)÷6)-1)
    x_cand[(i*6)+1:(i*6)+6] = x_cand[(i*6)+1:(i*6)+6] .+ randn(6) .* [σ_r, σ_r, σ_r, σ_t, σ_t, σ_t]
    x_cand
end

function solvation_free_energy(x::Vector{Float64}, template_centers::Matrix{Float64}, radii::Vector{Float64}, rs::Float64, prefactors::AbstractVector, overlap_jump::Float64, overlap_slope::Float64, delaunay_eps::Float64)
    n_mol = length(x) ÷ 6
    n_atoms_per_mol = size(template_centers)[2]
    flat_realization = MorphoMol.Utilities.get_flat_realization(x, template_centers)
    MorphoMol.Energies.solvation_free_energy(flat_realization, n_atoms_per_mol, radii, rs, prefactors, overlap_jump, overlap_slope, delaunay_eps)
end

function solvation_free_energy_and_measures(x::Vector{Float64}, template_centers::Matrix{Float64}, radii::Vector{Float64}, rs::Float64, prefactors::AbstractVector, overlap_jump::Float64, overlap_slope::Float64, delaunay_eps::Float64)
    n_mol = length(x) ÷ 6
    n_atoms_per_mol = size(template_centers)[2]
    flat_realization = MorphoMol.Utilities.get_flat_realization(x, template_centers)
    measures = MorphoMol.Energies.get_geometric_measures_and_overlap_value(flat_realization, n_atoms_per_mol, radii, rs, overlap_jump, overlap_slope, delaunay_eps)
    sum(measures .* [prefactors; 1.0]), measures
end

function solvation_free_energy_and_measures_in_bounds(x::Vector{Float64}, template_centers::Matrix{Float64}, radii::Vector{Float64}, rs::Float64, prefactors::AbstractVector, overlap_jump::Float64, overlap_slope::Float64, bounds::Float64, delaunay_eps::Float64)
    if any(0.0 >= e || e >= bounds for e in x[4:6:end]) || any(0.0 >= e || e >= bounds for e in x[5:6:end]) || any(0.0 >= e || e >= bounds for e in x[6:6:end])
        return Inf, [Inf, Inf, Inf, Inf, Inf]
    end
    n_mol = length(x) ÷ 6
    n_atoms_per_mol = size(template_centers)[2]
    flat_realization = MorphoMol.Utilities.get_flat_realization(x, template_centers)
    measures = MorphoMol.Energies.get_geometric_measures_and_overlap_value(flat_realization, n_atoms_per_mol, radii, rs, overlap_jump, overlap_slope, delaunay_eps)
    sum(measures .* [prefactors; 1.0]), measures
end