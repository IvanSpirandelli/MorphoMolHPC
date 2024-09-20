using Pkg
Pkg.activate("../../MorphoMolMonteCarlo/Project.toml")
Pkg.instantiate()

using MorphoMol
using JLD2
using LinearAlgebra
using Rotations

function hmc_call(
    config_string::String
    )

    eval(Meta.parse(config_string))
    
    @load template_file template_mol template_radii x_init
    n_atoms_per_mol = length(template_mol) ÷ 3
    n_mol = length(x_init) ÷ 6
    template_mol = reshape(template_mol,(3,n_atoms_per_mol))
    radii = vcat([template_radii for i in 1:n_mol]...);

    β = 1.0 / T
    pf = MorphoMol.Energies.get_prefactors(rs, η)
    Σ = vcat([[σ_r, σ_r, σ_r, σ_t, σ_t, σ_t] for _ in 1:n_mol]...)

    energy(x) = solvation_free_energy_and_measures(x, template_mol, radii, rs, pf, 0.0, overlap_slope, 1.0)
    energy_gradient!(∇E, x) = solvation_free_energy_gradient!(∇E, x, template_mol, radii, rs, pf, overlap_slope)

    hmc = MorphoMol.Algorithms.HamiltonianMonteCarlo(energy, energy_gradient!, MorphoMol.Algorithms.standard_leapfrog!, β, L, ε, Σ)

    input = MorphoMol.Algorithms.MorphometricSimulationInput(
        template_mol,
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
        ε,
        L
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

    MorphoMol.Algorithms.simulate!(hmc, output, deepcopy(x_init), simulation_time_minutes);

    in_out_data = MorphoMol.Algorithms.SimulationData(input, output)
    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" in_out_data
end

function get_flat_realization(x, template_mol)
    n_mol = length(x) ÷ 6
    [(hvcat((n_mol), [exp(Rotations.RotationVecGenerator(x[i:i+2]...)) * template_mol .+ x[i+3:i+5] for i in 1:6:length(x)]...)...)...]
end

function rotation_and_translation_gradient!(∇E, x, ∇FSol, template_mol)
    n_atoms_per_mol = size(template_mol)[2]
    n_mol = length(x) ÷ 6
    for i in 1:n_mol        
        R = exp(Rotations.RotationVecGenerator(x[(i-1)*6 + 1:(i-1)*6 + 3]...))
        ∇E[(i-1) * 6 + 1] = 0.5 * sum([-v[2]*(R[3,:] ⋅ w) + v[3]*(R[2,:] ⋅ w) for (v,w) in [(∇FSol[:,:,i][:,j], template_mol[:,j]) for j in 1:n_atoms_per_mol]])
        ∇E[(i-1) * 6 + 2] = 0.5 * sum([v[1]*(R[3,:] ⋅ w) - v[3]*(R[1,:] ⋅ w) for (v,w) in [(∇FSol[:,:,i][:,j], template_mol[:,j]) for j in 1:n_atoms_per_mol]])
        ∇E[(i-1) * 6 + 3] = 0.5 * sum([-v[1]*(R[2,:] ⋅ w) + v[2]*(R[1,:] ⋅ w) for (v,w) in [(∇FSol[:,:,i][:,j], template_mol[:,j]) for j in 1:n_atoms_per_mol]])
        ∇E[(i-1) * 6 + 4:(i-1) * 6 + 6] = sum([∇FSol[:,j,i] for j in 1:n_atoms_per_mol])
    end
    ∇E
end

function solvation_free_energy_gradient!(∇E, x, template_mol, radii, rs, pf, overlap_slope)
    n_atoms_per_mol = size(template_mol)[2]
    n_mol = length(x) ÷ 6
    flat_realization = get_flat_realization(x, template_mol)
    _, dvol, dsurf, dmean, dgauss, dlol = MorphoMol.Energies.get_geometric_measures_and_overlap_value_with_derivatives(
        flat_realization,
        n_atoms_per_mol,
        radii,
        rs,
        0.0,
        overlap_slope
    )
    ∇FSol = reshape(pf[1] * dvol + pf[2] * dsurf + pf[3] * dmean + pf[4] * dgauss + dlol, (3, n_atoms_per_mol, n_mol))
    rotation_and_translation_gradient!(∇E, x, ∇FSol, template_mol)
end

function solvation_free_energy(x::Vector{Float64}, template_mol::Matrix{Float64}, radii::Vector{Float64}, rs::Float64, prefactors::AbstractVector, overlap_jump::Float64, overlap_slope::Float64, delaunay_eps::Float64)
    n_mol = length(x) ÷ 6
    n_atoms_per_mol = size(template_mol)[2]
    flat_realization = get_flat_realization(x, template_mol)
    MorphoMol.Energies.solvation_free_energy(flat_realization, n_atoms_per_mol, radii, rs, prefactors, overlap_jump, overlap_slope, delaunay_eps)
end

function solvation_free_energy_and_measures(x::Vector{Float64}, template_mol::Matrix{Float64}, radii::Vector{Float64}, rs::Float64, prefactors::AbstractVector, overlap_jump::Float64, overlap_slope::Float64, delaunay_eps::Float64)
    n_mol = length(x) ÷ 6
    n_atoms_per_mol = size(template_mol)[2]
    flat_realization = get_flat_realization(x, template_mol)
    measures = MorphoMol.Energies.get_geometric_measures_and_overlap_value(flat_realization, n_atoms_per_mol, radii, rs, overlap_jump, overlap_slope, delaunay_eps)
    sum(measures .* [prefactors; 1.0]), measures
end