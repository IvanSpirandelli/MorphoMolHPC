using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
using MorphoMol
using JLD2
using LinearAlgebra
using Rotations
using PyCall

function rwm_only_persistence_call(
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

    energy(x) = interface_persistence_in_bounds(x, template_centers, persistence_weight, bnds, delaunay_eps)
    perturbation(x) = perturb_single_randomly_chosen(x, σ_r, σ_t)
    #perturbation(x) = perturb_all(x, Σ)

    rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, β)

    input = Dict(
        "template_centers" => template_centers,
        "template_radii" => template_radii,
        "n_mol" => n_mol,
        "σ_r" => σ_r,
        "σ_t" => σ_t,
        "persistence_weight" => persistence_weight,
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
        "IDGMs" => Vector{Any}([]),
        "αs" => Vector{Float32}([]),
    )

    MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), simulation_time_minutes, output);
    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" input output
end

perturb_all(x, Σ) = x .+ (randn(length(x)) .* Σ)

function perturb_single_randomly_chosen(x, σ_r, σ_t)
    x_cand = deepcopy(x)
    i  = rand(0:(length(x)÷6)-1)
    x_cand[(i*6)+1:(i*6)+6] = x_cand[(i*6)+1:(i*6)+6] .+ randn(6) .* [σ_r, σ_r, σ_r, σ_t, σ_t, σ_t]
    x_cand
end

function interface_persistence_in_bounds(x::Vector{Float64}, template_centers::Matrix{Float64}, persistence_weight::Float64, bounds::Float64, delaunay_eps::Float64)
    if any(0.0 >= e || e >= bounds for e in x[4:6:end]) || any(0.0 >= e || e >= bounds for e in x[5:6:end]) || any(0.0 >= e || e >= bounds for e in x[6:6:end])
        return Inf, Dict("Vs" => Inf, "As" => Inf, "Cs" => Inf, "Xs" => Inf, "OLs" => Inf, "IDGMs"  => nothing)
    end
    n_atoms_per_mol = size(template_centers)[2]
    flat_realization = MorphoMol.Utilities.get_flat_realization(x, template_centers)
    points = Vector{Vector{Float64}}([e for e in eachcol(reshape(flat_realization, (3, Int(length(flat_realization) / 3))))])
    idgm = MorphoMol.Energies.get_interface_diagram(points, n_atoms_per_mol)
    idgm = [idgm[1], idgm[2], idgm[3], idgm[4]]
    cp2 = sum(idgm[2][:,2] - idgm[2][:,1])
    persistence_weight * cp2, Dict("Vs" => 0.0, "As" =>0.0, "Cs" => 0.0, "Xs" => 0.0, "OLs" =>0.0, "IDGMs"  => idgm)
end