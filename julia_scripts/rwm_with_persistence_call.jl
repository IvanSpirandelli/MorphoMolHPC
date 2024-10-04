using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
using MorphoMol
using JLD2
using LinearAlgebra
using Rotations
using PyCall

function rwm_with_persistence_call(
    config_string::String
    )

    py"""
    import sys
    print(sys.executable)
    """
    # Just making sure all these variable EXIST!
    mol_type = "6r7m"
    n_mol = 2
    rs = 1.4
    η = 0.3665
    σ_r = 0.5
    σ_t = 1.25
    overlap_jump = 0.0
    overlap_slope = 0.85
    persistence_weight = -0.1
    delaunay_eps = 100.0
    bnds = 130.0
    comment = ""
    T = 2
    simulation_time_minutes = 1.0

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

    energy(x) = solvation_free_energy_with_persistence_and_measures_in_bounds(x, template_centers, radii, rs, pf, 0.0, overlap_slope, persistence_weight, bnds, delaunay_eps)
    perturbation(x) = perturb_single_randomly_chosen(x, σ_r, σ_t)
    #perturbation(x) = perturb_all(x, Σ)

    rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, β)

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
        "persistence_weight" => persistence_weight,
        "T" => T,
        "mol_type" => mol_type
    )

    energy_measures = Dict(
        "Es" => Vector{Float64}([]), 
        "Vs" => Vector{Float64}([]), 
        "As" => Vector{Float64}([]), 
        "Cs" => Vector{Float64}([]), 
        "Xs" => Vector{Float64}([]),
        "OLs" => Vector{Float64}([]),
        "IDGMs" => Vector{Any}([]),
        )
    
    algo_measures = Dict(
        "αs" => Vector{Float32}([])
    )
    
    output = MorphoMol.Algorithms.SimulationOutput(
        Vector{Vector{Float64}}([]),
        energy_measures,
        algo_measures
    )

    MorphoMol.Algorithms.simulate!(rwm, output, deepcopy(x_init), simulation_time_minutes);
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

function solvation_free_energy_with_persistence_and_measures_in_bounds(x::Vector{Float64}, template_centers::Matrix{Float64}, radii::Vector{Float64}, rs::Float64, prefactors::AbstractVector, overlap_jump::Float64, overlap_slope::Float64, persistence_weight::Float64, bounds::Float64, delaunay_eps::Float64)
    if any(0.0 >= e || e >= bounds for e in x[4:6:end]) || any(0.0 >= e || e >= bounds for e in x[5:6:end]) || any(0.0 >= e || e >= bounds for e in x[6:6:end])
        return Inf, Dict("Vs" => Inf, "As" => Inf, "Cs" => Inf, "Xs" => Inf, "OLs" => Inf, "IDGMs"  => nothing)
    end
    n_mol = length(x) ÷ 6
    n_atoms_per_mol = size(template_centers)[2]
    flat_realization = MorphoMol.Utilities.get_flat_realization(x, template_centers)
    points = Vector{Vector{Float64}}(eachcol(reshape(flat_realization, (3, Int(length(flat_realization) / 3)))))
    println(Vector{Vector{Float64}}(points))
    idgm = MorphoMol.Energies.get_interface_diagram(points, n_atoms_per_mol)
    idgm = [idgm[1], idgm[2], idgm[3], idgm[4]]
    cp2 = sum(idgm[2][:,2] - idgm[2][:,1])
    measures = MorphoMol.Energies.get_geometric_measures_and_overlap_value(flat_realization, n_atoms_per_mol, radii, rs, overlap_jump, overlap_slope, delaunay_eps)
    measures = [measures; [cp2]]
    sum(measures .* [prefactors; [1.0, persistence_weight]]), Dict("Vs" => measures[1], "As" => measures[2], "Cs" => measures[3], "Xs" => measures[4], "OLs" => measures[5], "IDGMs"  => idgm)
end