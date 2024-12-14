using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
using MorphoMol
using Random
using JLD2

function perturb_single_randomly_chosen(x, σ_r, σ_t)
    x_cand = deepcopy(x)
    i  = rand(0:(length(x)÷6)-1)
    x_cand[(i*6)+1:(i*6)+6] = x_cand[(i*6)+1:(i*6)+6] .+ (randn(6) .* [σ_r, σ_r, σ_r, σ_t, σ_t, σ_t])
    x_cand
end

function persistence(x::Vector{Float64}, template_centers::Matrix{Float64}, persistence_weights::Vector{Float64})
    flat_realization = MorphoMol.Utilities.get_flat_realization(x, template_centers)
    points = Vector{Vector{Float64}}([e for e in eachcol(reshape(flat_realization, (3, Int(length(flat_realization) / 3))))])
    pdgm = MorphoMol.Energies.get_alpha_shape_persistence_diagram(points)
    p0 = MorphoMol.Energies.get_total_persistence(pdgm[1], persistence_weights[1])
    p1 = MorphoMol.Energies.get_total_persistence(pdgm[2], persistence_weights[2])
    p2 = MorphoMol.Energies.get_total_persistence(pdgm[3], persistence_weights[3])
    p0 + p1 + p2, Dict{String, Any}("P0s" => p0, "P1s" => p1, "P2s" => p2)
end

function calculate_T0(Es, T_search, target_acceptance_rate)
    transitions = []
    for i in 1:length(Es)-1
        if Es[i] > Es[i+1]
            push!(transitions, Es[i])
            push!(transitions, Es[i+1])
        end
    end

    chi_bar(T) = sum([exp(-transitions[i]/T) for i in 1:2:length(transitions)-1])/sum([exp(-transitions[i]/T) for i in 2:2:length(transitions)])
    χ_0 = target_acceptance_rate
    T_0 = T_search
    try
        while abs(chi_bar(T_0) - χ_0) > 0.00001
            T_0 = T_0 * (log(chi_bar(T_0)) / log(χ_0 ))
        end
    catch 
        println("No energy decreasing transitions found!")
    end
    (isnan(T_0) || T_0 <= 0) ? T_search : T_0
end

function grid_search_call(config_string::String)
    eval(Meta.parse(config_string))
    template_centers = MorphoMol.Utilities.TMV_TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.Utilities.TMV_TEMPLATES[mol_type]["template_radii"];

    x_init = MorphoMol.Utilities.get_initial_state(n_mol, bnds)
    radii = vcat([template_radii for i in 1:n_mol]...);

    energy(x) = persistence(x, template_centers, persistence_weights)
    perturbation(x) = perturb_single_randomly_chosen(x, σ_r, σ_t)

    input = Dict(
        "template_centers" => template_centers,
        "template_radii" => template_radii,
        "n_mol" => n_mol,
        "σ_r" => σ_r,
        "σ_t" => σ_t,
        "persistence_weights" => persistence_weights,
        "T" => T_search,
        "mol_type" => mol_type
    )

    output = Dict{String, Vector}(
        "states" => Vector{Vector{Float64}}([]),
        "Es" => Vector{Float64}([]),
        "αs" => Vector{Float32}([]),
        "P0s" => Vector{Float64}([]),
        "P1s" => Vector{Float64}([]),
        "P2s" => Vector{Float64}([]),
    )

    rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, 1.0 / T_search)
    MorphoMol.Algorithms.simulate!(rwm, x_init, search_time_minutes, output);

    T_sim = calculate_T0(output["Es"], T_search, 0.24)

    input["T"] = T_sim
    output = Dict{String, Vector}(
        "states" => Vector{Vector{Float64}}([]),
        "Es" => Vector{Float64}([]),
        "αs" => Vector{Float32}([]),
        "P0s" => Vector{Float64}([]),
        "P1s" => Vector{Float64}([]),
        "P2s" => Vector{Float64}([]),
    )

    rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, 1.0 / T_sim)
    MorphoMol.Algorithms.simulate!(rwm, x_init, simulation_time_minutes, output);
    mkpath(output_directory)
    @save "$(output_directory)/$(name).jld2" input output
end