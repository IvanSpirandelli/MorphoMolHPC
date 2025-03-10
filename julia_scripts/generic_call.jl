using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
using MorphoMol
using JLD2

function generic_call(
    config_string::String
    )
    eval(Meta.parse(config_string))
    template_centers = MorphoMol.TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.TEMPLATES[mol_type]["template_radii"]

    n_atoms_per_mol = length(template_centers) ÷ 3
    template_centers = reshape(template_centers,(3,n_atoms_per_mol))
    radii = vcat([template_radii for i in 1:n_mol]...)

    prefactors = MorphoMol.Energies.get_prefactors(rs, η)

    input = Dict(
        "algorithm" => alg,
        "sa_level" => sa_level,
        "energy" => nrg,
        "perturbation" => prtbt,
        "initialization" => intl,
        "mol_type" => mol_type,
        "template_centers" => template_centers,
        "template_radii" => template_radii,
        "n_mol" => n_mol,
        "x_init" => x,
        "comment" => comment,
        "bounds" => bnds,
        "rs" => rs,
        "η" => η,
        "prefactors" => prefactors,
        "σ_r" => σ_r,
        "σ_t" => σ_t,
        "T_search_runs" => T_search_runs,
        "T_search_time" => T_search_time,
        "T" => temperature,
        "persistence_weights" => persistence_weights,
        "overlap_jump" => overlap_jump,
        "overlap_slope" => overlap_slope,
        "delaunay_eps" => delaunay_eps,
        "exact_delaunay" => exact_delaunay,
        "simulation_time_minutes" => simulation_time_minutes,
    )

    initialization = MorphoMol.get_initialization(input)
    x_init = x
    if length(x_init) == 0
        x_init = initialization()
        input["x_init"] = x_init
    end
    println(length(x_init))

    energy = MorphoMol.get_energy(input)
    perturbation = MorphoMol.get_perturbation(input)

    if isapprox(input["T"],0.0)
        T = MorphoMol.get_initial_temperature(input; n_samples = 100, scaling = 0.0025)
        input["T"] = T
    end

    search_Ts = [input["T"]]
    search_αs = Vector{Float64}([])

    α_targets = Dict(
        "sa" => 0.8,
        "rwm" => 0.24,
        "hmc" => 0.65,
    )

    α_target = α_targets[input["algorithm"]]

    for i in 1:T_search_runs
        T = search_Ts[end]
        output_search = Dict{String, Vector}(
            "states" => Vector{Vector{Float64}}([]),
            "Es" => Vector{Float64}([]), 
            "Vs" => Vector{Float64}([]), 
            "As" => Vector{Float64}([]), 
            "Cs" => Vector{Float64}([]), 
            "Xs" => Vector{Float64}([]),
            "OLs" => Vector{Float64}([]),
            "P0s" => Vector{Float64}([]),
            "P1s" => Vector{Float64}([]),
            "P2s" => Vector{Float64}([]),
            "αs" => Vector{Float32}([]),
        )
        if occursin("cc", input["energy"]) && input["n_mol"] > 2
            bol_nmol_l = (x, id1, id2) -> MorphoMol.are_bounding_spheres_overlapping(x, id1, id2, MorphoMol.get_bounding_radius(mol_type))
            get_initial_connected_components = (x) -> MorphoMol.get_initial_connected_component_energies(x, template_centers, template_radii, rs, prefactors, overlap_jump, overlap_slope, delaunay_eps, bol_nmol_l)
            cc_rwm = MorphoMol.Algorithms.ConnectedComponentRandomWalkMetropolis(energy, perturbation, get_initial_connected_components, 1/T)
            x_search = initialization()
            MorphoMol.Algorithms.simulate!(cc_rwm, x_search, T_search_time, output_search)
        else
            rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, 1/T)
            x_search = initialization()
            MorphoMol.Algorithms.simulate!(rwm, x_search, T_search_time, output_search)
        end
        α = output_search["αs"][end]
        push!(search_αs, α)

        if α > α_target
            push!(search_Ts, T * 0.5)
        else
            push!(search_Ts, T * 1.5)
        end
    end


    if length(search_αs) > 1
        println("Ts = $(search_Ts)")
        println("αs = $(search_αs)")
        input["T"] = search_Ts[argmin([abs(α - α_target) for α in search_αs])]
        input["T_search_αs"] = search_αs
        input["T_search_Ts"] = search_Ts
    end

    output = Dict{String, Vector}(
        "states" => Vector{Vector{Float64}}([]),
        "Es" => Vector{Float64}([]), 
        "Vs" => Vector{Float64}([]), 
        "As" => Vector{Float64}([]), 
        "Cs" => Vector{Float64}([]), 
        "Xs" => Vector{Float64}([]),
        "OLs" => Vector{Float64}([]),
        "P0s" => Vector{Float64}([]),
        "P1s" => Vector{Float64}([]),
        "P2s" => Vector{Float64}([]),
        "αs" => Vector{Float32}([]),
    )
    
    if input["algorithm"] == "hmc"
        @assert "TODO"
    elseif input["algorithm"] == "sa"
        temperature_decline(x) = MorphoMol.zig_zag(x, simulation_time_minutes, input["T"], 0.0, sa_level)
        if occursin("cc", input["energy"]) && input["n_mol"] > 2
            bol_nmol_l = (x, id1, id2) -> MorphoMol.are_bounding_spheres_overlapping(x, id1, id2, MorphoMol.get_bounding_radius(mol_type))
            get_initial_connected_components = (x) -> MorphoMol.get_initial_connected_component_energies(x, template_centers, template_radii, rs, prefactors, overlap_jump, overlap_slope, delaunay_eps, bol_nmol_l)
            cc_sa = MorphoMol.Algorithms.ConnectedComponentSimulatedAnnealing(energy, perturbation, temperature_decline, get_initial_connected_components)
            MorphoMol.Algorithms.simulate!(cc_sa, x_init, simulation_time_minutes, output)
        else
            sa = MorphoMol.Algorithms.SimulatedAnnealing(energy, perturbation, temperature_decline)
            MorphoMol.Algorithms.simulate!(sa, x_init, simulation_time_minutes, output)
        end
    elseif input["algorithm"] == "rwm"
        β = 1.0 / input["T"]
        if occursin("cc", input["energy"]) && input["n_mol"] > 2
            bol_nmol_l = (x, id1, id2) -> MorphoMol.are_bounding_spheres_overlapping(x, id1, id2, MorphoMol.get_bounding_radius(mol_type))
            get_initial_connected_components = (x) -> MorphoMol.get_initial_connected_component_energies(x, template_centers, template_radii, rs, prefactors, overlap_jump, overlap_slope, delaunay_eps, bol_nmol_l)
            rwm = MorphoMol.Algorithms.ConnectedComponentRandomWalkMetropolis(energy_call, perturbation_call, initial_connected_component_call, β)
            MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), simulation_time_minutes, output)
        else
            rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, β)
            MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), simulation_time_minutes, output)
        end
    end

    mkpath("$(output_directory)")
    @save "$(output_directory)/$(name).jld2" input output
end