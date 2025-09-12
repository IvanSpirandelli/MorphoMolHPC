# Standard library and package imports
using Pkg

# Activate the project environment to ensure all dependencies are met
Pkg.activate("Project.toml")
Pkg.instantiate()

# Import necessary libraries for the simulation
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
using MorphoMol
using JLD2
using Rotations
using JSON # Use the JSON library to parse configuration files

"""
    run_simulation_from_config(config::Dict)

Takes a dictionary containing all simulation parameters, sets up,
and runs a single simulation, then saves the output.
"""
function run_simulation_from_config(config::Dict)
    # --- 1. Extract parameters and set up the simulation 'input' dictionary ---
    
    # Directly access parameters from the 'config' dictionary
    mol_type = config["mol_type"]
    n_mol = config["n_mol"]
    rs = config["rs"]
    η = config["η"]
    x_init = config["x_init"]
    x_init = length(x_init) == 0 ? Vector{Tuple{QuatRotation{Float64}, Vector{Float64}}}([]) : [(QuatRotation(RotMatrix{3}(reduce(hcat, c[1]))), Vector{Float64}(c[2])) for c in x_init]
    
    # Get molecular templates
    template_centers = MorphoMol.TEMPLATES[mol_type]["template_centers"]
    template_radii = MorphoMol.TEMPLATES[mol_type]["template_radii"]


    n_atoms_per_mol = length(template_centers) ÷ 3
    template_centers = reshape(template_centers, (3, n_atoms_per_mol))
    radii = vcat([template_radii for _ in 1:n_mol]...)

    prefactors = MorphoMol.Energies.get_prefactors(rs, η)
    persistence_weights = Vector{Float64}(config["persistence_weights"])

    overlap_jump = config["overlap_jump"]
    overlap_slope = config["overlap_slope"]
    delaunay_eps = config["delaunay_eps"]

    # Build the main 'input' dictionary for MorphoMol functions
    input = Dict(
        "algorithm" => config["algorithm"],
        "sa_level" => config["sa_level"],
        "energy" => config["energy"],
        "perturbation" => config["perturbation"],
        "initialization" => config["initialization"],
        "mol_type" => mol_type,
        "template_centers" => template_centers,
        "template_radii" => template_radii,
        "n_mol" => n_mol,
        "x_init" => x_init,
        "comment" => config["comment"],
        "bounds" => config["bounds"],
        "rs" => rs,
        "η" => η,
        "mu" => config["mu"],
        "prefactors" => prefactors,
        "σ_r" => config["σ_r"],
        "σ_t" => config["σ_t"],
        "T_search_runs" => config["T_search_runs"],
        "T_search_time" => config["T_search_time"],
        "T" => config["temperature"],
        "persistence_weights" => persistence_weights,
        "overlap_jump" => overlap_jump,
        "overlap_slope" => overlap_slope,
        "delaunay_eps" => delaunay_eps,
        "exact_delaunay" => config["exact_delaunay"],
        "mode" => config["mode"],
        "simulation_time_minutes" => config["simulation_time_minutes"],
        "iterations" => config["iterations"],
    )
    println("Input constructed!")
    # --- 2. Prepare for the simulation run (same logic as before) ---

    # Get initialization function and generate initial state if not provided
    initialization_func = MorphoMol.get_initialization(input, true)
    if length(x_init) == 0
        x_init = initialization_func()
        input["x_init"] = x_init
    end
    
    energy = MorphoMol.get_energy(input)
    perturbation = MorphoMol.get_perturbation(input)

    # Automatically find initial temperature if T is set to 0
    if isapprox(input["T"], 0.0)
        T = MorphoMol.get_initial_temperature(input; n_samples = 100, scaling = 0.0025)
        input["T"] = T
    end

    # --- Temperature search logic ---
    search_Ts = [input["T"]]
    search_αs = Vector{Float64}([])
    α_targets = Dict("sa" => 0.8, "rwm" => 0.2, "hmc" => 0.65)
    α_target = α_targets[input["algorithm"]]

    for i in 1:input["T_search_runs"]
        T = search_Ts[end]
        output_search = Dict{String, Vector}(
            "states" => Vector{Vector{Tuple{QuatRotation{Float64}, Vector{Float64}}}}([]),
            "Es" => Vector{Float64}([]), 
            "Vs" => Vector{Float64}([]), 
            "As" => Vector{Float64}([]), 
            "Cs" => Vector{Float64}([]), 
            "Xs" => Vector{Float64}([]),
            "OLs" => Vector{Float64}([]),
            "P0s" => Vector{Float64}([]),
            "P1s" => Vector{Float64}([]),
            "P2s" => Vector{Float64}([]),
            "timestamps" => Vector{Float16}([]),
            "αs" => Vector{Int}([]),
            "total_step_attempts" => Vector{Int}([]),
        )
        if occursin("cc", input["energy"]) && input["n_mol"] > 2
            bol_nmol_l = (x, id1, id2) -> MorphoMol.are_bounding_spheres_overlapping(x, id1, id2, MorphoMol.get_bounding_radius(template_centers, template_radii, rs))
            get_initial_connected_components = (x) -> MorphoMol.get_initial_connected_component_energies(x, template_centers, template_radii, rs, prefactors, overlap_jump, overlap_slope, delaunay_eps, bol_nmol_l)
            cc_rwm = MorphoMol.Algorithms.ConnectedComponentRandomWalkMetropolis(energy, perturbation, get_initial_connected_components, 1/T)
            x_search = initialization_func()
            MorphoMol.Algorithms.simulate!(cc_rwm, x_search, T_search_time, output_search)
        else
            rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, 1/T)
            x_search = initialization_func()
            MorphoMol.Algorithms.simulate!(rwm, x_search, T_search_time, output_search)
        end
        α = length(output_search["αs"])/output_search["αs"][end]
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

    println("T search completed. Final temperature: $(input["T"])")

    # --- 3. Run the main simulation ---
    output = Dict{String, Vector}(
        "states" => Vector{Vector{Tuple{QuatRotation{Float64}, Vector{Float64}}}}(),
        "Es" => Vector{Float64}([]), 
        "Vs" => Vector{Float32}([]), 
        "As" => Vector{Float32}([]), 
        "Cs" => Vector{Float32}([]), 
        "Xs" => Vector{Float32}([]),
        "OLs" => Vector{Float32}([]),
        "P0s" => Vector{Float32}([]),
        "P1s" => Vector{Float32}([]),
        "P2s" => Vector{Float32}([]),
        "timestamps" => Vector{Float16}([]),
        "αs" => Vector{Int}([]),
        "total_step_attempts" => Vector{Int}([]),
    )
    
    mode = input["mode"]
    iterations = input["iterations"]
    simulation_time_minutes = input["simulation_time_minutes"]

    if input["algorithm"] == "hmc"
        @assert "TODO"
    elseif input["algorithm"] == "sa"
        temperature_decline(x) = MorphoMol.zig_zag(x, simulation_time_minutes, input["T"], 0.0, sa_level)
        if occursin("cc", input["energy"]) && input["n_mol"] > 2
            bol_nmol_l = (x, id1, id2) -> MorphoMol.are_bounding_spheres_overlapping(x, id1, id2, MorphoMol.get_bounding_radius(template_centers, template_radii, rs))
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
            bol_nmol_l = (x, id1, id2) -> MorphoMol.are_bounding_spheres_overlapping(x, id1, id2, MorphoMol.get_bounding_radius(template_centers, template_radii, rs))
            get_initial_connected_components = (x) -> MorphoMol.get_initial_connected_component_energies(x, template_centers, template_radii, rs, prefactors, overlap_jump, overlap_slope, delaunay_eps, bol_nmol_l)
            rwm = MorphoMol.Algorithms.ConnectedComponentRandomWalkMetropolis(energy, perturbation, get_initial_connected_components, β)
            if mode == "time"
                MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), simulation_time_minutes, output)
            elseif mode == "iterations"
                MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), iterations, output)
            end
        else
            rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, β)
            if mode == "time"
                println("Running Random Walk Metropolis for $n_mol molecules and $(simulation_time_minutes) minutes.")
                MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), simulation_time_minutes, output)
            elseif mode == "iterations"
                println("Running Random Walk Metropolis for $n_mol molecules and $(iterations) iterations.")
                MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), iterations, output)
            end
        end
    end

    # --- 4. Save the results ---
    output_directory = config["output_directory"]
    name = config["name"]
    mkpath(output_directory) # Ensure the directory exists
    @save "$(output_directory)/$(name).jld2" input output

    println("Simulation $(name) completed successfully.")
end


"""
    main()

Entry point of the script. Parses command-line arguments, loads the
correct configuration from the JSON file, and starts the simulation.
"""
function main()
    # ARGS will contain the command-line arguments.
    # We expect: julia script.jl <json_path> <task_id>
    if length(ARGS) != 2
        error("Usage: julia generic_call_from_json.jl <config.json_path> <task_id>")
    end

    json_path = ARGS[1]
    # SLURM_ARRAY_TASK_ID is 1-based, which matches Julia's array indexing.
    task_id = parse(Int, ARGS[2])

    # Read the JSON file which contains an array of configuration dictionaries
    all_configs = JSON.parsefile(json_path)

    if task_id > length(all_configs) || task_id < 1
        error("Task ID $(task_id) is out of bounds for the number of configurations ($(length(all_configs))).")
    end

    # Select the configuration for this specific array task
    my_config = all_configs[task_id]

    println("--- Starting Simulation ---")
    println("Task ID: $(task_id)")
    println("Configuration Path: $(json_path)")
    
    # Run the simulation with the selected configuration
    run_simulation_from_config(my_config)
end

# Execute the main function when the script is run
main()