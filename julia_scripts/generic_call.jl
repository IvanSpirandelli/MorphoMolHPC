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
        "simulation_time_minutes" => simulation_time_minutes,
    )

    initialization = MorphoMol.get_initialization(input)
    x_init = x
    if length(x_init) == 0
        x_init = initialization()
        input["x_init"] = x_init
    end

    energy = MorphoMol.get_energy(input)
    perturbation = MorphoMol.get_perturbation(input)

    if isapprox(input["T"],0.0)
        T = MorphoMol.get_initial_temperature(input; scaling = 0.0025)
        input["T"] = T
    end

    search_Ts = [input["T"]]
    search_αs = Vector{Float64}([])

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
        rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, 1/T)
        x_search = initialization()
        MorphoMol.Algorithms.simulate!(rwm, x_search, T_search_time, output_search);
        α = output_search["αs"][end]
        push!(search_αs, α)

        if α > 0.24
            push!(search_Ts, T * 0.5)
        else
            push!(search_Ts, T * 1.5)
        end
    end

    if length(search_αs) > 1
        println("Ts = $(search_Ts)")
        println("αs = $(search_αs)")
        input["T"] = search_Ts[argmin([abs(α - 0.24) for α in search_αs])]
        input["T_search_αs"] = search_αs
        input["T_search_Ts"] = search_Ts
    end
    β = 1.0 / input["T"]

    println("T = $(input["T"])")

    rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, β)
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


    MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), simulation_time_minutes, output);

    mkpath("$(output_directory)")
    @save "$(output_directory)/$(name).jld2" input output
end

# function search_for_best_temperature()
# end