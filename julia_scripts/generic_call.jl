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
    radii = vcat([template_radii for i in 1:n_mol]...);

    x_init = x
    if length(x_init) == 0
        x_init = MorphoMol.get_initial_state(n_mol, bnds)
    end

    prefactors = MorphoMol.Energies.get_prefactors(rs, η)

    input = Dict(
        "energy" => nrg,
        "perturbation" => prtbt,
        "mol_type" => mol_type,
        "template_centers" => template_centers,
        "template_radii" => template_radii,
        "n_mol" => n_mol,
        "x_init" => x_init,
        "comment" => comment,
        "bounds" => bnds,
        "rs" => rs,
        "η" => η,
        "prefactors" => prefactors,
        "σ_r" => σ_r,
        "σ_t" => σ_t,
        "T" => T,
        "persistence_weights" => persistence_weights,
        "overlap_jump" => overlap_jump,
        "overlap_slope" => overlap_slope,
        "delaunay_eps" => delaunay_eps,
        "simulation_time_minutes" => simulation_time_minutes,
    )

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

    energy = get_energy(input)
    perturbation = get_perturbation(input)

    if isapprox(input["T"],0.0)
        set_temperature!(input; scaling = 0.003)
    end

    println("T = $(input["T"])")
    β = 1.0/input["T"]

    rwm = MorphoMol.Algorithms.RandomWalkMetropolis(energy, perturbation, β)

    MorphoMol.Algorithms.simulate!(rwm, deepcopy(x_init), simulation_time_minutes, output);

    mkpath("$(output_directory)")
    @save "$(output_directory)/$(name).jld2" input output
end

function get_energy(input)
    if input["energy"] == "tasp"
        return (x) -> MorphoMol.total_alpha_shape_persistence(x, input["template_centers"], input["persistence_weights"])
    elseif input["energy"] == "dbbasp"
        return (x) -> MorphoMol.death_by_birth_alpha_shape_persistence(x, input["template_centers"], input["persistence_weights"])
    elseif input["energy"] == "tip"
        return (x) -> MorphoMol.total_interface_persistence(x, input["template_centers"], input["persistence_weights"])
    elseif input["energy"] == "dbbip"
        return (x) -> MorphoMol.death_by_birth_interface_persistence(x, input["template_centers"], input["persistence_weights"])
    else 
        return (x) -> 0.0
    end
end

function get_perturbation(input)
    σ_r = input["σ_r"]
    σ_t = input["σ_t"]
    n_mol = input["n_mol"]
    if input["perturbation"] == "single_random"
        return (x) -> MorphoMol.perturb_single_randomly_chosen(x, σ_r, σ_t)
    elseif input["perturbation"] == "all"
        Σ = vcat([[σ_r, σ_r, σ_r, σ_t, σ_t, σ_t] for _ in 1:n_mol]...)
        return (x) -> MorphoMol.perturb_all(x, Σ)
    end
end

function set_temperature!(input; n_samples=1000, scaling = 0.1)
    energy = get_energy(input)
    n_mol = input["n_mol"]
    bounds = input["bounds"]
    test_Es = [energy(MorphoMol.get_initial_state(n_mol, bounds))[1] for i in 1:n_samples]
    test_Es = [e - minimum(test_Es) for e in test_Es]
    input["T"] = sum(test_Es) / length(test_Es) * scaling
end
