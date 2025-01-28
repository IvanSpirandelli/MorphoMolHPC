using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

using MorphoMol
using JLD2
using Rotations
using StaticArrays

function get_simulation_info(folder::String, selection_folder::String = "")
    id_index = 1
    separator = "_"
    max_id = 1
    for file in readdir(folder)
        if split(file, ".")[end] == "jld2"
            max_id = max_id < parse(Int, split(file, separator)[id_index]) ? parse(Int, split(file, separator)[id_index]) : max_id
        end
    end

    pws = [0.0, 0.0, 0.0]
    acceptance_rates = Vector{Float64}([])
    min_Es = [Inf for _ in 1:max_id]
    thetas = [Inf for _ in 1:max_id]
    evaluation_strings = ["" for _ in 1:max_id]
    total_sampled_states = 0
    for file in readdir(folder)
        if split(file, ".")[end] == "jld2"
            try
                @load "$folder$file" input output
                if "persistence_weights" in keys(input)
                    if pws != input["persistence_weights"]
                        println("Warning: $pws != $(input["persistence_weights"])")
                    end
                    pws = input["persistence_weights"]
                end
                id = parse(Int, split(file, separator)[id_index])
                theta = get_min_theta(input, output)
                string = "$(id): T=$(input["T"]) , σ_r=$(input["σ_r"]), σ_t=$(input["σ_t"])| os = $(input["overlap_slope"])  |as: $(Int(round(length(output["Es"]) / output["αs"][end], digits = 0))) ss: $(length(output["Es"])) rate: $(round(output["αs"][end], digits=2))| E_min: $(round(minimum(output["Es"]), digits = 2)) | Theta: $(round(theta, digits = 2))"
                total_sampled_states += length(output["Es"])
                push!(acceptance_rates, output["αs"][end])
                evaluation_strings[id] = string
                min_Es[id] = minimum(output["Es"])
                thetas[id] = theta

                if theta < 2.0 && selection_folder != ""
                    mkpath(selection_folder)
                    n = length(filter(x -> occursin(".jld2", x), readdir("$(selection_folder)")))
                    @save "$(selection_folder)/$(n+1).jld2" input output
                end
            catch e
                println(e)
                println("Error in $file")
            end
        end
    end

    println("Persistence Weights: $(pws)")
    println("Total sampled states: $(total_sampled_states)")
    println("Average acceptance rate: $(sum(acceptance_rates) / length(acceptance_rates))")
    println("Minimal theta configuration id: $(argmin(thetas)) with theta $(round(thetas[argmin(thetas)], digits=2))")
    println("Minimal energy configuration id: $(argmin(min_Es)) has theta $(round(thetas[argmin(min_Es)], digits=2))")
    for (_,ev_string) in sort!(collect(zip(min_Es, evaluation_strings)))
        println(ev_string)
    end
end

function get_theta(n_mol::Int, mol_type::String, sim_template_centers, state)
    if !(mol_type in keys(MorphoMol.TWOTMVSU_EXPERIMENTAL_ASSEMBLY))
        return Inf
    end
    if n_mol == 2
        exp_template_centers = MorphoMol.TWOTMVSU_EXPERIMENTAL_ASSEMBLY[mol_type]["template_centers"]
        exp_state = MorphoMol.TWOTMVSU_EXPERIMENTAL_ASSEMBLY[mol_type]["state"]
        return MorphoMol.average_offset_distance(exp_template_centers, sim_template_centers, exp_state, state)
    elseif n_mol == 3
        # Consecutive assembly
        R0 = RotMatrix(@SMatrix[1.000000  0.000000  0.000000; 0.000000  1.000000  0.000000; 0.000000  0.000000  1.000000])
        T0 = @SVector[0.00000, 0.00000, 0.00000]

        R1 = RotMatrix(@SMatrix[0.628642  0.777695  0.000000; -0.777695  0.628642  0.000000; 0.000000  0.000000  1.000000])
        T1 = @SVector[-69.28043, 195.91352, -49.35000]

        R2 = RotMatrix(@SMatrix[0.874450  0.485115  0.000000; -0.485115  0.874450  0.000000; 0.000000  0.000000  1.000000])
        T2 = @SVector[-61.30589, 104.11829, -47.94000]

        R3 = RotMatrix(@SMatrix[0.992567  0.121696  0.000000; -0.121696  0.992567  0.000000; 0.000000  0.000000  1.000000])
        T3 = @SVector[-19.48193, 22.01644, -46.53000]

        consecutive = Vector{Float64}([])
        for (R,T) in zip([log(R1), log(R2), log(R3)], [T1, T2, T3])
            consecutive = [consecutive; [R[3,2], R[1,3], R[2,1], T[1], T[2], T[3]]]
        end
        
        # Two top one bottom
        R1 = RotMatrix(@SMatrix[0.628642  0.777695  0.000000; -0.777695  0.628642  0.000000; 0.000000  0.000000 1.000000])
        T1 = @SVector[-69.28043, 195.91352, -49.35000]

        R2 = RotMatrix(@SMatrix[0.874450  0.485115  0.000000; -0.485115  0.874450  0.000000; 0.000000  0.000000 1.000000])
        T2 = @SVector[-61.30589, 104.11829, -47.94000]

        R3 = RotMatrix(@SMatrix[0.52145615 0.85327808 0.000000; -0.85327808 0.52145615 -0.000000; 0.000000 0.000000 1.000000])
        T3 = @SVector[-63.89221, 227.07555, -26.79]

        R4 = RotMatrix(@SMatrix[0.803441  0.595384  0.000000; -0.595384 0.803441 0.000000; 0.000000 0.000000 1.000000])
        T4 = @SVector[-67.99970, 135.02619, -25.38000]

        two_top_one_bottom = Vector{Float64}([])
        for (R,T) in zip([log(R1), log(R2), log(R4)], [T1, T2, T4])
            two_top_one_bottom = [two_top_one_bottom; [R[3,2], R[1,3], R[2,1], T[1], T[2], T[3]]]
        end

        # Two bottom one top
        R1 = RotMatrix(@SMatrix[0.628642  0.777695  0.000000; -0.777695  0.628642  0.000000; 0.000000  0.000000 1.000000])
        T1 = @SVector[-69.28043, 195.91352, -49.35000]

        R2 = RotMatrix(@SMatrix[0.874450  0.485115  0.000000; -0.485115  0.874450  0.000000; 0.000000  0.000000 1.000000])
        T2 = @SVector[-61.30589, 104.11829, -47.94000]

        R3 = RotMatrix(@SMatrix[0.52145615 0.85327808 0.000000; -0.85327808 0.52145615 -0.000000; 0.000000 0.000000 1.000000])
        T3 = @SVector[-63.89221, 227.07555, -26.79]

        R4 = RotMatrix(@SMatrix[0.803441  0.595384  0.000000; -0.595384 0.803441 0.000000; 0.000000 0.000000 1.000000])
        T4 = @SVector[-67.99970, 135.02619, -25.38000]

        two_bottom_one_top = Vector{Float64}([])
        for (R,T) in zip([log(R1), log(R3), log(R4)], [T1, T3, T4])
            two_bottom_one_top = [two_bottom_one_top; [R[3,2], R[1,3], R[2,1], T[1], T[2], T[3]]]
        end
        simulated_assembly_state = state

        exp_template_centers = MorphoMol.TWOTMVSU_EXPERIMENTAL_ASSEMBLY[mol_type]["template_centers"]
        permutations = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3,1,2], [3,2,1]]
        min_theta = Inf
        for experimental_state in [consecutive, two_top_one_bottom, two_bottom_one_top]
            cand = minimum([MorphoMol.sum_of_permutation(sim_template_centers, exp_template_centers, simulated_assembly_state, experimental_state, [1, 2, 3], perm) for perm in permutations])
            if cand < min_theta
                min_theta = cand
            end
        end
        return min_theta
    end
    Inf
end

function get_min_theta(input::Dict{String, Any}, output::Dict{String, Vector})
    get_theta(input["n_mol"], input["mol_type"], input["template_centers"], output["states"][argmin(output["Es"])])
end