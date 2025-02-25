using Pkg
Pkg.activate("Project.toml")
Pkg.instantiate()

using JLD2


function distill(source_folder, target_folder)
    mkpath(target_folder)
    for file in readdir(source_folder)
        if split(file, ".")[end] == "jld2"
            try
                @load "$source_folder$file" input output
                E_mindex = argmin(output["Es"])
                for k in keys(output)
                    if length(output[k]) > 1
                        output[k] = [output[k][E_mindex]]
                    end
                end
                @save "$target_folder$file" input output
            catch e
                println(e)
            end
        end
    end
end