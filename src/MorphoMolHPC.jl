module MorphoMolHPC

include("template_data.jl")

export TEMPLATES
export get_initial_state

using Distributions

function get_initial_state(n_mol::Int, bounds::Float64)
    vcat([
        [rand(Uniform(0.0, 2*pi)), rand(Uniform(0.0, 2*pi)), rand(Uniform(0.0, 2*pi)), 
        rand(Uniform(0.0, bounds)), rand(Uniform(0.0, bounds)), rand(Uniform(0.0, bounds))] 
        for i in 1:n_mol]...);
end

end #module MorphoMolHelpers