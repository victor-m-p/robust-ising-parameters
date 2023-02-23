# 2023-02-08: refactored and re-run.
include("configuration.jl")
using .cn, Printf, Statistics, Distributions, DelimitedFiles, CSV, DataFrames, IterTools, StatsBase, Chain, FStrings, Base.Threads

# load shit 
# manage paths 
dir = @__DIR__
path_configuration_probabilities = replace(dir, "analysis" => "data/preprocessing/configuration_probabilities.txt")
path_configurations = replace(dir, "analysis" => "data/preprocessing/configurations.txt")
path_entry_config = replace(dir, "analysis" => "data/preprocessing/entry_maxlikelihood.csv")

configuration_probabilities = readdlm(path_configuration_probabilities)
configurations = readdlm(path_configurations, Int)
configurations = cn.slicematrix(configurations)

# load all maximum likelihood configurations 
#entry_maxlikelihood = DataFrame(CSV.File(path_entry_config))
#config_ids = @chain entry_maxlikelihood begin _.config_id end
#unique_configs = unique(config_ids) # think right, but double check 
#unique_configs = unique_configs .+ 1 # because of 0-indexing in python 
stability_list = Vector{Float64}()
for configuration in 1:length(configurations)  
    ConfObj = cn.Configuration(configuration, configurations, configuration_probabilities)
    p_move = ConfObj.p_move()
    push!(stability_list, p_move)
end 
stability_list = 1.0 .- stability_list
outname = replace(dir, "analysis" => "data/analysis/stability.txt")
writedlm(outname, stability_list)
stability_list

#println("saving file")
#d = DataFrame(
#config_id = [x-1 for x in unique_configs],
#config_prob = [x for (x, y) in sample_list],
#remain_prob = [y for (x, y) in sample_list] 
#)
#CSV.write(replace(dir, "analysis" => "data/analysis/maxlik_evo_stability.csv"), d)