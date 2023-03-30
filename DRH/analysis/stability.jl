# 2023-02-08: refactored and re-run.
include("configuration.jl")
using .cn, Printf, Statistics, Distributions, DelimitedFiles, CSV, DataFrames, IterTools, StatsBase, Chain, FStrings, Base.Threads

nt = Threads.nthreads()
println(nt)
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

size = length(configurations)
stability_list = Vector{Float64}(undef, size)
idx_list = Vector{Int32}(undef, size)

Threads.@threads for configuration in 1:size #length(configurations)  
    # the actual computation
    ConfObj = cn.Configuration(configuration, configurations, configuration_probabilities)
    stability_list[configuration] = 1-ConfObj.p_move()
    idx_list[configuration] = configuration
end 
out_stability = replace(dir, "analysis" => f"data/analysis/stability.txt")
out_idx = replace(dir, "analysis" => f"data/analysis/idx_stability.txt")
writedlm(out_stability, stability_list)
writedlm(out_idx, idx_list)
#println("saving file")
#d = DataFrame(
#config_id = [x-1 for x in unique_configs],
#config_prob = [x for (x, y) in sample_list],
#remain_prob = [y for (x, y) in sample_list] 
#)
#CSV.write(replace(dir, "analysis" => "data/analysis/maxlik_evo_stability.csv"), d)