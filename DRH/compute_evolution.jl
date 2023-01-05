include("configuration.jl")
using .cn, Printf, Statistics, Distributions, DelimitedFiles, CSV, DataFrames, IterTools, StatsBase, Chain, FStrings, Base.Threads

# load shit 
configuration_probabilities = readdlm("/home/vpoulsen/humanities-glass/data/analysis/configuration_probabilities.txt")
configurations = readdlm("/home/vpoulsen/humanities-glass/data/analysis/configurations.txt", Int)
## I need an array, rather than a matrix 
configurations = cn.slicematrix(configurations)

# load all maximum likelihood configurations 
entry_config_filename = "/home/vpoulsen/humanities-glass/data/analysis/entry_maxlikelihood.csv"
entry_maxlikelihood = DataFrame(CSV.File(entry_config_filename))
config_ids = @chain entry_maxlikelihood begin _.config_id end
unique_configs = unique(config_ids) # think right, but double check 
unique_configs = unique_configs .+ 1 # because of 0-indexing in python 

# command line arguments 
#sim_n, start = ARGS
#start = parse(Int64, start)
#sim_n = parse(Int64, sim_n)
#unique_configs = unique_configs[start:start+99] # 10 total 

# setup 
n_simulation = 100
n_timestep = 100
global sample_list = [] 
global conf_list = []
total_configs = length(unique_configs)
@time begin 
global num = 0
for unique_config in unique_configs
    global num += 1
    println("$num / $total_configs")
    for sim_number in 1:n_simulation
        x = findfirst(isequal(unique_config), [x for (x, y) in conf_list]) # is this what we want?
        if x isa Number 
            ConfObj = conf_list[x][2] # return the corresponding class 
        else 
            ConfObj = cn.Configuration(unique_config, configurations, configuration_probabilities)
        end 
        id = ConfObj.id 
        for time_step in 1:n_timestep
            push!(sample_list, [sim_number, time_step, id])
            if id ∉ [x for (x, y) in conf_list]
                push!(conf_list, [id, ConfObj]) 
            end 
            ConfObj = ConfObj.push_forward(configurations, configuration_probabilities, true, false, conf_list)
            id = ConfObj.id 
        end 
    end 
    if num % 20 == 0
        println("saving file")
        d = DataFrame(
        simulation = [x for (x, y, z) in sample_list],
        timestep = [y for (x, y, z) in sample_list],
        config_id = [z for (x, y, z) in sample_list]
        )
        CSV.write(f"/home/vpoulsen/humanities-glass/data/COGSCI23/evo/num_{num}_s_{n_simulation}_t_{n_timestep}.csv", d)
        global sample_list = []
    end 
end 
end 