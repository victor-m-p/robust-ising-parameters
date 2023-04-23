'''
Find extrema points; 
i.e. local minima and local maxima. 
'''

import pandas as pd 
import numpy as np 

configs = np.loadtxt('../data/preprocessing/conf_h0.txt', dtype=int)
p = np.loadtxt('../data/preprocessing/prob_h0.txt', dtype=float)

def extrema(configs, p):
    n_observations, n_nodes = configs.shape
    maxima = []
    minima = []
    for input_idx in range(n_observations): 
        neighbors = np.zeros(n_nodes, dtype=int)
        for i in range(n_nodes):
            bit_mask = 1 << i
            x = input_idx ^ bit_mask
            print(x)
            neighbors[i] = x
        p_neighbors = p[neighbors]
        p_self = p[input_idx]
        min_neighbors = np.min(p_neighbors)
        max_neighbors = np.max(p_neighbors)
        if p_self < min_neighbors:
            minima.append(input_idx)
        elif p_self > max_neighbors:
            maxima.append(input_idx)
    return minima, maxima

# just save here 
minima, maxima = extrema(configs, p)

# find basins of attraction
d = pd.DataFrame({
    'from': [1, 2, 3],
    'to': [2, 2, 1]})

def find_end_and_steps(start, df):
    current = start
    steps = -1
    while True:
        next_val = df.loc[df['from'] == current, 'to'].values[0]
        steps += 1
        if next_val == current:
            break
        current = next_val
    return current, steps

end = []
steps = []

for index, row in d.iterrows():
    e, s = find_end_and_steps(row['from'], d)
    end.append(e)
    steps.append(s)

d['end'] = end
d['steps'] = steps

print(d)
