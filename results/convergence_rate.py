import numpy as np
import matplotlib.pyplot as plt
import csv
import os


#%% get files
cur_path = os.path.dirname(os.path.realpath(__file__))
path = cur_path + "\\log_scales"
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

data = []

for fname in files:
    G = np.genfromtxt(path + '\\' +  fname, delimiter=',', skip_header=8)
    data.append(G[:,1:])
    
#%%
dists = data[0][0,:]
scale_factors = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
colors=['r', 'g', 'b', 'y', 'peru', 'tan', 'coral']

fig, ax = plt.subplots(1,1)

for i in range(len(data)):
    graph_dists = data[i][1:]
    ratios = graph_dists/dists
    exp_ratios = np.mean(ratios, axis=0)
    std_ratios = np.std(ratios, axis=0)
    
    # plotting
    color = colors[i]
    ax.plot(dists, exp_ratios, marker ='.', label=str(scale_factors[i]), color=color)
    ax.fill_between(dists, exp_ratios - std_ratios,\
                           exp_ratios + std_ratios,\
                           alpha=0.3, edgecolor=None, color=color)
    
ax.legend()

#%% ratio convergence
m = data[0].shape[1]
fig2, ax2 = plt.subplots(1,2)

idx_1 = []
idx_2 = []
for i in range(m):
    for j in range(i+1,m):
        if np.isclose(dists[j]/dists[i], 2.0):
            idx_1.append(i)
            idx_2.append(j)
            break
idx_1 = np.array(idx_1)
idx_2 = np.array(idx_2)
    
for i in range(len(data)):
    graph_dists_1 = data[i][:, idx_1]
    exp_dists_1 = np.mean(graph_dists_1, axis=0)
    
    graph_dists_2 = data[i][:, idx_2]
    exp_dists_2 = np.mean(graph_dists_2, axis=0)
    
    idx, = np.where((exp_dists_1 + exp_dists_2) < np.inf)
    
    if len(idx)>0:
        ratios = exp_dists_1[idx]/exp_dists_2[idx]
        color = colors[i]
        ax2[0].plot(dists[idx_1[idx]], ratios, marker ='.', label=str(scale_factors[i]), color=color)
        ax2[1].loglog(dists[idx_1[idx]], np.abs(ratios-0.5), marker ='.', label=str(scale_factors[i]),  color=color)

ax2[0].legend()