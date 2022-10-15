import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import rc
import csv
import os


#%% get files and extract data
cur_path = os.path.dirname(os.path.realpath(__file__))
path = cur_path + "\\log_scales_medium"
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

data = []

for fname in files:
    G = np.genfromtxt(path + '\\' +  fname, delimiter=',', skip_header=9)
    data.append(G[:,1:])

num_trials = data[0].shape[0]//2
dists = data[0][0,:]
#%%
plt.close('all')
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

scale_factors = [.5, .6, .7, .8, .9, 1.0]
colors=['peru', 'coral',  'olive', 'sienna', 'steelblue', 'tan','deeppink']

fig, ax = plt.subplots(1,2, figsize=(15, 5))
for i in range(len(data)):
    graph_dists = data[i][1:num_trials+1, :]
    graph_dists_2 = data[i][num_trials+1:, :]
    ratios = graph_dists/dists
    ratios_2 = graph_dists_2/dists
    
    exp_ratios = np.mean(ratios, axis=0)
    exp_ratios_2 = np.mean(ratios_2, axis=0)
    std_ratios_2 = np.std(ratios_2, axis=0)
    
    # plotting
    color = colors[i]
    ax[0].plot(dists, exp_ratios_2, marker ='.', label=str(scale_factors[i]), color=color)
   # ax[0].plot(dists, exp_ratios_2, marker ='.', label=str(scale_factors[i]), color=color)
    ax[0].fill_between(dists, exp_ratios_2 - std_ratios_2,\
                           exp_ratios_2 + std_ratios_2,\
                           alpha=0.3, edgecolor=None, color=color)
        
    ax[1].plot(dists, exp_ratios_2, marker ='.', label=str(scale_factors[i]), color=color)
    ax[1].fill_between(dists, (exp_ratios_2 - std_ratios_2),\
                           (exp_ratios_2 + std_ratios_2),\
                           alpha=0.3, edgecolor=None, color=color)
    

ax[0].set_xlabel('$s$')
ax[0].set_ylabel('$T_s/s$')
ax[1].set_yscale('log')
ax[1].set_xscale('log')   
legend = ax[0].legend()
legend.set_title('Factors')
fig.savefig("vis/exp_converegnce.pdf",bbox_inches="tight")

#%% ratio convergence
m = data[0].shape[1]
fig2, ax2 = plt.subplots(1,2, figsize=(15, 5))

idx_1 = []
idx_2 = []
for i in range(m):
    for j in range(i+1,m):
        if np.isclose(dists[j]/dists[i], 2.0):
            idx_1.append(i)
            idx_2.append(j)
            break
    
for i in range(len(data)):
    graph_dists_1 = data[i][1:num_trials+1, :]
    exp_dists_1 = np.mean(graph_dists_1, axis=0)
    
    graph_dists_2 = data[i][num_trials+1:, :]
    exp_dists_2 = np.mean(graph_dists_2, axis=0)
    
    trial_ratios = np.mean(graph_dists_1/graph_dists_2,axis=0)
    
    idx, = np.where((exp_dists_1 + exp_dists_2) < np.inf)
    #idx = idx[1:]
    
    if len(idx)>0:
        ratios = exp_dists_1[idx]/exp_dists_2[idx]
        errors = np.abs(ratios - 0.5)
        color = colors[i]
        #ax2[0].plot(dists[idx], ratios, marker ='.', label=str(scale_factors[i]),color=color)
        ax2[0].plot(dists[idx], trial_ratios[idx], marker ='.', label=str(scale_factors[i]),color=color)
       
        #ax2[1].plot(dists[idx], np.abs(trial_ratios[idx]-0.5), marker ='.', label=str(scale_factors[i]),  color=color)
        ax2[1].plot(dists[idx], np.abs(ratios-0.5), marker ='.', label=str(scale_factors[i]),  color=color)

ax2[1].set_yscale('log')
ax2[1].set_xscale('log')
ax2[1].axis('equal')
legend = ax2[0].legend()
legend.set_title('Factors')

fig2.savefig("vis/ratio_converegnce.pdf",bbox_inches="tight")