import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import graphlearning as gl
from matplotlib.pyplot import cm
import time
import pickle
import os
#%% custom imports
from utils import compute_optimal_path, scale, plot_handler, Poisson_process

#%% parameters
s_max = 500
max_dists = 20
dists = np.linspace(2.1, s_max, max_dists)
d = 2

bounds = np.zeros((d, 2))
bounds[0,0] = -s_max/4
bounds[0,1] = 1.25 * s_max

bounds[1:,0] = -s_max**(1/d)
bounds[1:,1] = s_max**(1/d)

delta = bounds[:,1] - bounds[:,0]
area = np.prod(delta)
lamda = 1 #intensity (ie mean density) of the Poisson process
random_seed = 42

#%% 
trials = 20
ratios = np.zeros((trials, len(dists)))
PP = Poisson_process(bounds, lamda = lamda, area = area, d=d)


#%% main loop
for T in range(trials):
    print(10*'<>')
    print('Trial {}'.format(T+1))
    print(10*'<>')
    
    np.random.seed(T)
    points = np.vstack([np.zeros((2, d)), PP()])
    
    for i,s in enumerate(dists):
        # update target point
        points[1,0] = s
        
        # get points in frame
        idx, = np.where(np.prod(np.abs(points) < 1.25*s, axis=1))
        loc_points = points[idx,:]
        
        #h = scale(s, d)
        h = .7 * np.log(s)**(1/d)
        #h = 1.3
        # get weight matrix
        W = gl.weightmatrix.epsilon_ball(loc_points, h, kernel='distance')
        G = gl.graph(W)
        
        if np.sum(W[0,:]) == 0:
            g_dist = np.inf
        else:
            # dist_matrix = scipy.sparse.csgraph.dijkstra(W, directed=False, indices=[0])
            dist_matrix, predecessors = scipy.sparse.csgraph.shortest_path(W, directed=False, indices=[0],return_predecessors=True)
            g_dist = dist_matrix[0,1] 
        
        print('Distance {}, ratio {}'.format(s,g_dist/s))
        ratios[T,i] = g_dist/s
        
#%% handle ratios
exp_ratios = np.mean(ratios, axis=0)
std_ratios = np.std(ratios, axis=0)
exp_errors = np.abs(exp_ratios - exp_ratios[-1])
idx = [i for i in range(len(dists)) if exp_ratios[i]<np.inf]

p = np.polyfit(np.log(dists[idx][:-1]),np.log(exp_errors[idx][:-1]),1)
rate = p[0]

#%% plotting
plt.close('all')
fig, ax = plt.subplots(1,1, squeeze=False)

ax[0,0].loglog(dists[idx][:-1],exp_errors[idx][:-1])
ax[0,0].loglog(dists[idx][:-1],np.exp(p[1])*dists[idx][:-1]**rate)



print(10*'<>')
print('Rate of expected value: {}'.format(rate))
print(10*'<>')

#%%
fig_2, ax_2 = plt.subplots(1,2, squeeze=False)
ax_2[0,0].plot(dists, exp_ratios, marker ='.', color='g')
ax_2[0,0].fill_between(dists, exp_ratios - std_ratios,\
                       exp_ratios + std_ratios, color='g',\
                       alpha=0.3, edgecolor=None)

idx = [i for i in range(len(dists)) if exp_ratios[i]<np.inf]
p = np.polyfit(np.log(dists[idx]),np.log(exp_ratios[idx]),1)

ax_2[0,1].plot(dists, exp_ratios, marker ='.', color='g')
ax_2[0,1].fill_between(dists, exp_ratios - std_ratios,\
                        exp_ratios + std_ratios, color='g',\
                        alpha=0.3, edgecolor=None)
#ax_2[0,1].plot(dists[idx],np.exp(p[1])*dists[idx]**p[0], color='r', marker='.')
ax_2[0,1].set_yscale('log')
ax_2[0,1].set_xscale('log')


#%% save ratios to file
time_str = time.strftime("%Y%m%d-%H%M%S")
fname = "results/ratios-" + time_str + '.pkl'

with open(fname, 'wb') as f:
    # Put them in the file 
    pickle.dump([dists, ratios], f)

#%%
load = False
if load:
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    with open(file_dir+'\\results\\ratios-20221005-112412.pkl','rb') as file:
        dists, ratios = pickle.load(file)