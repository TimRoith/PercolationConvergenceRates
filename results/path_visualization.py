import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import rc

#%% custom imports
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

import utils
 
 
#%% parameters
params = {
's_min' : 100,
's_max' : 10000,# maximal domain size in the first component
'num_s' : 4,# number of points for s
'd' : 2, # spatial dimension
'lamda' : 1,# intensity of the point process
'num_cores' : 40,
'num_trials': 100,
'factor':1}

s_disc = [2**i * params['s_min'] for i in range(params['num_s']) if (2**i * params['s_min'])<= params['s_max']]
seed = 420
np.random.seed(seed)
d = params['d']
scale = utils.log_scale(d=params['d'], factor=params['factor'])

s = s_disc[-1]
bounds = np.zeros((d, 2))
bounds[0,0] = -s**(1/d)
bounds[0,1] = s + s**(1/d)
bounds[1:,0] = -s**(1/d)
bounds[1:,1] = s**(1/d)
delta = bounds[:,1] - bounds[:,0]
area = np.prod(delta)

PP = utils.Poisson_process(bounds, lamda = params['lamda'], area = area, d=params['d'])
points = np.vstack([np.zeros((2, d)), PP()]) 


#%% set up plots
plt.close('all')
proj = '3d' if d==3 else None
rc('font',**{'family':'serif','serif':['Times'],'size':14})
rc('text', usetex=True)
fig, ax = plt.subplots(1,1,squeeze=False, subplot_kw={'projection': proj})

ph = utils.plot_handler(ax, points)
plt.show()
plt.pause(0.1)
        
#%% main loop
for i, s in enumerate(s_disc):
    # update target point
    points[1,0] = s
    
    h = scale(s)
    # get weight matrix
    kd_tree = scipy.spatial.KDTree(points)
    W = kd_tree.sparse_distance_matrix(kd_tree, h)
    
    if np.sum(W[0,:]) == 0:
        g_dist = np.inf
    else:
        # dist_matrix = scipy.sparse.csgraph.dijkstra(W, directed=False, indices=[0])
        dist_matrix, predecessors = scipy.sparse.csgraph.shortest_path(W, directed=False, indices=[0],return_predecessors=True)
        g_dist = dist_matrix[0,1]
        path = utils.compute_optimal_path(predecessors, 1)
        
        # update plot
        ph.update(path)
        plt.draw()
        plt.pause(0.5)
        
    name = './path_vis/path_' + str(d) + 'd_s-'+str(s) + '.png'
    #plt.tight_layout()
    plt.savefig(name, dpi=300)

    print('Distance {}, ratio {}'.format(s,g_dist/s))