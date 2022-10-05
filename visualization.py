import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import graphlearning as gl
from matplotlib.pyplot import cm

#%% custom imports
from utils import compute_optimal_path, scale, plot_handler, Poisson_process
 
 
#%% parameters
max_dists = 20
d = 3
s_max = 300

bounds = np.zeros((d, 2))

bounds[0,0] = -s_max/4
bounds[0,1] = 1.25 * s_max
#x1_delta = x1_max - x1_min

bounds[1:,0] = -s_max**(1/d)
bounds[1:,1] = s_max**(1/d)

delta = bounds[:,1] - bounds[:,0]
area = np.prod(delta)
lamda = 1 #intensity (ie mean density) of the Poisson process
random_seed = 42

#%% array for distances, points and scaling
dists = np.linspace(2.1, s_max, max_dists)
                
#%% construct Poisson cloud
ratios = []
np.random.seed(random_seed)
PP = Poisson_process(bounds, lamda = lamda, area = area, d=d)
points = np.vstack([np.zeros((2, d)), PP()])


#%% set up plots
plt.close('all')
proj = '3d' if d==3 else None
fig, ax = plt.subplots(1,1,squeeze=False, subplot_kw={'projection': proj})

ph = plot_handler(ax, points)
plt.show()
plt.pause(0.1)
        
#%% main loop
for i, s in enumerate(dists):
    # update target point
    points[1,0] = s
    
    # get points in frame
    idx, = np.where(np.prod(np.abs(points) < 1.25*s, axis=1))
    loc_points = points[idx,:]
    
    h = scale(s, d)
    # get weight matrix
    W = gl.weightmatrix.epsilon_ball(loc_points, h, kernel='distance')
    G = gl.graph(W)
    
    if np.sum(W[0,:]) == 0:
        g_dist = np.inf
    else:
        # dist_matrix = scipy.sparse.csgraph.dijkstra(W, directed=False, indices=[0])
        dist_matrix, predecessors = scipy.sparse.csgraph.shortest_path(W, directed=False, indices=[0],return_predecessors=True)
        g_dist = dist_matrix[0,1]
        path = compute_optimal_path(predecessors, 1)
        
        # update plot
        ph.update(idx[path])
        plt.draw()
        plt.pause(0.5)
        
    if i%5==0:
        name = './results/vis/vis_' + str(d) + 'd_'+str(i) + '.png'
        plt.tight_layout()
        plt.savefig(name)
    
    print('Distance {}, ratio {}'.format(s,g_dist/s))
    ratios.append(g_dist/s)