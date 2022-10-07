#%% imports
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import graphlearning as gl
from matplotlib.pyplot import cm
import time
import pickle
import os
import multiprocessing as mp
from joblib import Parallel, delayed
import csv
from contextlib import closing
# custom imports
import utils
#%% parameters
d = 2
params = {
's_max' : 2000,# maximal domain size in the first component
'num_s' : 10,# number of points for s
'd' : d, # spatial dimension
'lamda' : 1,# intensity of the point process
'num_cores' : 8,
'num_trials': 100,
'scaling' : utils.log_scale(d=d)}

#%% domain and point process setup
bounds = np.zeros((d, 2))
bounds[0,0] = -0.25 * params['s_max']
bounds[0,1] = 1.25 * params['s_max']
bounds[1:,0] = -params['s_max']**(1/d)
bounds[1:,1] = params['s_max']**(1/d)

# initialize the function for creating a Poisson point process
delta = bounds[:,1] - bounds[:,0]
area = np.prod(delta)
PP = utils.Poisson_process(bounds, lamda = params['lamda'], area = area, d=d)
s_disc = np.linspace(1.0, params['s_max']/2, params['num_s']//2) # array for s discretization
s_disc = np.sort(np.hstack([s_disc, 2*s_disc]))
params['num_s'] = len(s_disc)


#%% Trial
def trial(T):
    np.random.seed(T)
    points = np.vstack([np.zeros((2, d)), PP()])
    print('<>'*10, flush = True)
    print('Starting Trial:' + str(T),flush = True)
    print('<>'*10,flush = True)  
    
    for i,s in enumerate(s_disc):
        # update target point
        points[1,0] = s
        
        # get points in frame
        idx, = np.where(np.prod(np.abs(points) < 1.25*s, axis=1))
        loc_points = points[idx,:]
        if len(idx)>=2:
            h = params['scaling'](s)
            # get weight matrix
            #W = gl.weightmatrix.epsilon_ball(loc_points, h, kernel='distance')
            kd_tree = scipy.spatial.KDTree(loc_points)
            W = kd_tree.sparse_distance_matrix(kd_tree, h)
            
            if np.sum(W[0,:]) == 0:
                g_dist = np.inf
            else:
                # dist_matrix = scipy.sparse.csgraph.dijkstra(W, directed=False, indices=[0])
                dist_matrix = scipy.sparse.csgraph.shortest_path(W, directed=False, indices=[0],return_predecessors=False)
                g_dist = dist_matrix[0,1] 
            
            with mp_arr.get_lock():
                mp_arr[T*params['num_s']+i] = g_dist
        

#%% Main Loop
def init_arr(mp_arr_):
    global mp_arr
    mp_arr = mp_arr_
    

num_cores = min(mp.cpu_count(),params['num_cores'])
def main():    
    mp_arr = mp.Array('d', params['num_trials']*params['num_s']) # shared, can be used from multiple processes

    pool = mp.Pool(num_cores, initializer=init_arr, initargs=(mp_arr,))
    
    with closing(pool):
        pool.imap_unordered(trial, range(params['num_trials']))
    pool.join()
    return mp_arr
    
if __name__ == '__main__':    
    arr = main()
    
    #%% set up csv file and save parameters
    time_str = time.strftime("%Y%m%d-%H%M%S")
    fname = "results/distances-" + time_str + '.csv'
    with open(fname, 'w') as f:
        writer = csv.writer(f, lineterminator = '\n')
        for p in params:
            writer.writerow([p, str(params[p])])
            
        writer.writerow(['T'] + list(s_disc))
        for T in range(params['num_trials']):
            row = arr[T*params['num_s']:((T+1)*params['num_s'])]
            row = [T] + list(row)
            writer.writerow(row)
    
        

