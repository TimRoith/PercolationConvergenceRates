#%% imports
import numpy as np
import scipy.stats
import time
import multiprocessing as mp
import csv
from contextlib import closing
import sys,getopt
# custom imports
import utils
#%% Help for Params
#Print help
def print_help():
    
    print('<>'*30)
    print(' Compute Percolation Distances')
    print('<>'*30)
    print(' ')
    print('Options:')
    print('   -h (--help): Print help.')
    print('   -d (--d=): Dimension.')
    print('   -s (--s_max=): Maximal size of the domain.')
    print('   -n (--num_s=): Number of discretization points for s.') 
    print('   -f (--factor=): Factor in fromt of log scaling (default: a=1).')
    print('   -t (--num_trials=): Number of trials to run (default=10).')
    print('   -p (--parallel): Use parallel processing over the trials.')
    print('   -c (--num_cores=): Number of cores to use in parallel processing (default=1).')
    print('   -v (--verbose): Verbose mode.')
#%% parameters
d = 2
factor = 1.
params = {
's_max' : 2000,# maximal domain size in the first component
'num_s' : 10,# number of points for s
'd' : d, # spatial dimension
'lamda' : 1,# intensity of the point process
'num_cores' : 40,
'num_trials': 100,
'factor':1}

#%% read command line
#Read command line parameters
try:
    opts, args = getopt.getopt(sys.argv[1:],
                               "h:s:n:d:f:t:pc:v",
                               ["help","s_max=","num_s=","d=","factor=", 
                                "num_trials=",
                                "parallel","num_cores=","verbose"])
except getopt.GetoptError:
    print_help()
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-h", "--help"):
        print_help()
        sys.exit()
    elif opt in ("-d", "--d"):
        params['d'] = int(arg)
    elif opt in ("-f", "--factor"):
        params['factor'] = float(arg)
    elif opt in ("-s", "--s_max"):
        params['s_max'] = float(arg)
    elif opt in ("-n", "--num_s"):
        params['num_s'] = int(arg)
    elif opt in ("-t", "--num_trials"):
        num_trials = int(arg)
    elif opt in ("-p", "--parallel"):
        parallel = True
    elif opt in ("-c", "--num_cores"):
        params['num_cores'] = int(arg)
        
params['scaling'] = utils.log_scale(d=d, factor=params['factor'])

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
    
        

