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
params = {
's_min' : 1,
's_max' : 10000,# maximal domain size in the first component
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
                               "h:s:m:n:d:f:t:pc:v",
                               ["help","s_max=","s_min=","num_s=","dimension=","factor=", 
                                "num_trials=",
                                "parallel","num_cores=","verbose"])
except getopt.GetoptError:
    print_help()
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-h", "--help"):
        print_help()
        sys.exit()
    elif opt in ("-d", "--dimension"):
        params['d'] = int(arg)
    elif opt in ("-f", "--factor"):
        params['factor'] = float(arg)
    elif opt in ("-s", "--s_max"):
        params['s_max'] = float(arg)
    elif opt in ("-m", "--s_min"):
        params['s_min'] = float(arg)
    elif opt in ("-n", "--num_s"):
        params['num_s'] = int(arg)
    elif opt in ("-t", "--num_trials"):
        params['num_trials'] = int(arg)
    elif opt in ("-p", "--parallel"):
        parallel = True
    elif opt in ("-c", "--num_cores"):
        params['num_cores'] = int(arg)
        
params['scaling'] = utils.log_scale(d=d, factor=params['factor'])

#%% domain and point process setup


# initialize the function for creating a Poisson point process
#s_disc = np.linspace(params['s_min'], params['s_max'], params['num_s']) # array for s discretization

s_disc = [2**i * params['s_min'] for i in range(params['num_s']) if (2**i * params['s_min'])<= params['s_max']]
#%% Trial
def trial(T):
    seed = T**2
    np.random.seed(seed)
    
    print('<>'*10, flush = True)
    print('Starting Trial:' + str(T),flush = True)
    print('<>'*10,flush = True) 
    
    for i,s in enumerate(s_disc):
        print("Trial " + str(T) +" starts computing for s=" + str(s),flush = True)
        bounds = np.zeros((d, 2))
        bounds[0,0] = -s**(1/d)
        bounds[0,1] = s + s**(1/d)
        bounds[1:,0] = -s**(1/d)
        bounds[1:,1] = s**(1/d)
        delta = bounds[:,1] - bounds[:,0]
        area = np.prod(delta)
        
        PP = utils.Poisson_process(bounds, lamda = params['lamda'], area = area, d=params['d'])
        points = np.vstack([np.zeros((3, d)), PP()]) 
        # update target point
        points[1,0] = 0.5*s
        points[2,0] = s
        
        h = params['scaling'](s)
        # get weight matrix
        kd_tree = scipy.spatial.KDTree(points)
        W = kd_tree.sparse_distance_matrix(kd_tree, h)
        
        if np.sum(W[0,:]) == 0:
            g_dist_1 = np.inf
            g_dist_2 = np.inf
        else:
            # dist_matrix = scipy.sparse.csgraph.dijkstra(W, directed=False, indices=[0])
            dist_matrix = scipy.sparse.csgraph.shortest_path(W, directed=False, indices=[0],return_predecessors=False)
            g_dist_1 = dist_matrix[0,1]
            g_dist_2 = dist_matrix[0,2] 
        
        with mp_arr.get_lock():
            mp_arr[T*params['num_s']+i] = g_dist_1
            
        with mp_arr2.get_lock():
            mp_arr2[T*params['num_s']+i] = g_dist_2
        

#%% Main Loop
def init_arr(mp_arr_, mp_arr2_):
    global mp_arr
    global mp_arr2
    mp_arr = mp_arr_
    mp_arr2 = mp_arr2_
    

num_cores = min(mp.cpu_count(),params['num_cores'])
def main():    
    mp_arr = mp.Array('d', params['num_trials']*params['num_s']) # shared, can be used from multiple processes
    mp_arr2 = mp.Array('d', params['num_trials']*params['num_s'])
    
    pool = mp.Pool(num_cores, initializer=init_arr, initargs=(mp_arr, mp_arr2))
    
    with closing(pool):
        pool.imap_unordered(trial, range(params['num_trials']))
    pool.join()
    return mp_arr, mp_arr2
    
if __name__ == '__main__':    
    arr, arr2 = main()
    
    #%% set up csv file and save parameters
    time_str = time.strftime("%Y%m%d-%H%M%S")
    fname = "results/distances-" + str(params['factor']) + "-" + time_str + '.csv'
    with open(fname, 'w') as f:
        writer = csv.writer(f, lineterminator = '\n')
        for p in params:
            writer.writerow([p, str(params[p])])
            
        writer.writerow(['T'] + list(s_disc))
        for T in range(params['num_trials']):
            row = arr[T*params['num_s']:((T+1)*params['num_s'])]
            row = [T] + list(row)
            writer.writerow(row)
            
        for T in range(params['num_trials']):
            row = arr2[T*params['num_s']:((T+1)*params['num_s'])]
            row = [T] + list(row)
            writer.writerow(row)
    
        

