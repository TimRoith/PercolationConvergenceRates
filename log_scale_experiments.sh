# <><><><><><><><><><><><><><><><><><><><>
# Compute Percolation Distances
# <><><><><><><><><><><><><><><><><><><><>
#
# Options:
#   -h (--help): Print help.
#   -d (--d=): Dimension.
#   -s (--s_max=): Maximal size of the domain.
#   -n (--num_s=): Number of discretization points for s.'
#   -f (--factor=): Factor in fromt of log scaling (default: a=1).
#   -t (--num_trials=): Number of trials to run (default=10).
#   -p (--parallel): Use parallel processing over the trials.
#   -c (--num_cores=): Number of cores to use in parallel processing (default=1).
#   -v (--verbose): Verbose mode.

#All experiments star domain
python3 compute_dist.py -s 2000.0 -n 30 -f 0.1 -t 100 -p -c 50
python3 compute_dist.py -s 2000.0 -n 30 -f 0.2 -t 100 -p -c 50
python3 compute_dist.py -s 2000.0 -n 30 -f 0.4 -t 100 -p -c 50
python3 compute_dist.py -s 2000.0 -n 30 -f 0.6 -t 100 -p -c 50
python3 compute_dist.py -s 2000.0 -n 30 -f 0.8 -t 100 -p -c 50
python3 compute_dist.py -s 2000.0 -n 30 -f 1.0 -t 100 -p -c 50
python3 compute_dist.py -s 2000.0 -n 30 -f 1.2 -t 100 -p -c 50