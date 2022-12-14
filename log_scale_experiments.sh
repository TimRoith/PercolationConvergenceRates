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
# python3 compute_dist.py -s 9000.0 -n 30 -f 0.5 -t 100 -p -c 50
# python3 compute_dist.py -s 9000.0 -n 30 -f 0.7 -t 100 -p -c 50
python3 compute_dist.py -n 6 -f 0.5 -d 3 -t 100 -m 100 -p -c 50
python3 compute_dist.py -n 6 -f 0.6 -d 3 -t 100 -m 100 -p -c 50
python3 compute_dist.py -n 6 -f 0.7 -d 3 -t 100 -m 100 -p -c 50
python3 compute_dist.py -n 6 -f 0.8 -d 3 -t 100 -m 100 -p -c 50
python3 compute_dist.py -n 6 -f 0.9 -d 3 -t 100 -m 100 -p -c 50
#python3 compute_dist.py -n 6 -f 1.0 -t 100 -m 100 -p -c 50