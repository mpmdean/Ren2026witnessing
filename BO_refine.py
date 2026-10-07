import argparse
import pybobyqa
import sys

from optEDRIXS import import_config, reconstBOresults, getMask
import numpy as np

def _normalize_bounds(bounds, name):
    all_none = all(v is None for v in bounds)
    any_none = any(v is None for v in bounds)
    any_not_none = any(v is not None for v in bounds)

    if all_none:
        return None  # collapse [None, None, ...] -> None
    if any_none and any_not_none:
        raise ValueError(f"Mixed bounded/unbounded {name} limits are not supported.")
    return bounds


p = argparse.ArgumentParser(
    description="Run RIXS Greedy refinement optimization for a given config and index."
)
p.add_argument("--config-py", required=True,
                   help="Path to the Python config file")
p.add_argument(
        "index1",
        type=int,
        nargs="?",           # allow it to be omitted
        default=0,           # use 1 if not provided
        help="Start index (default: %(default)s)"
    )
p.add_argument(
        "index2",
        type=int,
        nargs="?",           # allow it to be omitted
        default=1,           # use 1 if not provided
        help="Final index (default: %(default)s)"
    )
p.add_argument("--num-points", action="store_true",
               help="Return number of available points to optimize and exit")
args = p.parse_args()

cfg = import_config(args.config_py)

# for all ions with occupation n_occu in the database:

record = cfg.record

params_dict = reconstBOresults(record)

tAll = params_dict['tAll']
paramsAll = params_dict['paramsAll']
paramsAll = paramsAll[-tAll < np.min(-tAll)*record['threshold_factor']]
tAll = tAll[-tAll < np.min(-tAll)*record['threshold_factor']]


len_thresh = len(tAll)

filters = record['greedy_filters']

param_names = params_dict['labels']
col = { name: paramsAll[:, i]
        for i, name in enumerate(param_names) }

mask_p = getMask(len_thresh, filters, col)

plistSelect = paramsAll[mask_p]

if args.num_points:
    print("Number of points available for optimization: "+str(len(plistSelect)))
    sys.exit(0)

lower_bounds, upper_bounds = map(list, zip(*record['greedy_bounds'].values()))
lower_bounds = _normalize_bounds(lower_bounds, "lower")
upper_bounds = _normalize_bounds(upper_bounds, "upper")


greedy_keys=record['greedy_bounds'].keys()

bounds = (lower_bounds, upper_bounds)

resTab2=[]


for i in range(args.index1,args.index2):
    param_updates = dict(zip(param_names, plistSelect[i]))

    greedy_start={k: record['true_values'][k] for k in record['true_values'].keys() if k in greedy_keys}
    print(param_updates)
    greedy_start.update(param_updates)

    x0 = np.array([greedy_start[k] for k in greedy_keys], dtype=float)
    print("x0: ")
    print(x0)
    print("maxfun: ")
    print(record['max_eval_greedy'])
    print("bounds: ")
    print(bounds)

    def f_wrapped(x):
        # start from the fully-populated baseline, then overwrite the variables by position
        params = greedy_start.copy()
        params.update({k: float(v) for k, v in zip(greedy_keys, x)})
        return cfg.funGreedy(**params)
    resTab2.append(pybobyqa.solve(f_wrapped, x0,maxfun=record['max_eval_greedy'],bounds=bounds,scaling_within_bounds=True ))
    with open(record['name']+record['optname']+str(args.index1)+'_'+str(args.index2)+"details.txt", 'a') as f:
        print(resTab2[-1], file=f)
    with open(record['name']+record['optname']+str(args.index1)+'_'+str(args.index2)+".txt", 'a') as f:
        print(resTab2[-1].x, file=f)
