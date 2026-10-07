from optEDRIXS import import_config, getDistanceMatrix, getTCC2
import argparse, importlib.util
import numpy as np
import sys
import json

p = argparse.ArgumentParser(
    description="Run RIXS Greedy refinement optimization for a given config and index."
)
p.add_argument("--config-py", required=True,
                   help="Path to the Python config file")
p.add_argument("--input", required=True, help="Path to the input coordinates file (e.g. .txt)")
p.add_argument("--output", required=True,
                   help="output file name")

                   

args = p.parse_args()

cfg = import_config(args.config_py)

record = cfg.record

coordsfin=np.loadtxt(args.input)

greedy_keys=record['greedy_bounds'].keys()
params={k: record['true_values'][k] for k in record['true_values'].keys() if k in greedy_keys}

def f_wrapped(x):
        # start from the fully-populated baseline, then overwrite the variables by position
        params.update({k: float(v) for k, v in zip(greedy_keys, x)})
        return cfg.funGreedy(**params)
        
valtab=[]
for i in range(len(coordsfin)):
    valtab.append(f_wrapped(coordsfin[i]))
valtab=np.array(valtab)

disttab=getDistanceMatrix(f_wrapped,valtab,coordsfin)

clusterinds=getTCC2(valtab,disttab, 1.25)

rec_cluster={
    'valtab' : valtab.tolist(),
    'disttab' : disttab.tolist(),
    'clusterinds' : clusterinds
}

master_file = f"{args.output}.json"
with open(master_file, 'w') as fp:
    json.dump(rec_cluster, fp, indent=2)
        
#np.savetxt("disttab_NiCl2_Final.txt",disttab)
