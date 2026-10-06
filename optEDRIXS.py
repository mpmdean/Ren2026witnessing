import os
import tarfile
import json
import numpy as np
#from bayes_opt_legacy.bayesian_optimization import BayesianOptimization
#from bayes_opt_legacy.event import Events
#from bayes_opt_legacy.logger import JSONLogger
from bayes_opt import BayesianOptimization, Events, JSONLogger
import argparse, importlib.util, sys

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import operator
#from filelock import FileLock


def import_config(path):
    spec = importlib.util.spec_from_file_location("user_config", path)
    cfg = importlib.util.module_from_spec(spec)
    sys.modules["user_config"] = cfg
    spec.loader.exec_module(cfg)
    return cfg

def numgrad(cmap,dEloss,dOm):
    dycmap=(cmap[1:]-cmap[:-1])/(dEloss)
    dxcmap=(cmap[:,1:]-cmap[:,:-1])/(dOm)
    return np.sqrt(dycmap[:,:-1]**2+dxcmap[:-1]**2)

def target_L1sum(params, black_box, rixs_ref):
    simtab = black_box(params)
    simtab=simtab/np.sum(abs(simtab));
    reftab=rixs_ref/np.sum(abs(rixs_ref))
    return -np.sum(abs(reftab-simtab))

def target_L1max(params, black_box, rixs_ref):
    simtab = black_box(params)
    simtab=simtab/np.max(abs(simtab));
    reftab=rixs_ref/np.max(abs(rixs_ref))
    return -np.sum(abs(reftab-simtab))

def target_All(params, black_box, rixs_ref, eres, omres):
    simtab = black_box(params)

    results=np.zeros(4);
    simtab2=simtab/np.sum(abs(simtab));
    reftab2=rixs_ref/np.sum(abs(rixs_ref))
    results[0] = -np.sum(abs(reftab2-simtab2))

    simtab3=simtab/np.max(abs(simtab));
    reftab3=rixs_ref/np.max(abs(rixs_ref))
    results[1] = -np.sum(abs(reftab3-simtab3))
    results[2] = -np.sum(abs(reftab3**2-simtab3**2)**0.5)

    simtab4=numgrad(simtab/np.max(abs(simtab)),eres,omres)
    reftab4=numgrad(rixs_ref/np.max(abs(rixs_ref)),eres,omres)
    results[3] = -np.sum(abs(reftab4-simtab4))

    return results

def run_bayesian_optimization(record, func):
    """
    Runs Bayesian optimization multiple times and logs results.

    Parameters:
        record (dict): Dictionary containing configuration:
            - name (str): base name for output files
            - num_runs_global (int): number of scripts run in parallel
            - num_runs_local (int): number of independent runs in script
            - num_iters (int): total iterations per run
            - pbounds (dict): parameter bounds for optimization
            - rand_seed (list or np.ndarray): placeholder for seeds, will be filled
            - init_seed (int) initial seed
            - any other keys will be preserved
        func (callable): objective function to maximize
        output_dir (str): directory to store logs and results

    Returns:
        dict: updated record with used random seeds
    """



    num_runs_global = int(record['num_runs_global'])
    num_runs_local = int(record['num_runs_local'])
    num_iters = int(record['num_iters'])
    run_ind = int((record['run_ind'])  )
    datastr = record['name']+'_run'+str(run_ind)
    pbounds = record['pbounds']

    # Ensure output directories exist
    os.makedirs(record['output_dir'], exist_ok=True)
    temp_dir = datastr + '_logs'
    os.makedirs(temp_dir, exist_ok=True)

    # Initialize seed list
    seeds = []

    for i in range(1, num_runs_local + 1):
        # Determine a reproducible random state
        rand_state = record['init_seed'] + i
        seeds.append(rand_state)

        # Set up optimizer
        optimizer = BayesianOptimization(
            f=func,
            pbounds=pbounds,
            random_state=rand_state,
        )

        # Log to JSON
        log_path = os.path.join(temp_dir, f"{datastr}_{i}.log.json")
        logger = JSONLogger(path=log_path)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        # Run optimization: initial random points + iterations

        optimizer.maximize(
            init_points = record['init_points'],
            n_iter = num_iters - record['init_points'],
        )

        print(f"Run {i}/{num_runs_local} complete. Best result: {optimizer.max}")

    # Update record with used seeds
    record['rand_seed'] = seeds

    # Create compressed archive of logs
    tar_path = os.path.join(record['output_dir'], f"{datastr}.bz2")
    with tarfile.open(tar_path, "w:bz2") as tar:
        for i in range(1, num_runs_local + 1):
            tar.add(os.path.join(temp_dir, f"{datastr}_{i}.log.json"),arcname=f"{datastr}{i}.log.json")

    # Clean up individual log files
    for i in range(1, num_runs_local + 1):
        os.remove(os.path.join(temp_dir, f"{datastr}_{i}.log.json"))
    os.rmdir(temp_dir)

    # Append record to a master JSONL file
    master_file = os.path.join(record['output_dir'], f"{datastr}.json")
    with open(master_file, 'a') as fp:
        json.dump(record, fp)
        fp.write('\n')

    return record

def reconstBOresults(record):
    input_dir = record['output_dir']
    num_runs_global = int(record['num_runs_global'])
    num_runs_local = int(record['num_runs_local'])
    num_iters = int(record['num_iters'])
    data = []

    for index in range(1,num_runs_global+1):
        datastr=record['name']+'_run'+str(index);
        recordfile=os.path.join(input_dir,f"{datastr}.json")

        rlist = []
        with open(recordfile) as f:
                for line in f:
                    rlist.append(json.loads(line))

        rec_in=next(x for x in rlist if x['name']==record['name'])
        pbounds = rec_in['pbounds']
        numParams=len(pbounds.keys())

        tar = tarfile.open(os.path.join(input_dir, f"{datastr}.bz2"), "r:bz2")

        for i in range (1,num_runs_local+1):
            data.append([])
            #with tar.extractfile(f"{datastr}_logs/{datastr}_{i}.log.json") as f:
            with tar.extractfile(f"{datastr}{i}.log.json") as f:

                for line in f:
                    data[-1].append(json.loads(line))

    numRuns=num_runs_global*num_runs_local

    t=np.array([[data[i][j]['target'] for j in range(num_iters)] for i in range(numRuns)])
    params=np.zeros([numParams,numRuns,num_iters])
    q = 0
    labels=list(data[0][0]['params'].keys())
    for key in  data[0][0]['params']:
        params[q] = np.array([[data[i][j]['params'][key] for j in range(num_iters)] for i in range(numRuns)])
        q=q+1

    params2=np.transpose(params,axes=[1,2,0]);

    tAll=t.reshape(-1)
    paramsAll=params2.reshape([numRuns*num_iters,numParams])
    return {'paramsAll' : paramsAll, 'tAll' : tAll, 'labels' : labels}

_ops = {
    "==" : operator.eq,
    "!=" : operator.ne,
    "<"  : operator.lt,
    "<=" : operator.le,
    ">"  : operator.gt,
    ">=" : operator.ge,
}

def getMask(len_thresh, filters, col):
    mask_p = np.ones(len_thresh, dtype=bool)
    for clause in filters:
        lhs = col[ clause["lhs"] ]
        # rhs may be a literal or a param name
        rhs_spec = clause["rhs"]
        if isinstance(rhs_spec, str):
            rhs = col[rhs_spec]
        else:
            # broadcast a scalar into a full vector
            rhs = np.full_like(lhs, rhs_spec, dtype=float)

        opfunc = _ops[ clause["op"] ]
        mask_p &= opfunc(lhs, rhs)
    return mask_p

def getDistanceMatrix(fun,valtab,coordsfin):
    inds=np.arange(len(valtab))
    disttab=np.zeros([len(inds),len(inds)])
    for i in range(len(inds)):
        print(i)
        for j in range(i+1,len(inds)): # evaluate the coordinates at 20 uniformly spaced points between current point and remaining points
            coordtab = [coordsfin[inds[i]]+k*(coordsfin[inds[j]]-coordsfin[inds[i]]) for k in np.linspace(0,1,20)]
            coordtab=np.array(coordtab).reshape(-1,len(coordsfin[0]));
            funtab=[]
            for k in range(len(coordtab)): #  calculate the distance at the coordinates of coordtab
                funtab.append(fun(coordtab[k]))
            disttab[i][j] = np.max(funtab)
            disttab[j][i] = disttab[i][j]
    return disttab

def cluster_points(distance_matrix, threshold):
    """
    Clusters points based on the distance matrix.

    Two points are in the same cluster if the distance between them is less than the threshold.

    Parameters:
    - distance_matrix (np.array): An N x N array of distances.
    - threshold (float): The threshold distance for linking points.

    Returns:
    - labels (np.array): An array of cluster labels for each point.
    """
    # Create an adjacency matrix: 1 if distance < threshold, 0 otherwise.
    # We ignore self connections by setting the diagonal to 0.
    adj = (distance_matrix < threshold).astype(int)
    np.fill_diagonal(adj, 0)

    # Convert the adjacency matrix to a sparse matrix
    graph = csr_matrix(adj)

    # Find connected components
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    return labels

def getTCC2(valtab,disttab,thresh_fac):
    inds=np.arange(len(valtab))
    group=[]
    clusters=[]
    while (len(inds)>0):
        indcur=np.argmin(valtab[inds])
        threshold = thresh_fac*np.min(valtab[inds])
        test0= cluster_points(disttab[inds][:,inds], threshold);
        test = test0 == test0[indcur]

        group2=np.where(test)[0]
        group2=inds[group2]
        clusters.append(group2.tolist())
        group=np.concatenate([group,group2]).astype(np.int32)
        ind1=np.ones(len(valtab))
        ind1[group]=0
        inds=np.arange(len(valtab))
        inds=inds[ind1==1]
    return clusters
