import edrixs
import numpy as np

from utils_Ce import saveops_Ce, RIXS_runner

from optEDRIXS import target_L1sum

rixs_ref=np.load('Ce.npy');
rixs_ref_Emap=np.loadtxt('CeEmap.numpy');
rixs_ref_Emap_Large=np.loadtxt('CeEmapEg.numpy');

output_dir ='Thalia'

# descriptions to be saved in log file
descr='Ba3CeIr2O9 experiment, t2g restricted atomic model, L1 divergence measure, sum normalized'
method="BayesianOptimization with out of the box hyperparameters"
tfunc="distance_neg"

temperature = 9  # in K

indices = np.array([ 68,  78,  87,  97, 107, 117, 126, 136, 146, 155, 165, 175, 185,
       194])
Ls = np.arange(0.0, 20.1, 0.1)[indices]

Qs = np.sqrt(1.228965**2/4+(Ls*0.426991866)**2)
tths = np.rad2deg(np.arcsin(1.1054837271511369*Qs/4./np.pi)*2)

L_Emap = 18.94
Q_Emap = np.sqrt(1.228965**2/4+(L_Emap*0.426991866)**2)
tth_Emap = np.rad2deg(np.arcsin(1.1054837271511369*Q_Emap/4./np.pi)*2)
thin_Emap = tth_Emap/2 - np.rad2deg(np.arctan(1.228965/2/L_Emap/0.426991866))

geometry = {
    'phi' : 10.,
    'HKL' : [-0.5,0.0,0.0],
    'Ls' : Ls,
    'Qs' : Qs,
    'tths' : tths,
    'thins' : tths/2 - np.rad2deg(np.arctan(1.228965/2/Ls/0.426991866)),
    'omega' : -558.5,

    'L_Emap' : L_Emap,
    'Q_Emap' : Q_Emap,
    'tth_Emap' : tth_Emap,
    'thin_Emap' : thin_Emap,
    'omega_Emap' : np.arange(-562.5,-554.5),

    #polarization
    'pol' : [(0, 0), (0, np.pi/2.0)]
}

eloss0 = np.linspace(-0.3975,1.9925,240)
choose = eloss0 >= 0.2

resolution = {

    # energy loss bounds and resolution
    'eloss' : eloss0[choose],
    'eloss_Emap' : np.arange(0.21,2.01,0.01),
    'eloss_Emap_large' : np.arange(0.21,6.00,0.01)
}

data_str='Ba3CeIr2O9_test4'
data_opt='_subseq_optV4V'
num_runs_global=9
num_runs_local=6
num_iters=1000
init_points = 10
init_seed = 0

# RIXS_params
sigma=0.075
gamma_c=2.47
fraction=0.
eline=None

basis_n1=np.array(edrixs.get_fock_bin_by_N(8,0,12,11,6,5,6,6))[:,
np.hstack([[0,1,8,9,10,11,2,3,12,13,4,5,14,15,16,17,6,7,18,19],np.arange(20,32)])]
basis_n2=np.array(edrixs.get_fock_bin_by_N(8,0,12,11,6,6,6,5))[:,
np.hstack([[0,1,8,9,10,11,2,3,12,13,4,5,14,15,16,17,6,7,18,19],np.arange(20,32)])]
basis_i=np.array(edrixs.get_fock_bin_by_N(8,0,12,10,6,6,6,6),dtype=np.int32)[:,
np.hstack([[0,1,8,9,10,11,2,3,12,13,4,5,14,15,16,17,6,7,18,19],np.arange(20,32)])]

sptab2i, sptab2n1, sptab2n2, sptab4i, sptab4n1, sptab4n2, sptabtr = saveops_Ce(basis_i,basis_n1,basis_n2)

U_dd_param=3.
U_q_param=2.5

J_dd_param=0.4
F4F2_ratio = 0.625
F2_dd = J_dd_param * 14. / (1. + F4F2_ratio)
F4_dd = F2_dd * F4F2_ratio

G1_dp = 0.889*0.9
G3_dp = 0.528*0.9

F0_dd = U_dd_param - 4.0 / 49.0 * F2_dd - 4.0 / 49.0 * F4_dd
F0_dp = edrixs.get_F0('dp', G1_dp, G3_dp) + U_q_param





true_values=dict(tenDq=3.5, theta=50.86, a0=0.,
                 Vdd_sigma=-1.4,Vdd_pi=-0.66,Vdd_delta=0.05,
                 kappa = 0.1, soc = 0.33,
                 soc_c = 1140.332, F0_dd = F0_dd, F2_dd = F2_dd,
                 F4_dd = F4_dd, F0_dp = F0_dp,
                 F2_dp = 1.03*0.9, G1_dp = G1_dp,
                 G3_dp = G3_dp, Gam_c = gamma_c, sigma = sigma)

# parameter bounds
pbounds = {#'tenDq': (2.5,4.5),
           'Vdd_sigma': (-3.0,0.0),
           'Vdd_pi': (-1.5,0.0),
           'Vdd_delta': (-0.5,0.5),
           'F2_dd': (true_values['F2_dd']*0.1, true_values['F2_dd']*1.5),
           'F4_dd': (true_values['F4_dd']*0.1, true_values['F4_dd']*1.5),
           'F2_dp': (true_values['F2_dp']*0.1,true_values['F2_dp']*1.5),
           'G1_dp': (true_values['G1_dp']*0.1,true_values['G1_dp']*1.5),
           'G3_dp': (true_values['G3_dp']*0.1,true_values['G3_dp']*1.5),
           'soc' : (0.0,0.5),
           'soc_c' : (1130.332,1150.332)
           }

fixed_params = {
       'tenDq' : true_values['tenDq'],

       'theta' : true_values['theta'],
       'a0' : true_values['a0'],
       'kappa' : true_values['kappa'],
       'Gam_c' : true_values['Gam_c'],
       'sigma' : true_values['sigma'],
       'U_dd_param' : U_dd_param,
       'U_q_param' : U_q_param
}

RIXS_params = dict(sigma=true_values['sigma'], fraction=0.,

                   geometry=geometry,resolution=resolution,
                   T=temperature, eline=None)


rixs_funV = RIXS_runner(RIXS_params,[sptab2i, sptab2n1, sptab2n2, sptab4i, sptab4n1, sptab4n2, sptabtr])
rixs_funV_E = RIXS_runner(RIXS_params,[sptab2i, sptab2n1, sptab2n2, sptab4i, sptab4n1, sptab4n2, sptabtr],mode='E')
rixs_funV_EL = RIXS_runner(RIXS_params,[sptab2i, sptab2n1, sptab2n2, sptab4i, sptab4n1, sptab4n2, sptabtr],mode='EL')


def fun(Vdd_sigma,Vdd_pi, Vdd_delta, F2_dd, F4_dd, F2_dp, G1_dp, G3_dp, soc,soc_c):
    params = {**locals(), **fixed_params}
    return target_L1sum(params,rixs_funV,rixs_ref)+target_L1sum(params,rixs_funV_E,rixs_ref_Emap)

def funE(Vdd_sigma,Vdd_pi, Vdd_delta, F2_dd, F4_dd, F2_dp, G1_dp, G3_dp, soc,soc_c):
    params = {**locals(), **fixed_params}
    return target_L1sum(params,rixs_funV_E,rixs_ref_Emap)

# will be set by running script
rand_seed = np.zeros(num_runs_local, dtype=np.int32)

# parameters for optimization
threshold_factor = 1.2
max_eval_greedy = 3600

greedy_bounds = {
        'Vdd_sigma': (-3.0,3.0),
        'Vdd_pi': (-1.5,1.5),
        'Vdd_delta': (-0.5,0.5),
        'F2_dd' : [true_values['F2_dd']*0.001,true_values['F2_dd']*1.5],
        'F2_dp' : [true_values['F2_dp']*0.001, true_values['F2_dp']*1.5],
        'F4_dd' : [true_values['F4_dd']*0.001, true_values['F2_dd']],
        'G1_dp' : [true_values['G1_dp']*0.001, true_values['G1_dp']*1.5],
        'G3_dp' : [true_values['G3_dp']*0.001, true_values['G3_dp']*1.5],
        'tenDq' : [2.5, 4.5],
        'soc' : [0., 1.0],
        'soc_c' : [1130.332,1150.332],
        'Gam_c' : [2.0,3.0],
        }

greedy_filters = [
    { "lhs": "F2_dd",  "op": ">=", "rhs": "F4_dd"  },

]

fixed_params_greedy = {

       'theta' : true_values['theta'],
       'a0' : true_values['a0'],
       'kappa' : true_values['kappa'],
       'sigma' : true_values['sigma'],
       'U_dd_param' : U_dd_param,
       'U_q_param' : U_q_param
}

def funGreedy(Vdd_sigma, Vdd_pi, Vdd_delta, F2_dd, F2_dp, F4_dd, G1_dp, G3_dp, tenDq,soc,soc_c, Gam_c):
    params = {**locals(), **fixed_params_greedy}
    return -(target_L1sum(params,rixs_funV,rixs_ref)+target_L1sum(params,rixs_funV_E,rixs_ref_Emap))

def funEGreedy(Vdd_sigma, Vdd_pi, Vdd_delta, F2_dd, F2_dp, F4_dd,  G1_dp, G3_dp, tenDq, soc,soc_c, Gam_c):
    params = {**locals(), **fixed_params}
    return target_L1sum(params,rixs_funV_E,rixs_ref_Emap)

record=dict(name = data_str,optname=data_opt,  num_runs_local = num_runs_local, num_runs_global = num_runs_global, init_points = init_points,
            init_seed = init_seed, run_ind = 0, num_iters = num_iters,
            true_values = true_values, pbounds = pbounds, method = method,
            tfunc = tfunc, descr = descr, output_dir = output_dir,threshold_factor=threshold_factor,
            max_eval_greedy=max_eval_greedy, greedy_bounds = greedy_bounds, greedy_filters = greedy_filters,rand_seed = rand_seed.tolist())
