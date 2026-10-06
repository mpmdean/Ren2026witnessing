import edrixs
import numpy as np

from utils_Ta import saveops, RIXS_runner

from optEDRIXS import target_L1sum

rixs_ref=dict(np.load('Ta_Ldep.npz'))['intensity'];
rixs_ref_Emap=dict(np.load('Ta_Edep.npz'))['arr_0'];
rixs_ref_Emap_Large=dict(np.load('Ta_Edep_Large.npz'))['arr_0'];

output_dir ='Thalia'

# descriptions to be saved in log file
descr='Ba3CeIr2O9 experiment, t2g restricted atomic model, L1 divergence measure, sum normalized'
method="BayesianOptimization with out of the box hyperparameters"
tfunc="distance_neg"

temperature = 11  # in K

indices = np.array([95, 105, 115, 125, 135, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 193])


Bmat = np.array([[ 1.09177851e+00,  6.30338615e-01, -1.33704305e-16],
                 [ 0.00000000e+00,  1.26067723e+00, -1.33704305e-16],
                 [ 0.00000000e+00, -0.00000000e+00,  4.36029515e-01]])

# Motors table for the L-scan (this is the one used for momentum dependence in  Ta scripts)
# Columns: H, K, L, tth, th, chi, phi

motors = np.vstack([[-0.37135718, -0.19678939, 0.0,  6.35613593, -93.84443325,  -35.71702316, -10.],
                    [-0.36642012, -0.20042758, 0.5,  6.70426443, -70.27088312, -112.93945895, -10.],
                    [-0.36148305, -0.20406577, 1.0,  7.69383586, -54.60755422, -102.08468304, -10.],
                    [-0.35654599, -0.20770396, 1.5,  9.11950715, -42.55814882,  -98.79459525, -10.],
                    [-0.35160893, -0.21134215, 2.0, 10.81173739, -33.62642692,  -97.21722821, -10.],
                    [-0.34667186, -0.21498034, 2.5, 12.66610405, -26.87058629,  -96.29276610, -10.],
                    [-0.34173480, -0.21861853, 3.0, 14.62328021, -21.57673293,  -95.68563709, -10.],
                    [-0.33679773, -0.22225672, 3.5, 16.64946876, -17.27553243,  -95.25647221, -10.],
                    [-0.33186067, -0.22589491, 4.0, 18.72485802, -13.66522264,  -94.93705831, -10.],
                    [-0.32692361, -0.22953311, 4.5, 20.83746597, -10.54905286,  -94.69008224, -10.],
                    [-0.32198654, -0.23317130, 5.0, 22.97986626,  -7.79546702,  -94.49341877, -10.],
                    [-0.31704948, -0.23680949, 5.5, 25.14739707,  -5.31403908,  -94.33311852, -10.],
                    [-0.31211241, -0.24044768, 6.0, 27.33714508,  -3.04092539,  -94.19995153, -10.],
                    [-0.30717535, -0.24408587, 6.5, 29.54734754,  -0.92991142,  -94.08756742, -10.],
                    [-0.30223829, -0.24772406, 7.0, 31.77702859,   1.05323073,  -93.99145380, -10.],
                    [-0.29730122, -0.25136225, 7.5, 34.02577160,   2.93438842,  -93.90831624, -10.],
                    [-0.29236416, -0.25500044, 8.0, 36.29357337,   4.73352253,  -93.83569342, -10.],
                    [-0.28742710, -0.25863863, 8.5, 38.58074920,   6.46630300,  -93.77170940, -10.],
                    [-0.28249003, -0.26227682, 9.0, 40.88787072,   8.14524100,  -93.71490943, -10.],
                    [-0.27739414882818690, -0.26597417911526850, 9.5104268416546700, 43.264375,  9.815250, -93.663173, -10.0],
                    [-0.26764896087249423, -0.27330293275657100, 10.510874050028260, 47.989750, 12.984500, -93.576479, -10.0],
                    [-0.25778835467293990, -0.28057927767632750, 11.510507877509921, 52.808000, 16.063875, -93.504979, -10.0],
                    [-0.24777219675676215, -0.28779323774242505, 12.509375498043902, 57.731750, 19.092500, -93.445016, -10.0],
                    [-0.23796699084652193, -0.29510625975440596, 13.509375345735329, 62.786750, 22.101000, -93.394073, -10.0],
                    [-0.22785287701054702, -0.30221098200350494, 14.507727481478634, 67.976875, 25.115875, -93.349547, -10.0],
                    [-0.22341693910140217, -0.30610439562840724, 15.009870255621223, 70.650125, 26.633375, -93.330210, -10.0],
                    [-0.21800221366851740, -0.30950033549205264, 15.507542476207073, 73.343000, 28.161500, -93.311360, -10.0],
                    [-0.21346254040160886, -0.31339680456990965, 16.009171507913727, 76.108250, 29.703375, -93.294785, -10.0],
                    [-0.20821606842705367, -0.31680014487519276, 16.507547237513325, 78.907625, 31.262125, -93.277723, -10.0],
                    [-0.20356182986295102, -0.32066862160791510, 17.008601568605542, 81.781875, 32.841000, -93.263260, -10.0],
                    [-0.19849298265545512, -0.32414055133336717, 17.507710839799360, 84.708125, 34.443750, -93.248066, -10.0],
                    [-0.19360491108270322, -0.32791598960334883, 18.007898567617170, 87.711875, 36.074250, -93.235229, -10.0],
                    [-0.18850285302241201, -0.33138267089740425, 18.506986904695648, 90.786375, 37.736750, -93.221660, -10.0],
                    [-0.18369770751773082, -0.33514959402719720, 19.007283337061917, 93.955875, 39.436125, -93.209960, -10.0],
                    [-0.17973557818122593, -0.33795464345342190, 19.406928947872096, 96.556750, 40.826125, -93.200047, -10.0]])

Ls = np.arange(0.,19.5,0.1)[indices]

Hs = np.interp(Ls, motors[:,2], motors[:,0])
Ks = np.interp(Ls, motors[:,2], motors[:,1])
HKLs = np.vstack([Hs, Ks, Ls]).T
tths = np.interp(Ls, motors[:,2], motors[:,3])
ths  = np.interp(Ls, motors[:,2], motors[:,4])
chis = np.interp(Ls, motors[:,2], motors[:,5])
phis = np.interp(Ls, motors[:,2], motors[:,6])

Qs_vec = (Bmat @ HKLs.T).T                         # (N,3)
Qs_mag = np.sqrt(np.sum(Qs_vec * Qs_vec, axis=1))  # (N,)

motors_E = np.vstack([[-0.2773941488281869, -0.2659741791152685, 9.51042684165467, 43.264375, 9.81525, -93.663173, -10.0],
                    [-0.26764896087249423, -0.273302932756571, 10.51087405002826, 47.98975, 12.9845, -93.576479, -10.0],
                    [-0.2577883546729399, -0.2805792776763275, 11.510507877509921, 52.808, 16.063875, -93.504979, -10.0],
                    [-0.24777219675676215, -0.28779323774242505, 12.509375498043902, 57.73175, 19.0925, -93.445016, -10.0],
                    [-0.23796699084652193, -0.29510625975440596, 13.509375345735329, 62.78675, 22.101, -93.394073, -10.0],
                    [-0.22785287701054702, -0.30221098200350494, 14.507727481478634, 67.976875, 25.115875, -93.349547, -10.0],
                    [-0.22341693910140217, -0.30610439562840724, 15.009870255621223, 70.650125, 26.633375, -93.33021, -10.0],
                    [-0.2180022136685174, -0.30950033549205264, 15.507542476207073, 73.343, 28.1615, -93.31136, -10.0],
                    [-0.21346254040160886, -0.31339680456990965, 16.009171507913727, 76.10825, 29.703375, -93.294785, -10.0],
                    [-0.20821606842705367, -0.31680014487519276, 16.507547237513325, 78.907625, 31.262125, -93.277723, -10.0],
                    [-0.20356182986295102, -0.3206686216079151, 17.008601568605542, 81.781875, 32.841, -93.26326, -10.0],
                    [-0.19849298265545512, -0.32414055133336717, 17.50771083979936, 84.708125, 34.44375, -93.248066, -10.0],
                    [-0.19360491108270322, -0.32791598960334883, 18.00789856761717, 87.711875, 36.07425, -93.235229, -10.0],
                    [-0.18850285302241201, -0.33138267089740425, 18.506986904695648, 90.786375, 37.73675, -93.22166, -10.0],
                    [-0.18369770751773082, -0.3351495940271972, 19.007283337061917, 93.955875, 39.436125, -93.20996, -10.0],
                    [-0.17973557818122593, -0.3379546434534219, 19.406928947872096, 96.55675, 40.826125, -93.200047, -10.0]])

L_Emap = 18.4

H_Emap = np.interp(L_Emap, motors_E[:, 2], motors_E[:, 0])
K_Emap = np.interp(L_Emap, motors_E[:, 2], motors_E[:, 1])

HKL_Emap  = np.array([H_Emap, K_Emap, L_Emap])
tth_Emap  = float(np.interp(L_Emap, motors_E[:, 2], motors_E[:, 3]))
th_Emap   = float(np.interp(L_Emap, motors_E[:, 2], motors_E[:, 4]))
chi_Emap  = float(np.interp(L_Emap, motors_E[:, 2], motors_E[:, 5]))
phi_Emap  = float(np.interp(L_Emap, motors_E[:, 2], motors_E[:, 6]))

geometry = {
    # L-scan (momentum dependence)
    'Ls'   : Ls,
    'HKLs' : HKLs,          # (N,3) since Ta needs varying H,K too
    'Qs_vec' : Qs_vec,      # (N,3)
    'Qs'   : Qs_mag,        # (N,)
    'tths' : tths,
    'ths'  : ths,
    'chis' : chis,
    'phis' : phis,

    'omega' : -559.0,

    'HKL_Emap' : HKL_Emap,
    'tth_Emap' : tth_Emap,
    'th_Emap'  : th_Emap,
    'chi_Emap' : chi_Emap,
    'phi_Emap' : phi_Emap,
    'omega_Emap' : np.arange(-570.0, -540.0, 0.3)[[25, 32, 35, 38, 42, 48]],

    #polarization
     'pol' : [0, 1]
}

eloss0 = np.arange(0.304,2.003,0.01)

resolution = {

    # energy loss bounds and resolution
    'eloss' : eloss0,
    'eloss_Emap' : eloss0,
    'eloss_Emap_large' : np.arange(0.304,5.96,0.01)
}

data_str='Ba3TaIr2O9_fitEg'
data_opt='_subseq_optV'

num_runs_global=9
num_runs_local=6 #6
num_iters=1000    #1000
init_points = 10
init_seed = 0

# RIXS_params
sigma0   = 0.15
sigmas   = [0.022, 0.022, 0.022, 0.022, 0.054, 0.054]
gamma_c  = 2.47
fraction = 0.5
eline    = None

basis_n1=np.array(edrixs.get_fock_bin_by_N(8,0,12,12,6,5,6,6)+edrixs.get_fock_bin_by_N(8,1,12,11,6,5,6,6))[:,
np.hstack([[0,1,8,9,10,11,2,3,12,13,4,5,14,15,16,17,6,7,18,19],np.arange(20,32)])]
basis_n2=np.array(edrixs.get_fock_bin_by_N(8,0,12,12,6,6,6,5)+edrixs.get_fock_bin_by_N(8,1,12,11,6,6,6,5))[:,
np.hstack([[0,1,8,9,10,11,2,3,12,13,4,5,14,15,16,17,6,7,18,19],np.arange(20,32)])]
basis_i=np.array(edrixs.get_fock_bin_by_N(8,0,12,11,6,6,6,6)+edrixs.get_fock_bin_by_N(8,1,12,10,6,6,6,6),dtype=np.int32)[:,
np.hstack([[0,1,8,9,10,11,2,3,12,13,4,5,14,15,16,17,6,7,18,19],np.arange(20,32)])]

sptab2i, sptab2n1, sptab2n2, sptab4i, sptab4n1, sptab4n2, sptabtr = saveops(basis_i,basis_n1,basis_n2)

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

true_values=dict(tenDq=3.5, theta=49.0, a0=0.,
                 Vdd_sigma=-0.96,Vdd_pi=-0.6,Vdd_delta=0.2,
                 kappa = 0.1, soc = 0.33,
                 soc_c = 1140.332, F0_dd = F0_dd, F2_dd = F2_dd,
                 F4_dd = F4_dd, F0_dp = F0_dp,
                 F2_dp = 1.03*0.9, G1_dp = G1_dp,
                 G3_dp = G3_dp, Gam_c = gamma_c, sigma0 = sigma0)

# parameter bounds
pbounds = {'Vdd_sigma': (-3.0,0.0),
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
       'sigma0' : true_values['sigma0'],
       'sigmas' : sigmas,
       'U_dd_param' : U_dd_param,
       'U_q_param' : U_q_param
}

RIXS_params = dict(sigma0=true_values['sigma0'], sigmas=sigmas, fraction=fraction,
                   geometry=geometry,resolution=resolution,
                   T=temperature, eline=None)


rixs_funV = RIXS_runner(RIXS_params,[sptab2i, sptab2n1, sptab2n2, sptab4i, sptab4n1, sptab4n2, sptabtr])
rixs_funV_E = RIXS_runner(RIXS_params,[sptab2i, sptab2n1, sptab2n2, sptab4i, sptab4n1, sptab4n2, sptabtr],mode='E')
rixs_funV_EL = RIXS_runner(RIXS_params,[sptab2i, sptab2n1, sptab2n2, sptab4i, sptab4n1, sptab4n2, sptabtr],mode='EL')


#funLE
def fun(Vdd_sigma,Vdd_pi, Vdd_delta, F2_dd, F4_dd, F2_dp, G1_dp, G3_dp, soc,soc_c):
    params = {**locals(), **fixed_params}
    return target_L1sum(params,rixs_funV,rixs_ref)+target_L1sum(params,rixs_funV_E,rixs_ref_Emap)+target_L1sum(params,rixs_funV_EL,rixs_ref_Emap_Large)

def funE(Vdd_sigma,Vdd_pi, Vdd_delta, F2_dd, F4_dd, F2_dp, G1_dp, G3_dp, soc,soc_c):
    params = {**locals(), **fixed_params}
    return target_L1sum(params,rixs_funV_E,rixs_ref_Emap)

# will be set by running script
rand_seed = np.zeros(num_runs_local, dtype=np.int32)
# parameters for optimization
threshold_factor = 1.15
max_eval_greedy = 3600

greedy_bounds = {
        'Vdd_sigma': (-3.0,None),
        'Vdd_pi': (-1.5,None),
        'Vdd_delta': (-0.5,None),
        'F2_dd' : [0.0, None],
        'F2_dp' : [0.0, None],
        'F4_dd' : [0.0, None],
        'G1_dp' : [0.0, None],
        'G3_dp' : [0.0, None],
        'tenDq' : [2.5, None],
        'soc' : [0., None],
        'soc_c' : [0., None],
        'Gam_c' : [0.,None],
        }

greedy_filters = [
    { "lhs": "F2_dd",  "op": ">=", "rhs": "F4_dd"  },
]

fixed_params_greedy = {

       'theta' : true_values['theta'],
       'a0' : true_values['a0'],
       'kappa' : true_values['kappa'],
       'sigma0' : true_values['sigma0'],
       'sigmas' : sigmas,
       'U_dd_param' : U_dd_param,
       'U_q_param' : U_q_param
}

def funGreedy(Vdd_sigma, Vdd_pi, Vdd_delta, F2_dd, F4_dd, F2_dp,  G1_dp, G3_dp, tenDq, soc,soc_c, Gam_c):
    params = {**locals(), **fixed_params_greedy}
    return -(target_L1sum(params,rixs_funV,rixs_ref)+target_L1sum(params,rixs_funV_E,rixs_ref_Emap)+target_L1sum(params,rixs_funV_EL,rixs_ref_Emap_Large)
 )

def funEGreedy(Vdd_sigma, Vdd_pi, Vdd_delta, F2_dd, F4_dd, F2_dp,  G1_dp, G3_dp, tenDq, soc,soc_c, Gam_c):
    params = {**locals(), **fixed_params}
    return target_L1sum(params,rixs_funV_E,rixs_ref_Emap)

record=dict(name = data_str,optname=data_opt,  num_runs_local = num_runs_local, num_runs_global = num_runs_global, init_points = init_points,
            init_seed = init_seed, run_ind = 0, num_iters = num_iters,
            true_values = true_values, pbounds = pbounds, method = method,
            tfunc = tfunc, descr = descr, output_dir = output_dir,threshold_factor=threshold_factor,
            max_eval_greedy=max_eval_greedy, greedy_bounds = greedy_bounds, greedy_filters = greedy_filters,rand_seed = rand_seed.tolist())
