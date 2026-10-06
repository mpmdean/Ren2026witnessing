import numpy as np
import edrixs
from edrixs import cb_op, get_trans_oper, tmat_c2r, two_fermion, four_fermion
from edrixs import atom_hsoc, rmat_to_euler, dmat_spinor, transform_utensor, get_umat_slater, get_F0
from edrixs import scattering_mat, dipole_polvec_rixs

from witness_utils import get_tpp, get_tpd, get_tdd, tmat_t2o, tmat_o2t, rmat_t2o, rmat_o2t
import scipy

def saveops_Ce(basis_i,basis_n1,basis_n2):
    sptabtr=[]
    npol=3
    tmp = cb_op(get_trans_oper('dp'), tmat_c2r('d', True), tmat_c2r('p', True))
    for n in range(npol):
        dipole_tmp = np.zeros((32,32), dtype=complex)
        dipole_tmp[0:10, 20:26] = tmp[n]
        sptabtr.append(scipy.sparse.csr_array(two_fermion(dipole_tmp, basis_n1, basis_i)))
    for n in range(npol):
        dipole_tmp = np.zeros((32,32), dtype=complex)
        dipole_tmp[10:20, 26:32] = tmp[n]
        sptabtr.append(scipy.sparse.csr_array(two_fermion(dipole_tmp, basis_n2, basis_i)))

    tdd_sigma, tdd_pi, tdd_delta = get_tdd(0., 0., 1.)
    # transform to local axis
    tmat = np.zeros((10,10), dtype=complex)

    tmat[:5,:5] = tmat_t2o(1)
    tmat[5:,5:] = tmat_t2o(2)

    # transform to local axis
    dmat = np.zeros((2, 2, 2), dtype=complex)
    ang1, ang2, ang3 = rmat_to_euler(rmat_o2t(1))
    dmat[0] = dmat_spinor(ang1, ang2, ang3)
    ang1, ang2, ang3 = rmat_to_euler(rmat_o2t(2))
    dmat[1] = dmat_spinor(ang1, ang2, ang3)

    t_spinor = np.zeros((20,20), dtype=complex)
    for nTM in range(2):
        for norb in range(5):
            t_spinor[(nTM*10+norb*2):(nTM*10+norb*2+2),(nTM*10+norb*2):(nTM*10+norb*2+2)] = dmat[nTM]

    dmatS = np.zeros((2, 2, 2), dtype=complex)
    dmatS[0] = dmat_spinor(0, 0, 0)
    dmatS[1] = 1j*dmat_spinor(0, 0, np.pi)
    t_spinorS = np.zeros((20,20), dtype=complex)
    for nTM in range(2):
        for norb in range(5):
            t_spinorS[(nTM*10+norb*2):(nTM*10+norb*2+2),(nTM*10+norb*2):(nTM*10+norb*2+2)] = dmatS[nTM]

    sptab2i=[]
    sptab2n1=[]
    sptab2n2=[]

    emat0 = np.zeros((10,10), dtype=complex)
    emat_tmp = np.diag([(-18.)/15., (12.)/15., (12.)/15., (-3.)/15., (-3.)/15.])
    emat0[:5,:5] += emat_tmp
    emat_tmp = np.diag([(-18.)/15., (12.)/15., (12.)/15., (-3.)/15., (-3.)/15.])
    emat0[5:,5:] += emat_tmp

    emat0 = cb_op(emat0, tmat)
    emat=np.zeros((32,32), dtype=complex)
    #tenDq*a4*sptab2[0]
    emat[:20:2,:20:2] += emat0
    emat[1:20:2,1:20:2] += emat0
    emat[:20,:20] = cb_op(emat[:20,:20], t_spinorS)
    sptab2i.append(scipy.sparse.csr_array(two_fermion(emat,basis_i)))
    sptab2n1.append(scipy.sparse.csr_array(two_fermion(emat,basis_n1)))
    sptab2n2.append(scipy.sparse.csr_array(two_fermion(emat,basis_n2)))

    emat0 = np.zeros((10,10), dtype=complex)
    emat_tmp = np.diag([(10.)/15., (5.)/15., (5.)/15., (-10.)/15., (-10.)/15.])
    emat0[:5,:5] += emat_tmp
    emat_tmp = np.diag([(10.)/15., (5.)/15., (5.)/15., (-10.)/15., (-10.)/15.])
    emat0[5:,5:] += emat_tmp

    emat0 = cb_op(emat0, tmat)
    emat=np.zeros((32,32), dtype=complex)
    #tenDq*a2*sptab2[1]
    emat[:20:2,:20:2] += emat0
    emat[1:20:2,1:20:2] += emat0
    emat[:20,:20] = cb_op(emat[:20,:20], t_spinorS)
    sptab2i.append(scipy.sparse.csr_array(two_fermion(emat,basis_i)))
    sptab2n1.append(scipy.sparse.csr_array(two_fermion(emat,basis_n1)))
    sptab2n2.append(scipy.sparse.csr_array(two_fermion(emat,basis_n2)))

    emat0 = np.zeros((10,10), dtype=complex)
    emat_tmp = np.zeros((5,5), dtype=complex)
    emat_tmp[1,3] = emat_tmp[3,1] = -1./2
    emat_tmp[2,4] = emat_tmp[4,2] = 1./2
    emat0[:5,:5] += emat_tmp
    emat_tmp[1,3] = emat_tmp[3,1] = 1./2
    emat_tmp[2,4] = emat_tmp[4,2] = -1./2
    emat0[5:,5:] += emat_tmp

    emat0 = cb_op(emat0, tmat)
    emat=np.zeros((32,32), dtype=complex)
    #tenDq*b*sptab2[2]
    emat[:20:2,:20:2] += emat0
    emat[1:20:2,1:20:2] += emat0
    emat[:20,:20] = cb_op(emat[:20,:20], t_spinorS)
    sptab2i.append(scipy.sparse.csr_array(two_fermion(emat,basis_i)))
    sptab2n1.append(scipy.sparse.csr_array(two_fermion(emat,basis_n1)))
    sptab2n2.append(scipy.sparse.csr_array(two_fermion(emat,basis_n2)))

    emat0 = np.zeros((10,10), dtype=complex)
    emat0[5:,:5] = tdd_sigma
    emat0[:5,5:] = np.transpose(tdd_sigma)
    emat0 = cb_op(emat0, tmat)
    emat=np.zeros((32,32), dtype=complex)
    #Vdd_sigma*sptab2[3]
    emat[:20:2,:20:2] += emat0
    emat[1:20:2,1:20:2] += emat0
    emat[:20,:20] = cb_op(emat[:20,:20], t_spinorS)
    sptab2i.append(scipy.sparse.csr_array(two_fermion(emat,basis_i)))
    sptab2n1.append(scipy.sparse.csr_array(two_fermion(emat,basis_n1)))
    sptab2n2.append(scipy.sparse.csr_array(two_fermion(emat,basis_n2)))

    emat0 = np.zeros((10,10), dtype=complex)
    emat0[5:,:5] = tdd_pi
    emat0[:5,5:] = np.transpose(tdd_pi)
    emat0 = cb_op(emat0, tmat)
    emat=np.zeros((32,32), dtype=complex)
    #Vdd_pi*sptab2[4]
    emat[:20:2,:20:2] += emat0
    emat[1:20:2,1:20:2] += emat0
    emat[:20,:20] = cb_op(emat[:20,:20], t_spinorS)
    sptab2i.append(scipy.sparse.csr_array(two_fermion(emat,basis_i)))
    sptab2n1.append(scipy.sparse.csr_array(two_fermion(emat,basis_n1)))
    sptab2n2.append(scipy.sparse.csr_array(two_fermion(emat,basis_n2)))

    emat0 = np.zeros((10,10), dtype=complex)
    emat0[5:,:5] = tdd_delta
    emat0[:5,5:] = np.transpose(tdd_delta)
    emat0 = cb_op(emat0, tmat)
    emat=np.zeros((32,32), dtype=complex)
    #Vdd_delta*sptab2[5]
    emat[:20:2,:20:2] += emat0
    emat[1:20:2,1:20:2] += emat0
    emat[:20,:20] = cb_op(emat[:20,:20], t_spinorS)
    sptab2i.append(scipy.sparse.csr_array(two_fermion(emat,basis_i)))
    sptab2n1.append(scipy.sparse.csr_array(two_fermion(emat,basis_n1)))
    sptab2n2.append(scipy.sparse.csr_array(two_fermion(emat,basis_n2)))

    # spin orbital coupling
    soc_mat =np.zeros((20,20), dtype=complex)
    soc_mat[:10,:10] = cb_op(atom_hsoc('d', 1.0), tmat_c2r('d', True))
    soc_mat[10:,10:] = cb_op(atom_hsoc('d', 1.0), tmat_c2r('d', True))

    soc_emat = cb_op(soc_mat, t_spinor)
    emat=np.zeros((32,32), dtype=complex)
    #soc*sptab2[6]
    emat[:20,:20] += soc_emat
    emat[:20,:20] = cb_op(emat[:20,:20], t_spinorS)
    sptab2i.append(scipy.sparse.csr_array(two_fermion(emat,basis_i)))
    sptab2n1.append(scipy.sparse.csr_array(two_fermion(emat,basis_n1)))
    sptab2n2.append(scipy.sparse.csr_array(two_fermion(emat,basis_n2)))

    # soc for core
    soc_mat =np.zeros((12,12), dtype=complex)
    soc_mat[:6,:6] = cb_op(atom_hsoc('p', 1.0), tmat_c2r('p', True))
    soc_mat[6:,6:] = cb_op(atom_hsoc('p', 1.0), tmat_c2r('p', True))

    t_spinor = np.zeros((12,12), dtype=complex)
    for nTM in range(2):
        for norb in range(3):
            t_spinor[(nTM*6+norb*2):(nTM*6+norb*2+2),(nTM*6+norb*2):(nTM*6+norb*2+2)] = dmat[nTM]
    t_spinorS = np.zeros((12,12), dtype=complex)
    for nTM in range(2):
        for norb in range(3):
            t_spinorS[(nTM*6+norb*2):(nTM*6+norb*2+2),(nTM*6+norb*2):(nTM*6+norb*2+2)] = dmatS[nTM]

    soc_mat = cb_op(soc_mat, t_spinor)
    emat=np.zeros((32,32), dtype=complex)
    #soc_c*sptab2[7]
    emat[20:,20:] += soc_mat
    emat[20:,20:] = cb_op(emat[20:,20:], t_spinorS)
    sptab2i.append(scipy.sparse.csr_array(two_fermion(emat,basis_i)))
    sptab2n1.append(scipy.sparse.csr_array(two_fermion(emat,basis_n1)))
    sptab2n2.append(scipy.sparse.csr_array(two_fermion(emat,basis_n2)))

    tmat = np.zeros((16,16), dtype=complex)
    tmat[:10,:10] = tmat_c2r('d', True)
    tmat[10:,10:] = tmat_c2r('p', True)

    sptab4i=[]
    sptab4n1=[]
    sptab4n2=[]

    umat=np.zeros((32,32,32,32), dtype=complex)
    umat_tmp = transform_utensor(get_umat_slater('dp', 1.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0, 0), tmat)
    #umat_tmp=get_umat_slater('dp', 1.0, 0.0, 0.0,
    #                                            0.0, 0.0, 0.0, 0.0, 0, 0)
    index = np.hstack([np.arange(10), np.arange(20,26)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    index = np.hstack([np.arange(10,20), np.arange(26,32)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    sptab4i.append(scipy.sparse.csr_array(four_fermion(umat,basis_i)))
    sptab4n1.append(scipy.sparse.csr_array(four_fermion(umat,basis_n1)))
    sptab4n2.append(scipy.sparse.csr_array(four_fermion(umat,basis_n2)))

    umat=np.zeros((32,32,32,32), dtype=complex)
    umat_tmp = transform_utensor(get_umat_slater('dp', 0.0, 1.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0, 0), tmat)
    #umat_tmp=get_umat_slater('dp', 0.0, 1.0, 0.0,
    #                                           0.0, 0.0, 0.0, 0.0, 0, 0)
    index = np.hstack([np.arange(10), np.arange(20,26)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    index = np.hstack([np.arange(10,20), np.arange(26,32)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    sptab4i.append(scipy.sparse.csr_array(four_fermion(umat,basis_i)))
    sptab4n1.append(scipy.sparse.csr_array(four_fermion(umat,basis_n1)))
    sptab4n2.append(scipy.sparse.csr_array(four_fermion(umat,basis_n2)))

    umat=np.zeros((32,32,32,32), dtype=complex)
    umat_tmp = transform_utensor(get_umat_slater('dp', 0.0, 0.0, 1.0,
                                            0.0, 0.0, 0.0, 0.0, 0, 0), tmat)
    #umat_tmp=get_umat_slater('dp', 0.0, 0.0, 1.0,
    #                                            0.0, 0.0, 0.0, 0.0, 0, 0)
    index = np.hstack([np.arange(10), np.arange(20,26)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    index = np.hstack([np.arange(10,20), np.arange(26,32)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    sptab4i.append(scipy.sparse.csr_array(four_fermion(umat,basis_i)))
    sptab4n1.append(scipy.sparse.csr_array(four_fermion(umat,basis_n1)))
    sptab4n2.append(scipy.sparse.csr_array(four_fermion(umat,basis_n2)))

    umat=np.zeros((32,32,32,32), dtype=complex)
    umat_tmp = transform_utensor(get_umat_slater('dp', 0.0, 0.0, 0.0,
                                            1.0, 0.0, 0.0, 0.0, 0, 0), tmat)
    #umat_tmp=get_umat_slater('dp', 0.0, 0.0, 0.0,
    #                                           1.0, 0.0, 0.0, 0.0, 0, 0)
    index = np.hstack([np.arange(10), np.arange(20,26)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    index = np.hstack([np.arange(10,20), np.arange(26,32)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    sptab4i.append(scipy.sparse.csr_array(four_fermion(umat,basis_i)))
    sptab4n1.append(scipy.sparse.csr_array(four_fermion(umat,basis_n1)))
    sptab4n2.append(scipy.sparse.csr_array(four_fermion(umat,basis_n2)))

    umat=np.zeros((32,32,32,32), dtype=complex)
    umat_tmp = transform_utensor(get_umat_slater('dp', 0.0, 0.0, 0.0,
                                            0.0, 1.0, 0.0, 0.0, 0, 0), tmat)
    #umat_tmp=get_umat_slater('dp', 0.0, 0.0, 0.0,
    #                                            0.0, 1.0, 0.0, 0.0, 0, 0)
    index = np.hstack([np.arange(10), np.arange(20,26)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    index = np.hstack([np.arange(10,20), np.arange(26,32)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    sptab4i.append(scipy.sparse.csr_array(four_fermion(umat,basis_i)))
    sptab4n1.append(scipy.sparse.csr_array(four_fermion(umat,basis_n1)))
    sptab4n2.append(scipy.sparse.csr_array(four_fermion(umat,basis_n2)))

    umat=np.zeros((32,32,32,32), dtype=complex)
    umat_tmp = transform_utensor(get_umat_slater('dp', 0.0, 0.0, 0.0,
                                            0.0, 0.0, 1.0, 0.0, 0, 0), tmat)
    #umat_tmp=get_umat_slater('dp', 0.0, 0.0, 0.0,
    #                                            0.0, 0.0, 1.0, 0.0, 0, 0)
    index = np.hstack([np.arange(10), np.arange(20,26)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    index = np.hstack([np.arange(10,20), np.arange(26,32)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    sptab4i.append(scipy.sparse.csr_array(four_fermion(umat,basis_i)))
    sptab4n1.append(scipy.sparse.csr_array(four_fermion(umat,basis_n1)))
    sptab4n2.append(scipy.sparse.csr_array(four_fermion(umat,basis_n2)))

    umat=np.zeros((32,32,32,32), dtype=complex)
    umat_tmp = transform_utensor(get_umat_slater('dp', 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 1.0, 0, 0), tmat)
    #umat_tmp=get_umat_slater('dp', 0.0, 0.0, 0.0,
    #                                            0.0, 0.0, 0.0, 1.0, 0, 0)
    index = np.hstack([np.arange(10), np.arange(20,26)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    index = np.hstack([np.arange(10,20), np.arange(26,32)])
    for n1 in range(16):
        for n2 in range(16):
            for n3 in range(16):
                for n4 in range(16):
                    umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
    sptab4i.append(scipy.sparse.csr_array(four_fermion(umat,basis_i)))
    sptab4n1.append(scipy.sparse.csr_array(four_fermion(umat,basis_n1)))
    sptab4n2.append(scipy.sparse.csr_array(four_fermion(umat,basis_n2)))

    return sptab2i, sptab2n1, sptab2n2, sptab4i, sptab4n1, sptab4n2, sptabtr

def calc_RIXS_dimerCeMod(evals_i,evals_n_all,T_abs_all,HKL=None, Gam_c=1., sigma=0.1, fraction=0.5,
                      omega=None, eloss=None, eline=None,
                      phi=0., thin=15., tth=150., pol=[(0, 0), (0, np.pi / 2.0)],
                      T=10., tol=1e-5, ngs=None):
    """
    Calculate the RIXS spectra for Ba3CeIr2O9.
    Eigen values and transition operators will be loaded from the folder.

    Parameters
    ----------
    HKL : list,
        Q points in reciprocal lattice units.
    Gam_c : float,
        Inverse core-hole lifetime in unit of eV (default: 1.).
    sigma : float,
        Sigma of PseudoVoigt profile (default: 0.1).
    fraction : float,
        Fraction of PseudoVoigt profile (default: 0.5).
    omega : 1d array or float,
        Incident energy. If None, determined by evals (default: None).
    eloss : 1d array,
        Energy loss. If None, determined by evals (default: None).
    phi : float,
        Azimuthal angle in unit of degree (default: 0).
    thin : 1d array or float,
        Incident angle w.r.t. sample surface in unit of degree (default: 15).
        It must have the same length as HKL.
    tth : float,
        Scattering angle in unit of degree (default: 150).
    pol : list of 2-element turple,
        X-ray polarization, (incident, emission), w.r.t the scattering plane.
    T : float,
        Temperature in unit of Kelvin (default: 10).
    tol : float,
        Tolerance of bose factor to include the initial states (default: 1e-3).
    ngs : int,
        Number of initial states. If None it will be determined based on tol (default: None).


    Returns
    -------
    rixs[pol,omega,thin,eloss] : 4d array.
    """


    # Determine the ground state occupation
    GS_energy = np.min(evals_i) ## ground state energy
    beta = 1.0 / 8.6173303E-5 / T
    if ngs is None:
        choose = np.exp(-beta * (evals_i - GS_energy)) >= tol
        gs = np.arange(len(evals_i))[choose]
    else:
        gs = np.arange(ngs)
    prob = np.exp(-beta * (evals_i[gs] - GS_energy)) / np.sum(np.exp(-beta * (evals_i[gs] - GS_energy)))

    # Determine omega
    if omega is None:
        omega_min = np.min(evals_n_all[0]) - GS_energy - 10.
        omega_max = np.max(evals_n_all[0]) - GS_energy + 10.
        omega_step = Gam_c / 10
        omega = np.arange(omega_min, omega_max, omega_step)

    # Determine eloss
    if eloss is None:
        eloss_min = -1.
        eloss_max = np.max(evals_i) - GS_energy + 1.
        eloss_step = sigma / 10
        eloss = np.arange(eloss_min, eloss_max, eloss_step)

    # Reformulate the params
    if isinstance(omega, list):
        omega = np.array(omega)
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega])
    if isinstance(thin, list):
        thin = np.array(thin)
    elif not isinstance(thin, np.ndarray):
        thin = np.array([thin])
    if not isinstance(pol, list):
        pol = [pol]
    if HKL is None:
        HKL = [[0., 0., 0.] for _ in range(len(thin))]
    elif len(np.array(HKL).shape)==1:
        HKL = [HKL] * len(thin)

    # Extra params
    sigma_g = sigma / np.sqrt(2.*np.log(2))
    pos = [[2.95175, 1.70419, 2.41068],
           [2.95175, 1.70419, 4.94682],
           [0.00000, 3.40839, 12.30432],
           [0.00000, 3.40839, 9.76818]]
    Umat = np.array([[ 1.06431529e+00,  6.14482718e-01, -1.30341031e-16],
                     [ 0.00000000e+00,  1.22896544e+00, -1.30341031e-16],
                     [ 0.00000000e+00, -0.00000000e+00,  4.26991866e-01]])

    # Calculate RIXS
    F_fis = np.zeros((2, len(omega), 3, 3, len(evals_i), len(gs)), dtype=complex)
    for nomega, om in enumerate(omega):
        for nnTM, (evals_n, T_abs) in enumerate(zip(evals_n_all, T_abs_all)):
            T_emi = np.stack([np.conj(np.transpose(T0)) for T0 in T_abs])
            F_fis[nnTM,nomega,:,:,:,:] += scattering_mat(evals_i[:], evals_n[:], T_abs[:, :, gs], T_emi, om, Gam_c)

    rixs = np.zeros((len(pol), len(omega), len(thin), len(eloss)), dtype=float)
    for nomega, om in enumerate(omega):
        for npol, (alpha, beta) in enumerate(pol):
            for nth, th in enumerate(thin):
                ei0, ef0 = dipole_polvec_rixs(th/180.*np.pi, (-th+tth)/180.*np.pi, phi/180.*np.pi, alpha, beta)
                Q = np.dot(HKL[nth], Umat)
                F_mag = np.zeros((len(evals_i), len(gs)), dtype=complex)
                for nnTM in range(2):
                    ei = np.dot(ei0, rmat_t2o(nnTM+1))
                    ef = np.dot(ef0, rmat_t2o(nnTM+1))
                    phase = np.exp(-1j * np.dot(Q, pos[nnTM]))
                    for m in range(3):
                        for n in range(3):
                            F_mag[:, :] += ef[m] * F_fis[nnTM, nomega, m, n] * phase * ei[n]
                for nngs, gs_id in enumerate(gs):
                    for n in range(len(evals_i)):
                        if eline is None or evals_i[n]-evals_i[gs_id]>=eline:
                            Gaussian = (1 - fraction) / sigma_g / np.sqrt(2. * np.pi) * np.exp(-(eloss - (evals_i[n] - evals_i[gs_id]))**2 / 2. / sigma_g**2)
                            Lorentzian = fraction / np.pi * sigma / ((eloss - (evals_i[n] - evals_i[gs_id]))**2 + sigma**2)
                            rixs[npol,nomega,nth,:] += prob[nngs] * np.abs(F_mag[n, nngs])**2 * (Gaussian + Lorentzian)
    for nomega, om in enumerate(omega):
        for npol, (alpha, beta) in enumerate(pol):
            for nth, th in enumerate(thin):
                ei0, ef0 = dipole_polvec_rixs(th/180.*np.pi, (-th+tth)/180.*np.pi, phi/180.*np.pi, alpha, beta)
                Q = np.dot(HKL[nth], Umat)
                F_mag = np.zeros((len(evals_i), len(gs)), dtype=complex)
                for nnTM in range(2):
                    ei = np.dot(np.dot(ei0, np.diag([1.,-1.,-1.])), rmat_t2o(nnTM+1))
                    ef = np.dot(np.dot(ef0, np.diag([1.,-1.,-1.])), rmat_t2o(nnTM+1))
                    phase = np.exp(-1j * np.dot(Q, pos[nnTM+2]))
                    for m in range(3):
                        for n in range(3):
                            F_mag[:, :] += ef[m] * F_fis[nnTM, nomega, m, n] * phase * ei[n]
                for nngs, gs_id in enumerate(gs):
                    for n in range(len(evals_i)):
                        if eline is None or evals_i[n]-evals_i[gs_id]>=eline:
                            Gaussian = (1 - fraction) / sigma_g / np.sqrt(2. * np.pi) * np.exp(-(eloss - (evals_i[n] - evals_i[gs_id]))**2 / 2. / sigma_g**2)
                            Lorentzian = fraction / np.pi * sigma / ((eloss - (evals_i[n] - evals_i[gs_id]))**2 + sigma**2)
                            rixs[npol,nomega,nth,:] += prob[nngs] * np.abs(F_mag[n, nngs])**2 * (Gaussian + Lorentzian)

    return rixs

class RIXS_runner:
#    def __init__(self, RIXS_params,geometry,sptabs = None):
    def __init__(self, RIXS_params,sptabs = None,mode='L'):

        self.RIXS_params = RIXS_params
        self.sptab2i  = sptabs[0]
        self.sptab2n1  = sptabs[1]
        self.sptab2n2  = sptabs[2]
        self.sptab4i  = sptabs[3]
        self.sptab4n1  = sptabs[4]
        self.sptab4n2  = sptabs[5]
        self.sptabtr = sptabs[6]
        self.mode=mode
        #self.tths = geometry['tths']
        #self.thins = geometry['thins']
        #self.Ls = geometry['Ls']
        #self.Qs = geometry['Qs']
        #self.HKL = geometry['HKL']



    def __call__(self, params, ed_only=False):
        F0_dd = params['U_dd_param'] - 4.0 / 49.0 * params['F2_dd'] - 4.0 / 49.0 * params['F4_dd']
        F0_dp = get_F0('dp', params['G1_dp'], params['G3_dp']) + params['U_q_param']

        Hmat_isp=scipy.sparse.csr_array(self.sptab4i[0].shape,dtype=np.complex128)
        Hmat_n1sp=scipy.sparse.csr_array(self.sptab4n1[0].shape,dtype=np.complex128)
        Hmat_n2sp=scipy.sparse.csr_array(self.sptab4n2[0].shape,dtype=np.complex128)

        a2 = 27./35.*params['kappa']*(3.*np.cos(np.deg2rad(params['theta']))**2 - 1.) - params['a0']
        a4 = -1.5 * (2.5*np.cos(np.deg2rad(params['theta']))**4-15./7.*np.cos(np.deg2rad(params['theta']))**2+3./14.)
        b = 3. * np.sin(np.deg2rad(params['theta']))**3 * np.cos(np.deg2rad(params['theta']))

        Hmat_isp = params['tenDq']*a4*self.sptab2i[0] + params['tenDq']*a2*self.sptab2i[1] + params['tenDq']*b*self.sptab2i[2] + params['Vdd_sigma']*self.sptab2i[3];
        Hmat_isp += params['Vdd_pi']*self.sptab2i[4] + params['Vdd_delta']*self.sptab2i[5] + params['soc']*self.sptab2i[6] + params['soc_c']*self.sptab2i[7];
        Hmat_isp += F0_dd*self.sptab4i[0] + params['F2_dd']*self.sptab4i[1] + params['F4_dd']*self.sptab4i[2];
        Hmat_isp += F0_dp*self.sptab4i[3] + params['F2_dp']*self.sptab4i[4] + params['G1_dp']*self.sptab4i[5] + params['G3_dp']*self.sptab4i[6];

        Hmat_n1sp = params['tenDq']*a4*self.sptab2n1[0] + params['tenDq']*a2*self.sptab2n1[1] + params['tenDq']*b*self.sptab2n1[2] + params['Vdd_sigma']*self.sptab2n1[3];
        Hmat_n1sp += params['Vdd_pi']*self.sptab2n1[4] + params['Vdd_delta']*self.sptab2n1[5] + params['soc']*self.sptab2n1[6] + params['soc_c']*self.sptab2n1[7];
        Hmat_n1sp += F0_dd*self.sptab4n1[0] + params['F2_dd']*self.sptab4n1[1] + params['F4_dd']*self.sptab4n1[2];
        Hmat_n1sp += F0_dp*self.sptab4n1[3] + params['F2_dp']*self.sptab4n1[4] + params['G1_dp']*self.sptab4n1[5] + params['G3_dp']*self.sptab4n1[6];

        Hmat_n2sp = params['tenDq']*a4*self.sptab2n2[0] + params['tenDq']*a2*self.sptab2n2[1] + params['tenDq']*b*self.sptab2n2[2] + params['Vdd_sigma']*self.sptab2n2[3];
        Hmat_n2sp += params['Vdd_pi']*self.sptab2n2[4] + params['Vdd_delta']*self.sptab2n2[5] + params['soc']*self.sptab2n2[6] + params['soc_c']*self.sptab2n2[7];
        Hmat_n2sp += F0_dd*self.sptab4n2[0] + params['F2_dd']*self.sptab4n2[1] + params['F4_dd']*self.sptab4n2[2];
        Hmat_n2sp += F0_dp*self.sptab4n2[3] + params['F2_dp']*self.sptab4n2[4] + params['G1_dp']*self.sptab4n2[5] + params['G3_dp']*self.sptab4n2[6];

        hmat_i = Hmat_isp.toarray()
        hmat_n1 = Hmat_n1sp.toarray()
        hmat_n2 = Hmat_n2sp.toarray()

        eval_i, evec_i = scipy.linalg.eigh(hmat_i);

        if ed_only:
            return eval_i

        eval_n1, evec_n1 = scipy.linalg.eigh(hmat_n1);
        eval_n2, evec_n2 = scipy.linalg.eigh(hmat_n2);

        evecs_n=[evec_n1,evec_n2]
        T_abs_all  = []
        for nnsite in range(2):
            T_abs = []
            for xyz in range(3):
                T_abs.append(self.sptabtr[nnsite*3+xyz].transpose().dot(np.conj(evecs_n[nnsite])).transpose().dot(evec_i))
            T_abs_all.append(np.stack(T_abs))


        rixs_all2 = []
        if self.mode == 'L':
            tths = self.RIXS_params['geometry']['tths']
            thins = self.RIXS_params['geometry']['thins']
            Ls = self.RIXS_params['geometry']['Ls']
            Qs = self.RIXS_params['geometry']['Qs']
            pol = self.RIXS_params['geometry']['pol']
            eloss=self.RIXS_params['resolution']['eloss']
            HKL = self.RIXS_params['geometry']['HKL']

            RIXSpar = dict(Gam_c=params['Gam_c'], sigma=self.RIXS_params['sigma'], fraction=self.RIXS_params['fraction'],
            omega=self.RIXS_params['geometry']['omega'], eloss=eloss,
            pol=pol, T=self.RIXS_params['T'], eline=None)

            for tth, thin, L in zip(tths, thins, Ls):
                rixs_all2.append(calc_RIXS_dimerCeMod(eval_i,[eval_n1,eval_n2],T_abs_all,tth=tth, thin=thin, HKL=[HKL[0],HKL[1],L], phi=10.0, **RIXSpar))
            rixs_Mmap2 = dict(momenta=Qs, pol=pol, eloss=eloss, rixs=np.squeeze(rixs_all2))
            Ldep_calc2=rixs_Mmap2
            Ldep_calc2['intensity'] = Ldep_calc2['rixs'].T

            return Ldep_calc2['intensity'][:,0]+Ldep_calc2['intensity'][:,1]

        elif self.mode == 'E' or self.mode == 'EL':
            if self.mode == 'E':
                eloss=self.RIXS_params['resolution']['eloss_Emap']
            else:
                eloss=self.RIXS_params['resolution']['eloss_Emap_large']
            tth = self.RIXS_params['geometry']['tth_Emap']
            thin = self.RIXS_params['geometry']['thin_Emap']
            L = self.RIXS_params['geometry']['L_Emap']
            pol = self.RIXS_params['geometry']['pol']

            HKL = self.RIXS_params['geometry']['HKL']

            print(self.RIXS_params['geometry']['omega_Emap'])

            RIXSpar = dict(Gam_c=params['Gam_c'], sigma=self.RIXS_params['sigma'], fraction=self.RIXS_params['fraction'],
            omega=self.RIXS_params['geometry']['omega_Emap'], eloss=eloss,
            pol=pol, T=self.RIXS_params['T'], eline=None)


            rixs_all2 = calc_RIXS_dimerCeMod(eval_i,[eval_n1,eval_n2],T_abs_all,tth=tth, thin=thin, HKL=[HKL[0],HKL[1],L], phi=10.0, **RIXSpar)

            rixs_Emap2 = dict(omega=self.RIXS_params['geometry']['omega_Emap'], pol=pol, eloss=eloss, rixs=np.squeeze(rixs_all2))
            Edep_calc2=rixs_Emap2
            Edep_calc2['intensity'] = Edep_calc2['rixs'].T

            #return Edep_calc2['intensity'][:,0]+Edep_calc2['intensity'][:,1]
            return Edep_calc2['intensity'][:,:,0]+ Edep_calc2['intensity'][:,:,1]
        return 0
