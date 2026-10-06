#!/usr/bin/env python3
"""Generate the Ta momentum-dependence files used by the plotting notebook.

Run from the directory containing config_Ta_Eg.py, its input data and
Ba3TaIr2O9_collectedV.txt. Project modules must be importable on the Python path.
Dependencies: NumPy, SciPy, EDRIXS, tqdm, and the project modules
optEDRIXS, utils_Ta, witness, and witness_utils.

Unpolarized results retain original fitted rows 0 through 23. Polarized
results use original fitted row 9, stored as row 0 in the polarized output.
Each bound uses 100 random L-BFGS-B restarts per charge sector, with the
original objectives, gradients, bounds, tolerances and temperature.
"""

import argparse
from pathlib import Path
import numpy as np
from edrixs import scattering_mat
from scipy.optimize import minimize
from tqdm import tqdm
from optEDRIXS import import_config
from witness import calc_RIXS_dimerTa, conjugate_RIXS_dimerTa, dimer_witness
from witness_utils import SparseBinary, get_fock_bin_by_N_S, rmat_t2o


def save_charge_sector_embeddings(folder):
    """Save embeddings of the ordered (6, 5) and (5, 6) charge-sector bases."""
    coordinates = np.array(
        [0, 1, 6, 7, 10, 11, 16, 17, 2, 3, 4, 5, 8, 9, 12, 13, 14, 15, 18, 19]
    )
    full_basis = SparseBinary(
        np.vstack(
            [get_fock_bin_by_N_S(8, eg, 12, 11 - eg, dispinfo=False) for eg in range(2)]
        ),
        shape=[None, 20],
    )
    full_basis.change_coordinates(coordinates, shape=[None, 20])
    full_states = full_basis.toarray()
    for left, right in [(6, 5), (5, 6)]:
        sector_basis = SparseBinary(
            np.vstack(get_fock_bin_by_N_S(8, 0, 6, left, 6, right, dispinfo=False)),
            shape=[None, 20],
        )
        sector_basis.change_coordinates(coordinates, shape=[None, 20])
        embeddings = np.all(
            sector_basis.toarray()[:, None, :] == full_states[None, :, :], axis=2
        ).astype(float)
        np.save(str(folder) + f"/states_{left}{right}.npy", embeddings)


def polarization_vectors(tth, th, phi, chi):
    """Return incident and outgoing polarization vectors from the motor angles."""
    UBmatrix = np.array(
        [
            [-0.0935575, 1.04121448, -0.01640006],
            [1.25487514, 0.70574976, -0.02770928],
            [-0.07643596, -0.08424219, -0.43483902],
        ]
    )
    Bmatrix = np.array(
        [
            [1.09177851, 0.630338615, -1.33704305e-16],
            [0.0, 1.26067723, -1.33704305e-16],
            [0.0, -0.0, 0.436029515],
        ]
    )
    Umatrix = np.dot(UBmatrix, np.linalg.inv(Bmatrix))
    omega = th - tth / 2.0
    mat_PHI = np.vstack(
        [
            [np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi)), 0.0],
            [-np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi)), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mat_CHI = np.vstack(
        [
            [np.cos(np.deg2rad(chi)), 0.0, np.sin(np.deg2rad(chi))],
            [0.0, 1.0, 0.0],
            [-np.sin(np.deg2rad(chi)), 0.0, np.cos(np.deg2rad(chi))],
        ]
    )
    mat_OMEGA = np.vstack(
        [
            [np.cos(np.deg2rad(omega)), np.sin(np.deg2rad(omega)), 0.0],
            [-np.sin(np.deg2rad(omega)), np.cos(np.deg2rad(omega)), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tmat_Q2TH = np.dot(mat_OMEGA, np.dot(mat_CHI, np.dot(mat_PHI, Umatrix)))
    tmat_TH2Q = np.linalg.inv(tmat_Q2TH)
    ei_TH = [np.cos(np.deg2rad(tth / 2)), np.sin(np.deg2rad(tth / 2)), 0.0]
    ef1_TH = [np.cos(np.deg2rad(tth / 2)), -np.sin(np.deg2rad(tth / 2)), 0.0]
    ef2_TH = [0.0, 0.0, 1.0]
    ei = np.dot(tmat_TH2Q, ei_TH)
    ei = ei / np.linalg.norm(ei)
    ef1 = np.dot(tmat_TH2Q, ef1_TH)
    ef1 = ef1 / np.linalg.norm(ef1)
    ef2 = np.dot(tmat_TH2Q, ef2_TH)
    ef2 = ef2 / np.linalg.norm(ef2)
    return (ei, ef1, ef2)


def build_scattering_operator(
    HKL=(0.0, 0.0, 0.0),
    Gam_c=0.6,
    omega=0,
    chi=90.0,
    phi=0.0,
    th=15.0,
    tth=150.0,
    polind=0,
    folder="",
    ext="",
):
    """Save the RIXS scattering operator in the ED eigenbasis."""
    evals_i = np.load(folder + "/evals_i.npy")
    evals_n_all = [np.load(folder + "/evals_n1.npy"), np.load(folder + "/evals_n2.npy")]
    T_abs_all = dict(np.load(folder + "/T_abs.npz"))
    T_abs_all = [T_abs_all["T_abs1"], T_abs_all["T_abs2"]]
    pos = [
        [8.76833, 1.68149, 2.29793],
        [8.71867, 1.68149, 4.99793],
        [5.80417, 3.36651, 12.11207],
        [5.85383, 3.36651, 9.41207],
    ]
    Bmat = np.array(
        [
            [1.09177851, 0.630338615, -1.33704305e-16],
            [0.0, 1.26067723, -1.33704305e-16],
            [0.0, -0.0, 0.436029515],
        ]
    )
    F_fis = np.zeros((2, 3, 3, len(evals_i), len(evals_i)), dtype=complex)
    for nnTM, (evals_n, T_abs) in enumerate(zip(evals_n_all, T_abs_all)):
        T_emi = np.stack([np.conj(np.transpose(T0)) for T0 in T_abs])
        F_fis[nnTM, :, :, :, :] += scattering_mat(
            evals_i[:], evals_n[:], T_abs[:, :, :], T_emi, omega, Gam_c
        )
    ei0, ef1, ef2 = polarization_vectors(tth, th, phi, chi)
    polvec = [ef1, ef2]
    ef0 = polvec[polind]
    Q = np.dot(Bmat, HKL)
    F_mag = np.zeros((len(evals_i), len(evals_i)), dtype=complex)
    str1 = ""
    if ext[0] == "M":
        str1 = str(HKL[2])
    elif ext[0] == "E":
        str1 = str(omega)
    for nnTM in range(2):
        ei = np.dot(ei0, rmat_t2o(nnTM + 1))
        ef = np.dot(ef0, rmat_t2o(nnTM + 1))
        phase = np.exp(-1j * np.dot(Q, pos[nnTM]))
        for m in range(3):
            for n in range(3):
                F_mag[:, :] += ef[m] * F_fis[nnTM, m, n] * phase * ei[n]
    np.savez(folder + "/optimize/FQ_Fmag_" + ext + "_" + str1 + ".npz", F_mag)


def random_initial_state(n, rng=None, eps=1e-12):
    """Draw local-state amplitudes and phases for one random restart."""
    if rng is None:
        rng = np.random.default_rng()
    r0 = rng.random(n)
    r0 = np.maximum(r0, eps)
    theta0 = rng.uniform(0.0, 2 * np.pi, size=n)
    return np.concatenate([r0, theta0])


def objective_gradient(params, A_r, S_r, A_i, S_i):
    """Gradient of minus four times the variance of a local state."""
    n = len(params) // 2
    u = np.asarray(params[:n], dtype=float)
    theta = np.asarray(params[n : 2 * n], dtype=float)
    norm = np.linalg.norm(u)
    if norm == 0:
        raise ValueError("The first n parameters (r1) must not be all zeros.")
    r = u / norm
    p = np.exp(1j * theta)
    z = r * p

    def quad(M):
        return np.vdot(z, M @ z)

    ar = quad(A_r)
    ai = quad(A_i)
    g = (S_r + S_r.conj().T) @ z + (S_i + S_i.conj().T) @ z
    g += -2 * ar * (A_r @ z) - 2 * np.conj(ar) * (A_r.conj().T @ z)
    g += -2 * ai * (A_i @ z) - 2 * np.conj(ai) * (A_i.conj().T @ z)
    h = np.conj(p) * g
    grad_r = np.real(h)
    grad_theta = r * np.imag(h)
    dot = np.dot(r, grad_r)
    grad_u = (grad_r - r * dot) / norm
    grad = -4.0 * np.concatenate([grad_u, grad_theta])
    return grad


def optimize_unpolarized_bound(
    HKL=(0.0, 0.0, 0.0),
    Gam_c=0.6,
    omega=0,
    chi=90.0,
    phi=0.0,
    th=15.0,
    tth=150.0,
    polind=0,
    folder="",
    ext="",
):
    """Maximize summed polarization variances in both charge sectors."""
    evecs_i = np.load(folder + "/evecs_i.npy")
    states_65 = np.load(folder + "/states_65.npy")
    states_56 = np.load(folder + "/states_56.npy")
    M_65 = states_65.conjugate() @ evecs_i
    M_56 = states_56.conjugate() @ evecs_i
    str1 = ""
    if ext == "M":
        str1 = str(HKL[2])
    elif ext == "E":
        str1 = str(omega)
    F_mag1 = np.load(folder + "/optimize/FQ_Fmag_" + ext + "_pol0_" + str1 + ".npz")[
        "arr_0"
    ]
    F_mag2 = np.load(folder + "/optimize/FQ_Fmag_" + ext + "_polpi2_" + str1 + ".npz")[
        "arr_0"
    ]
    F_mag_real1 = (F_mag1 + F_mag1.conj().T) * 0.5
    F_mag_imag1 = (F_mag1 - F_mag1.conj().T) * -0.5j
    F_mag_real2 = (F_mag2 + F_mag2.conj().T) * 0.5
    F_mag_imag2 = (F_mag2 - F_mag2.conj().T) * -0.5j
    A_65_real1 = M_65 @ F_mag_real1 @ M_65.conjugate().T
    A_65_imag1 = M_65 @ F_mag_imag1 @ M_65.conjugate().T
    S_65_real1 = M_65 @ F_mag_real1 @ F_mag_real1 @ M_65.conjugate().T
    S_65_imag1 = M_65 @ F_mag_imag1 @ F_mag_imag1 @ M_65.conjugate().T
    A_65_real2 = M_65 @ F_mag_real2 @ M_65.conjugate().T
    A_65_imag2 = M_65 @ F_mag_imag2 @ M_65.conjugate().T
    S_65_real2 = M_65 @ F_mag_real2 @ F_mag_real2 @ M_65.conjugate().T
    S_65_imag2 = M_65 @ F_mag_imag2 @ F_mag_imag2 @ M_65.conjugate().T
    A_56_real1 = M_56 @ F_mag_real1 @ M_56.conjugate().T
    A_56_imag1 = M_56 @ F_mag_imag1 @ M_56.conjugate().T
    S_56_real1 = M_56 @ F_mag_real1 @ F_mag_real1 @ M_56.conjugate().T
    S_56_imag1 = M_56 @ F_mag_imag1 @ F_mag_imag1 @ M_56.conjugate().T
    A_56_real2 = M_56 @ F_mag_real2 @ M_56.conjugate().T
    A_56_imag2 = M_56 @ F_mag_imag2 @ M_56.conjugate().T
    S_56_real2 = M_56 @ F_mag_real2 @ F_mag_real2 @ M_56.conjugate().T
    S_56_imag2 = M_56 @ F_mag_imag2 @ F_mag_imag2 @ M_56.conjugate().T

    def objective(params, A_r, S_r, A_i, S_i):
        n = len(params) // 2
        r = params[:n]
        r = r / np.sqrt(np.sum(r**2))
        theta = params[n:]
        z = r * np.exp(1j * theta)
        z = z.reshape(1, n)
        result = z.conjugate() @ S_r @ z.T + z.conjugate() @ S_i @ z.T
        result -= (z.conjugate() @ A_r @ z.T) ** 2 + (z.conjugate() @ A_i @ z.T) ** 2
        return -np.real(result * 4)

    def objective_65(params):
        return objective(
            params, A_65_real1, S_65_real1, A_65_imag1, S_65_imag1
        ) + objective(params, A_65_real2, S_65_real2, A_65_imag2, S_65_imag2)

    def objective_56(params):
        return objective(
            params, A_56_real1, S_56_real1, A_56_imag1, S_56_imag1
        ) + objective(params, A_56_real2, S_56_real2, A_56_imag2, S_56_imag2)

    def grad_65(params):
        return objective_gradient(
            params, A_65_real1, S_65_real1, A_65_imag1, S_65_imag1
        ) + objective_gradient(params, A_65_real2, S_65_real2, A_65_imag2, S_65_imag2)

    def grad_56(params):
        return objective_gradient(
            params, A_56_real1, S_56_real1, A_56_imag1, S_56_imag1
        ) + objective_gradient(params, A_56_real2, S_56_real2, A_56_imag2, S_56_imag2)

    n = 6
    object_func = objective_65
    grad_func = grad_65
    bounds = [(10 ** (-10), 1)] * n + [(0, 2 * np.pi)] * n
    best1 = None
    best = None
    n_restarts = 100
    for k in range(n_restarts):
        x0 = random_initial_state(n)
        result = minimize(
            fun=object_func,
            x0=x0,
            jac=grad_func,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=2000, ftol=1e-12, gtol=1e-08, maxls=50, maxcor=20),
        )
        if best1 is None or result.fun < best1.fun:
            best1 = result
    object_func = objective_56
    grad_func = grad_56
    bounds = [(10 ** (-10), 1)] * n + [(0, 2 * np.pi)] * n
    best2 = None
    n_restarts = 100
    for k in range(n_restarts):
        x0 = random_initial_state(n)
        result = minimize(
            fun=object_func,
            x0=x0,
            jac=grad_func,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=2000, ftol=1e-12, gtol=1e-08, maxls=50, maxcor=20),
        )
        if best2 is None or result.fun < best2.fun:
            best2 = result
    if best1.fun < best2.fun:
        best = best1
    else:
        best = best2
    return [-best.fun * 2, best.x]


def optimize_polarized_bound(
    HKL=(0.0, 0.0, 0.0),
    Gam_c=1.0,
    omega=0,
    pol=0,
    phi=0.0,
    thin=15.0,
    tth=150.0,
    folder="",
    ext="",
):
    """Maximize one polarization variance in both charge sectors."""
    evecs_i = np.load(folder + "/evecs_i.npy")
    states_65 = np.load(folder + "/states_65.npy")
    states_56 = np.load(folder + "/states_56.npy")
    M_65 = states_65.conjugate() @ evecs_i
    M_56 = states_56.conjugate() @ evecs_i
    str1 = ""
    if ext[0] == "M":
        str1 = str(HKL[2])
    elif ext[0] == "E":
        str1 = str(omega)
    if pol == 0:
        F_mag1 = np.load(
            folder + "/optimize/FQ_Fmag_" + ext + "_pol0_" + str1 + ".npz"
        )["arr_0"]
    else:
        F_mag1 = np.load(
            folder + "/optimize/FQ_Fmag_" + ext + "_polpi2_" + str1 + ".npz"
        )["arr_0"]
    F_mag_real1 = (F_mag1 + F_mag1.conj().T) * 0.5
    F_mag_imag1 = (F_mag1 - F_mag1.conj().T) * -0.5j
    A_65_real1 = M_65 @ F_mag_real1 @ M_65.conjugate().T
    A_65_imag1 = M_65 @ F_mag_imag1 @ M_65.conjugate().T
    S_65_real1 = M_65 @ F_mag_real1 @ F_mag_real1 @ M_65.conjugate().T
    S_65_imag1 = M_65 @ F_mag_imag1 @ F_mag_imag1 @ M_65.conjugate().T
    A_56_real1 = M_56 @ F_mag_real1 @ M_56.conjugate().T
    A_56_imag1 = M_56 @ F_mag_imag1 @ M_56.conjugate().T
    S_56_real1 = M_56 @ F_mag_real1 @ F_mag_real1 @ M_56.conjugate().T
    S_56_imag1 = M_56 @ F_mag_imag1 @ F_mag_imag1 @ M_56.conjugate().T

    def objective(params, A_r, S_r, A_i, S_i):
        n = len(params) // 2
        r = params[:n]
        r = r / np.sqrt(np.sum(r**2))
        theta = params[n:]
        z = r * np.exp(1j * theta)
        z = z.reshape(1, n)
        result = z.conjugate() @ S_r @ z.T + z.conjugate() @ S_i @ z.T
        result -= (z.conjugate() @ A_r @ z.T) ** 2 + (z.conjugate() @ A_i @ z.T) ** 2
        return -np.real(result * 4)

    def objective_65(params):
        return objective(params, A_65_real1, S_65_real1, A_65_imag1, S_65_imag1)

    def objective_56(params):
        return objective(params, A_56_real1, S_56_real1, A_56_imag1, S_56_imag1)

    def grad_65(params):
        return objective_gradient(
            params, A_65_real1, S_65_real1, A_65_imag1, S_65_imag1
        )

    def grad_56(params):
        return objective_gradient(
            params, A_56_real1, S_56_real1, A_56_imag1, S_56_imag1
        )

    n = 6
    object_func = objective_65
    grad_func = grad_65
    bounds = [(10 ** (-10), 1)] * n + [(0, 2 * np.pi)] * n
    best1 = None
    best = None
    n_restarts = 100
    for k in range(n_restarts):
        x0 = random_initial_state(n)
        result = minimize(
            fun=object_func,
            x0=x0,
            jac=grad_func,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=2000, ftol=1e-12, gtol=1e-08, maxls=50, maxcor=20),
        )
        if best1 is None or result.fun < best1.fun:
            best1 = result
    object_func = objective_56
    grad_func = grad_56
    bounds = [(10 ** (-10), 1)] * n + [(0, 2 * np.pi)] * n
    best2 = None
    n_restarts = 100
    for k in range(n_restarts):
        x0 = random_initial_state(n)
        result = minimize(
            fun=object_func,
            x0=x0,
            jac=grad_func,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=2000, ftol=1e-12, gtol=1e-08, maxls=50, maxcor=20),
        )
        if best2 is None or result.fun < best2.fun:
            best2 = result
    if best1.fun < best2.fun:
        best = best1
    else:
        best = best2
    return [-best.fun * 2, best.x]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("ta_momentum_work"),
        help="Directory for Hamiltonians and intermediate operators.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for the final files loaded by the plotting notebook.",
    )
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdirectory in ("", "garage", "optimize", "pol"):
        (work_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    cfg = import_config("config_Ta_Eg.py")
    coordsfin = np.loadtxt("Ba3TaIr2O9_collectedV.txt")
    labels0 = list(cfg.greedy_bounds.keys())
    witness = dimer_witness(folder=str(work_dir))
    witness.construct_basis(11, eg_max=1)
    witness.produce_dipole()
    save_charge_sector_embeddings(work_dir)
    omega = -559
    motors = cfg.motors
    Ls = np.arange(0.0, 19.4, 0.1)
    Hs = np.interp(Ls, motors[:, 2], motors[:, 0])
    Ks = np.interp(Ls, motors[:, 2], motors[:, 1])
    HKLs = np.vstack([Hs, Ks, Ls]).T
    tths = np.interp(Ls, motors[:, 2], motors[:, 3])
    ths = np.interp(Ls, motors[:, 2], motors[:, 4])
    chis = np.interp(Ls, motors[:, 2], motors[:, 5])
    phis = np.interp(Ls, motors[:, 2], motors[:, 6])
    HKLs_transpose = np.array([np.vstack((Hs, Ks, Ls))])
    Bmat = np.array(
        [
            [1.09177851, 0.630338615, -1.33704305e-16],
            [0.0, 1.26067723, -1.33704305e-16],
            [0.0, -0.0, 0.436029515],
        ]
    )
    Qs = np.squeeze(np.dot(Bmat, HKLs_transpose).transpose())
    theorSpectra = []
    numNegTheor = []
    theorSpectraL2 = []
    numNegTheorL2 = []
    denomTheor = []
    for ind in range(24):
        dp = {labels0[i]: coordsfin[ind][i] for i in range(len(labels0))}
        emat_params = dict(
            tenDq=dp["tenDq"],
            soc=dp["soc"],
            theta=49.0,
            kappa=0.1,
            a0=0.0,
            Vdd_sigma=dp["Vdd_sigma"],
            Vdd_pi=dp["Vdd_pi"],
            Vdd_delta=dp["Vdd_delta"],
            soc_c=dp["soc_c"],
        )
        Jdd = (dp["F2_dd"] + dp["F4_dd"]) / 14.0
        umat_params = dict(
            U_dd=3.0,
            J_dd=Jdd,
            U_q=2.5,
            F2_dp=dp["F2_dp"],
            G1_dp=dp["G1_dp"],
            G3_dp=dp["G3_dp"],
            F4F2_ratio=dp["F4_dd"] / dp["F2_dd"],
        )
        witness.load_params(**emat_params, **umat_params)
        witness.load_hmat_4f(overwrite=True)
        witness.perform_ED()
        FQ_params = dict(Gam_c=dp["Gam_c"], omega=omega, folder=witness.folder)
        for i in tqdm(range(len(Ls))):
            build_scattering_operator(
                tth=tths[i],
                th=ths[i],
                phi=phis[i],
                HKL=HKLs[i],
                chi=chis[i],
                polind=0,
                **FQ_params,
                ext="M_pol0",
            )
            build_scattering_operator(
                tth=tths[i],
                th=ths[i],
                phi=phis[i],
                HKL=HKLs[i],
                chi=chis[i],
                polind=1,
                **FQ_params,
                ext="M_polpi2",
            )
        FQ = np.zeros((2, len(Ls)), dtype=float)
        for i in tqdm(range(len(Ls))):
            FQ_result = optimize_unpolarized_bound(
                tth=tths[i],
                th=ths[i],
                phi=phis[i],
                HKL=HKLs[i],
                chi=chis[i],
                polind=0,
                **FQ_params,
                ext="M",
            )
            FQ[0][i] = FQ_result[0]
        FQ_MmapT1 = dict(momenta=Qs, FQ=FQ)
        np.savez(witness.folder + "/pol/FQ_MmapT1.npz", **FQ_MmapT1)
        FQ = np.zeros((2, len(Ls), 2), dtype=float)
        RIXS_params = dict(
            T=9.0,
            Gam_c=dp["Gam_c"],
            sigma0=0.15,
            sigmas=[0.022, 0.022, 0.022, 0.022, 0.054, 0.054],
            fraction=0.5,
            omega=omega,
            eloss=np.arange(0.0, 10.0, 0.005),
            eline=None,
            folder=witness.folder,
        )
        rixs = calc_RIXS_dimerTa(
            tth=tths, th=ths, chi=chis, phi=phis, HKL=HKLs, **RIXS_params
        )
        rixs_Mmap = dict(
            pol=[0, 1], Qs=Qs, eloss=RIXS_params["eloss"], rixs=np.squeeze(rixs)
        )
        np.savez(witness.folder + "/pol/rixs_Mmap.npz", **rixs_Mmap)
        conjugate_rixs = conjugate_RIXS_dimerTa(
            tth=tths, th=ths, chi=chis, phi=phis, HKL=HKLs, **RIXS_params
        )
        conjugate_rixs_Mmap = dict(
            pol=[0, 1],
            Qs=Qs,
            eloss=RIXS_params["eloss"],
            rixs=np.squeeze(conjugate_rixs),
        )
        np.savez(witness.folder + "/pol/conjugate_rixs_Mmap.npz", **conjugate_rixs_Mmap)
        T = 9.0
        kB = 8.6173303e-05
        beta = 1 / (kB * T)
        rixs = np.load(witness.folder + "/pol/rixs_Mmap.npz")
        rixs_con = np.load(witness.folder + "/pol/conjugate_rixs_Mmap.npz")
        rixs = dict(rixs)
        rixs1 = rixs["rixs"][0, :, :]
        rixs2 = rixs["rixs"][1, :, :]
        rixs_con = dict(rixs_con)
        rixs1_con = rixs_con["rixs"][0, :]
        rixs2_con = rixs_con["rixs"][1, :]
        Ls = np.arange(0.0, 19.4, 0.1)
        eloss = rixs["eloss"]
        factor = 2 * np.tanh(beta * eloss / 2) * (1 - np.exp(-beta * eloss))
        given_axis = 1
        dim = np.ones((1, rixs1.ndim), int).ravel()
        dim[given_axis] = -1
        factor_reshaped = factor.reshape(dim)
        summand1 = rixs1 * factor_reshaped + rixs1_con * factor_reshaped
        summand2 = rixs2 * factor_reshaped + rixs2_con * factor_reshaped
        summand_neg = rixs1_con * factor_reshaped + rixs2_con * factor_reshaped
        step = eloss[1] - eloss[0]
        chooser = (eloss > 0.3) * (eloss < 6.0)
        chooser_neg = (eloss > 0.3) * (eloss < 6.0)
        summand1_chooser = summand1[:, chooser]
        summand2_chooser = summand2[:, chooser]
        summand_neg_chooser = summand_neg[:, chooser_neg]
        nominator1 = np.sum(summand1_chooser, axis=1) * step
        nominator2 = np.sum(summand2_chooser, axis=1) * step
        nominator_neg = np.sum(summand_neg_chooser, axis=1) * step
        numNegTheor.append(nominator_neg)
        denomTheor.append(FQ_MmapT1["FQ"][0])
        theorSpectra.append((nominator1 + nominator2) / FQ_MmapT1["FQ"][0])
        chooser = (eloss > 2.0) * (eloss < 6.0)
        chooser_neg = (eloss > 2.0) * (eloss < 6.0)
        summand1_chooser = summand1[:, chooser]
        summand2_chooser = summand2[:, chooser]
        summand_neg_chooser = summand_neg[:, chooser_neg]
        nominator1 = np.sum(summand1_chooser, axis=1) * step
        nominator2 = np.sum(summand2_chooser, axis=1) * step
        nominator_neg = np.sum(summand_neg_chooser, axis=1) * step
        numNegTheorL2.append(nominator_neg)
        theorSpectraL2.append((nominator1 + nominator2) / FQ_MmapT1["FQ"][0])
    theorSpectra_arr = np.stack(theorSpectra, axis=0)
    numNegTheor_arr = np.stack(numNegTheor, axis=0)
    theorSpectraL2_arr = np.stack(theorSpectraL2, axis=0)
    numNegTheorL2_arr = np.stack(numNegTheorL2, axis=0)
    denomTheor_arr = np.stack(denomTheor, axis=0)
    np.savez_compressed(
        output_dir / "Ta_Eg_momdep_outputsV.npz",
        theorSpectra=theorSpectra_arr,
        numNegTheor=numNegTheor_arr,
        theorSpectraL2=theorSpectraL2_arr,
        numNegTheorL2=numNegTheorL2_arr,
        denomTheor=denomTheor_arr,
    )
    omega = -559
    theorSpectraM0 = []
    theorSpectraMpi2 = []
    for ind in [9]:
        dp = {labels0[i]: coordsfin[ind][i] for i in range(len(labels0))}
        emat_params = dict(
            tenDq=dp["tenDq"],
            soc=dp["soc"],
            theta=49.0,
            kappa=0.1,
            a0=0.0,
            Vdd_sigma=dp["Vdd_sigma"],
            Vdd_pi=dp["Vdd_pi"],
            Vdd_delta=dp["Vdd_delta"],
            soc_c=dp["soc_c"],
        )
        Jdd = (dp["F2_dd"] + dp["F4_dd"]) / 14.0
        umat_params = dict(
            U_dd=3.0,
            J_dd=Jdd,
            U_q=2.5,
            F2_dp=dp["F2_dp"],
            G1_dp=dp["G1_dp"],
            G3_dp=dp["G3_dp"],
            F4F2_ratio=dp["F4_dd"] / dp["F2_dd"],
        )
        witness.load_params(**emat_params, **umat_params)
        witness.load_hmat_4f(overwrite=True)
        witness.perform_ED()
        FQ_params = dict(Gam_c=dp["Gam_c"], omega=omega, folder=witness.folder)
        for i in tqdm(range(len(Ls))):
            build_scattering_operator(
                tth=tths[i],
                th=ths[i],
                phi=phis[i],
                HKL=HKLs[i],
                chi=chis[i],
                polind=0,
                **FQ_params,
                ext="M_pol0",
            )
            build_scattering_operator(
                tth=tths[i],
                th=ths[i],
                phi=phis[i],
                HKL=HKLs[i],
                chi=chis[i],
                polind=1,
                **FQ_params,
                ext="M_polpi2",
            )
        FQ_pol = np.zeros((2, len(Ls)), dtype=float)
        for i in tqdm(range(len(Ls))):
            res = optimize_polarized_bound(
                tth=tths[i],
                thin=ths[i],
                HKL=HKLs[i],
                pol=0,
                phi=phis[i],
                **FQ_params,
                ext="M",
            )
            FQ_pol[0, i] = res[0]
        for i in tqdm(range(len(Ls))):
            res = optimize_polarized_bound(
                tth=tths[i],
                thin=ths[i],
                HKL=HKLs[i],
                pol=1,
                phi=phis[i],
                **FQ_params,
                ext="M",
            )
            FQ_pol[1, i] = res[0]
        FQ_MmapT55pol0 = dict(momenta=Qs, FQ=np.copy(FQ_pol[0]))
        FQ_MmapT55polpi2 = dict(momenta=Qs, FQ=np.copy(FQ_pol[1]))
        RIXS_params = dict(
            T=9.0,
            Gam_c=dp["Gam_c"],
            sigma0=0.15,
            sigmas=[0.022, 0.022, 0.022, 0.022, 0.054, 0.054],
            fraction=0.5,
            omega=omega,
            eloss=np.arange(0.0, 10.0, 0.005),
            eline=None,
            folder=witness.folder,
        )
        rixs = calc_RIXS_dimerTa(
            tth=tths, th=ths, chi=chis, phi=phis, HKL=HKLs, **RIXS_params
        )
        rixs_Mmap = dict(
            pol=[0, 1], Qs=Qs, eloss=RIXS_params["eloss"], rixs=np.squeeze(rixs)
        )
        np.savez(witness.folder + "/pol/rixs_Mmap.npz", **rixs_Mmap)
        conjugate_rixs = conjugate_RIXS_dimerTa(
            tth=tths, th=ths, chi=chis, phi=phis, HKL=HKLs, **RIXS_params
        )
        conjugate_rixs_Mmap = dict(
            pol=[0, 1],
            Qs=Qs,
            eloss=RIXS_params["eloss"],
            rixs=np.squeeze(conjugate_rixs),
        )
        np.savez(witness.folder + "/pol/conjugate_rixs_Mmap.npz", **conjugate_rixs_Mmap)
        T = 9.0
        kB = 8.6173303e-05
        beta = 1 / (kB * T)
        rixs = np.load(witness.folder + "/pol/rixs_Mmap.npz")
        rixs_con = np.load(witness.folder + "/pol/conjugate_rixs_Mmap.npz")
        rixs = dict(rixs)
        rixs1 = rixs["rixs"][0]
        rixs2 = rixs["rixs"][1]
        rixs_con = dict(rixs_con)
        rixs1_con = rixs_con["rixs"][0]
        rixs2_con = rixs_con["rixs"][1]
        eloss = rixs["eloss"]
        factor = 2 * np.tanh(beta * eloss / 2) * (1 - np.exp(-beta * eloss))
        given_axis = 1
        dim = np.ones((1, rixs1.ndim), int).ravel()
        dim[given_axis] = -1
        factor_reshaped = factor.reshape(dim)
        summand1 = rixs1 * factor_reshaped + rixs1_con * factor_reshaped
        summand2 = rixs2 * factor_reshaped + rixs2_con * factor_reshaped
        step = eloss[1] - eloss[0]
        chooser = (eloss > 0.3) * (eloss < 6.0)
        summand1_chooser = summand1[:, chooser]
        summand2_chooser = summand2[:, chooser]
        nominator1 = np.sum(summand1_chooser, axis=1) * step
        nominator2 = np.sum(summand2_chooser, axis=1) * step
        theorSpectraM0.append(nominator1 / FQ_MmapT55pol0["FQ"])
        theorSpectraMpi2.append(nominator2 / FQ_MmapT55polpi2["FQ"])
    theorSpectraM0_arr = np.stack(theorSpectraM0, axis=0)
    theorSpectraMpi2_arr = np.stack(theorSpectraMpi2, axis=0)
    np.savez_compressed(
        output_dir / "Ta_Eg_momdep_pol_outputsV.npz",
        theorSpectraM0=theorSpectraM0_arr,
        theorSpectraMpi2=theorSpectraMpi2_arr,
        nominator1=nominator1,
        nominator2=nominator2,
    )


if __name__ == "__main__":
    main()
