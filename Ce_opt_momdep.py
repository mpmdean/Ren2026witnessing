#!/usr/bin/env python3
"""Generate the Ce momentum-dependence files used by the plotting notebook.

Run from the directory containing config_Ce_t2g.py,
Ba3CeIr2O9_test4_collectedV4V.txt, Ce.npy, CeEmap.numpy and CeEmapEg.numpy.
The project modules must be importable on the Python path.
Dependencies: NumPy, SciPy, EDRIXS, tqdm, and the project modules
optEDRIXS, utils_Ce, witness, and witness_utils.

All fitted rows are retained in their original order for unpolarized results.
Polarized results use fitted row 0. Each bound uses 100 random L-BFGS-B
restarts, with the original objective, gradient, bounds and tolerances.
"""

import argparse
from pathlib import Path
import edrixs
import numpy as np
from edrixs import dipole_polvec_rixs, scattering_mat
from scipy.optimize import minimize
from tqdm import tqdm
from optEDRIXS import import_config
from witness import calc_RIXS_dimerCe, conjugate_RIXS_dimerCe, dimer_witness
from witness_utils import SparseBinary, get_fock_bin_by_N_S, rmat_t2o


def product_state_indices():
    """Map the ordered six-by-six local product basis into the ED basis."""
    basis = SparseBinary(
        np.vstack([get_fock_bin_by_N_S(8, 0, 12, 10, dispinfo=False)]), shape=[None, 20]
    )
    coordinates = np.array(
        [0, 1, 6, 7, 10, 11, 16, 17, 2, 3, 4, 5, 8, 9, 12, 13, 14, 15, 18, 19]
    )
    basis.change_coordinates(coordinates, shape=[None, 20])
    weights = 1 << np.arange(19, -1, -1)
    basis_indices = basis.toarray() @ weights
    local_basis = np.array(edrixs.get_fock_bin_by_N(6, 5, 4, 0), dtype=np.int64)[
        :, [6, 7, 0, 1, 2, 3, 8, 9, 4, 5]
    ]
    product_basis = np.array(
        [np.concatenate([a, b]) for a in local_basis for b in local_basis]
    )
    product_indices = product_basis @ weights
    return np.array(
        [np.argwhere(basis_indices == index)[0] for index in product_indices]
    ).ravel()


def build_scattering_operator(
    HKL=(0.0, 0.0, 0.0),
    Gam_c=1.0,
    omega=0,
    phi=0.0,
    thin=15.0,
    tth=150.0,
    pol=(0, 0),
    folder="",
    ext="",
):
    """Save the RIXS scattering operator in the ED eigenbasis."""
    evals_i = np.load(folder + "/evals_i.npy")
    evals_n_all = [np.load(folder + "/evals_n1.npy"), np.load(folder + "/evals_n2.npy")]
    T_abs_all = dict(np.load(folder + "/T_abs.npz"))
    T_abs_all = [T_abs_all["T_abs1"], T_abs_all["T_abs2"]]
    pos = [
        [2.95175, 1.70419, 2.41068],
        [2.95175, 1.70419, 4.94682],
        [0.0, 3.40839, 12.30432],
        [0.0, 3.40839, 9.76818],
    ]
    Umat = np.array(
        [
            [1.06431529, 0.614482718, -1.30341031e-16],
            [0.0, 1.22896544, -1.30341031e-16],
            [0.0, -0.0, 0.426991866],
        ]
    )
    F_fis = np.zeros((2, 3, 3, len(evals_i), len(evals_i)), dtype=complex)
    for nnTM, (evals_n, T_abs) in enumerate(zip(evals_n_all, T_abs_all)):
        T_emi = np.stack([np.conj(np.transpose(T0)) for T0 in T_abs])
        F_fis[nnTM, :, :, :, :] += scattering_mat(
            evals_i[:], evals_n[:], T_abs[:, :, :], T_emi, omega, Gam_c
        )
    alpha, beta = pol
    F_mag = np.zeros((len(evals_i), len(evals_i)), dtype=complex)
    for nnTM in range(2):
        ei0, ef0 = dipole_polvec_rixs(
            thin / 180.0 * np.pi,
            (-thin + tth) / 180.0 * np.pi,
            phi / 180.0 * np.pi,
            alpha,
            beta,
        )
        Q = np.dot(HKL, Umat)
        ei = np.dot(ei0, rmat_t2o(nnTM + 1))
        ef = np.dot(ef0, rmat_t2o(nnTM + 1))
        phase = np.exp(-1j * np.dot(Q, pos[nnTM]))
        for m in range(3):
            for n in range(3):
                F_mag[:, :] += ef[m] * F_fis[nnTM, m, n] * phase * ei[n]
    str1 = ""
    if ext[0] == "M":
        str1 = str(HKL[2])
    elif ext[0] == "E":
        str1 = str(omega)
    np.savez(folder + "/optimize/FQ_Fmag_" + ext + "_" + str1 + ".npz", F_mag)


def objective_gradient(params, A_r, S_r, A_i, S_i):
    """Gradient of minus four times the variance for identical local states."""
    n = len(params) // 2
    u = np.asarray(params[:n], dtype=float)
    theta = np.asarray(params[n : 2 * n], dtype=float)
    norm = np.linalg.norm(u)
    if norm == 0:
        raise ValueError("The first n parameters (r1) must not be all zeros.")
    r = u / norm
    p = np.exp(1j * theta)
    y = r * p
    z = np.kron(y, y)

    def quad(M):
        return np.vdot(z, M @ z)

    ar = quad(A_r)
    ai = quad(A_i)
    g = (S_r + S_r.conj().T) @ z + (S_i + S_i.conj().T) @ z
    g += -2 * ar * (A_r @ z) - 2 * np.conj(ar) * (A_r.conj().T @ z)
    g += -2 * ai * (A_i @ z) - 2 * np.conj(ai) * (A_i.conj().T @ z)
    G = g.reshape(n, n)
    g_y = (G + G.T) @ np.conj(y)
    h = np.conj(p) * g_y
    grad_r = np.real(h)
    grad_theta = r * np.imag(h)
    dot = np.dot(r, grad_r)
    grad_u = (grad_r - r * dot) / norm
    grad = -4.0 * np.concatenate([grad_u, grad_theta])
    return grad


def random_initial_state(n, rng=None, eps=1e-12):
    """Draw six amplitudes and phases using the original restart distribution."""
    if rng is None:
        rng = np.random.default_rng()
    r0 = rng.random(n)
    r0 = np.maximum(r0, eps)
    theta0 = rng.uniform(0.0, 2 * np.pi, size=n)
    return np.concatenate([r0, theta0])


def optimize_unpolarized_bound(
    HKL=(0.0, 0.0, 0.0),
    Gam_c=1.0,
    omega=0,
    phi=0.0,
    thin=15.0,
    tth=150.0,
    folder="",
    ext="",
    *,
    indlist
):
    """Maximize the summed polarization variances over identical local states."""
    evecs_i = np.load(folder + "/evecs_i.npy")
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
    A_55_real1 = (evecs_i @ F_mag_real1 @ evecs_i.conjugate().T)[indlist][:, indlist]
    A_55_imag1 = (evecs_i @ F_mag_imag1 @ evecs_i.conjugate().T)[indlist][:, indlist]
    S_55_real1 = (evecs_i @ F_mag_real1 @ F_mag_real1 @ evecs_i.conjugate().T)[indlist][
        :, indlist
    ]
    S_55_imag1 = (evecs_i @ F_mag_imag1 @ F_mag_imag1 @ evecs_i.conjugate().T)[indlist][
        :, indlist
    ]
    A_55_real2 = (evecs_i @ F_mag_real2 @ evecs_i.conjugate().T)[indlist][:, indlist]
    A_55_imag2 = (evecs_i @ F_mag_imag2 @ evecs_i.conjugate().T)[indlist][:, indlist]
    S_55_real2 = (evecs_i @ F_mag_real2 @ F_mag_real2 @ evecs_i.conjugate().T)[indlist][
        :, indlist
    ]
    S_55_imag2 = (evecs_i @ F_mag_imag2 @ F_mag_imag2 @ evecs_i.conjugate().T)[indlist][
        :, indlist
    ]

    def objective(params, A_r, S_r, A_i, S_i):
        n = len(params) // 2
        r1 = params[:n]
        r1 = r1 / np.sqrt(np.sum(r1**2))
        theta1 = params[n : 2 * n]
        z1 = r1 * np.exp(1j * theta1)
        z1 = z1.reshape(1, n)
        z = np.kron(z1, z1)
        result = z.conjugate() @ S_r @ z.T + z.conjugate() @ S_i @ z.T
        result -= (z.conjugate() @ A_r @ z.T) ** 2 + (z.conjugate() @ A_i @ z.T) ** 2
        return -np.real(result * 4)

    def objective_55(params):
        return objective(
            params, A_55_real1, S_55_real1, A_55_imag1, S_55_imag1
        ) + objective(params, A_55_real2, S_55_real2, A_55_imag2, S_55_imag2)

    def grad_kron(params):
        return objective_gradient(
            params, A_55_real1, S_55_real1, A_55_imag1, S_55_imag1
        ) + objective_gradient(params, A_55_real2, S_55_real2, A_55_imag2, S_55_imag2)

    n = 6
    object_func = objective_55
    grad_func = grad_kron
    bounds = [(10 ** (-10), 1)] * n + [(0, 2 * np.pi)] * n
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
        if best is None or result.fun < best.fun:
            best = result
    return [-best.fun * 2, best.x]


def optimize_polarized_bound(
    HKL=(0.0, 0.0, 0.0),
    Gam_c=1.0,
    omega=0,
    phi=0.0,
    thin=15.0,
    tth=150.0,
    folder="",
    ext="",
    *,
    indlist
):
    """Maximize one polarization variance over identical local states."""
    evecs_i = np.load(folder + "/evecs_i.npy")
    str1 = ""
    if ext[0] == "M":
        str1 = str(HKL[2])
    elif ext[0] == "E":
        str1 = str(omega)
    F_mag1 = np.load(folder + "/optimize/FQ_Fmag_" + ext + "_" + str1 + ".npz")["arr_0"]
    F_mag_real1 = (F_mag1 + F_mag1.conj().T) * 0.5
    F_mag_imag1 = (F_mag1 - F_mag1.conj().T) * -0.5j
    A_55_real1 = (evecs_i @ F_mag_real1 @ evecs_i.conjugate().T)[indlist][:, indlist]
    A_55_imag1 = (evecs_i @ F_mag_imag1 @ evecs_i.conjugate().T)[indlist][:, indlist]
    S_55_real1 = (evecs_i @ F_mag_real1 @ F_mag_real1 @ evecs_i.conjugate().T)[indlist][
        :, indlist
    ]
    S_55_imag1 = (evecs_i @ F_mag_imag1 @ F_mag_imag1 @ evecs_i.conjugate().T)[indlist][
        :, indlist
    ]

    def objective(params, A_r, S_r, A_i, S_i):
        n = len(params) // 2
        r1 = params[:n]
        r1 = r1 / np.sqrt(np.sum(r1**2))
        theta1 = params[n : 2 * n]
        z1 = r1 * np.exp(1j * theta1)
        z1 = z1.reshape(1, n)
        z = np.kron(z1, z1)
        result = z.conjugate() @ S_r @ z.T + z.conjugate() @ S_i @ z.T
        result -= (z.conjugate() @ A_r @ z.T) ** 2 + (z.conjugate() @ A_i @ z.T) ** 2
        return -np.real(result * 4)

    def objective_55(params):
        return objective(params, A_55_real1, S_55_real1, A_55_imag1, S_55_imag1)

    def grad_kron(params):
        return objective_gradient(
            params, A_55_real1, S_55_real1, A_55_imag1, S_55_imag1
        )

    n = 6
    object_func = objective_55
    grad_func = grad_kron
    bounds = [(10 ** (-10), 1)] * n + [(0, 2 * np.pi)] * n
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
        if best is None or result.fun < best.fun:
            best = result
    return [-best.fun * 2, best.x]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("ce_momentum_work"),
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
    cfg = import_config("config_Ce_t2g.py")
    coordsfin = np.loadtxt("Ba3CeIr2O9_test4_collectedV4V.txt")
    labels0 = list(cfg.greedy_bounds.keys())
    indlist = product_state_indices()
    witness = dimer_witness(folder=str(work_dir))
    witness.construct_basis(10, eg_max=0)
    witness.produce_dipole()
    omega = -558.5
    Ls = np.arange(0.0, 20.1, 0.1)
    Qs = np.sqrt(1.228965**2 / 4 + (Ls * 0.426991866) ** 2)
    tths = np.rad2deg(np.arcsin(1.1054837271511369 * Qs / 4.0 / np.pi) * 2)
    thins = tths / 2 - np.rad2deg(np.arctan(1.228965 / 2 / Ls / 0.426991866))
    initial_guess = []
    theorSpectra = []
    numNegTheor = []
    denomTheor = []
    for ind in range(len(coordsfin)):
        initial_guess.append([])
        dp = {labels0[i]: coordsfin[ind][i] for i in range(len(labels0))}
        emat_params = dict(
            tenDq=dp["tenDq"],
            soc=dp["soc"],
            theta=50.86,
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
        for nth, (tth, thin, L) in tqdm(enumerate(zip(tths, thins, Ls))):
            FQ_params = dict(
                Gam_c=dp["Gam_c"], omega=omega, pol=(0, 0), folder=witness.folder
            )
            build_scattering_operator(
                tth=tth,
                thin=thin,
                HKL=[-0.5, 0.0, L],
                phi=10.0,
                **FQ_params,
                ext="M_pol0"
            )
            FQ_params = dict(
                Gam_c=dp["Gam_c"],
                omega=omega,
                pol=(0, np.pi / 2),
                folder=witness.folder,
            )
            build_scattering_operator(
                tth=tth,
                thin=thin,
                HKL=[-0.5, 0.0, L],
                phi=10.0,
                **FQ_params,
                ext="M_polpi2"
            )
        FQ = np.zeros((2, len(Ls), 2), dtype=float)
        for nth, (tth, thin, L) in tqdm(enumerate(zip(tths, thins, Ls))):
            FQ_params = dict(Gam_c=dp["Gam_c"], omega=omega, folder=witness.folder)
            res = optimize_unpolarized_bound(
                tth=tth,
                thin=thin,
                HKL=[-0.5, 0.0, L],
                phi=10.0,
                **FQ_params,
                ext="M",
                indlist=indlist
            )
            FQ[0][nth] = [entry[0] for entry in [res]]
            initial_guess[ind].append(res[1])
        FQ_MmapT55unpol = dict(momenta=Qs, FQ=FQ)
        np.savez(witness.folder + "/pol/FQ_MmapT55unpol.npz", **FQ_MmapT55unpol)
        RIXS_params = dict(
            Gam_c=dp["Gam_c"],
            sigma=0.075,
            fraction=0.0,
            omega=-558.5,
            eloss=np.arange(-0.0, 10.0, 0.005),
            pol=[(0, 0), (0, np.pi / 2.0)],
            T=9.0,
            eline=None,
            folder=witness.folder,
        )
        rixs_all = []
        for tth, thin, L in zip(tths, thins, Ls):
            rixs_all.append(
                calc_RIXS_dimerCe(
                    tth=tth, thin=thin, HKL=[-0.5, 0.0, L], phi=10.0, **RIXS_params
                )
            )
        rixs_Mmap = dict(
            momenta=Qs,
            pol=RIXS_params["pol"],
            eloss=RIXS_params["eloss"],
            rixs=np.squeeze(rixs_all),
        )
        np.savez(witness.folder + "/pol/rixs_Mmap.npz", **rixs_Mmap)
        conjugate_rixs_all = []
        for tth, thin, L in zip(tths, thins, Ls):
            conjugate_rixs_all.append(
                conjugate_RIXS_dimerCe(
                    tth=tth, thin=thin, HKL=[-0.5, 0.0, L], phi=10.0, **RIXS_params
                )
            )
        conjugate_rixs_Mmap = dict(
            momenta=Qs,
            pol=RIXS_params["pol"],
            eloss=RIXS_params["eloss"],
            rixs=np.squeeze(conjugate_rixs_all),
        )
        np.savez(witness.folder + "/pol/conjugate_rixs_Mmap.npz", **conjugate_rixs_Mmap)
        T = 9.0
        kB = 8.6173303e-05
        beta = 1 / (kB * T)
        rixs = np.load(witness.folder + "/pol/rixs_Mmap.npz")
        rixs_con = np.load(witness.folder + "/pol/conjugate_rixs_Mmap.npz")
        rixs = dict(rixs)
        rixs1 = rixs["rixs"][:, 0, :]
        rixs2 = rixs["rixs"][:, 1, :]
        rixs_con = dict(rixs_con)
        rixs1_con = rixs_con["rixs"][:, 0]
        rixs2_con = rixs_con["rixs"][:, 1]
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
        chooser = (eloss > 0.2) * (eloss < 2.0)
        chooser_neg = (eloss > 0.2) * (eloss < 2.0)
        summand1_chooser = summand1[:, chooser]
        summand2_chooser = summand2[:, chooser]
        summand_neg_chooser = summand_neg[:, chooser_neg]
        nominator1 = np.sum(summand1_chooser, axis=1) * step
        nominator2 = np.sum(summand2_chooser, axis=1) * step
        nominator_neg = np.sum(summand_neg_chooser, axis=1) * step
        numNegTheor.append(nominator_neg)
        denomTheor.append(FQ_MmapT55unpol["FQ"][0][:, 0])
        theorSpectra.append((nominator1 + nominator2) / FQ_MmapT55unpol["FQ"][0][:, 0])
    theorSpectra_arr = np.stack(theorSpectra, axis=0)
    numNegTheor_arr = np.stack(numNegTheor, axis=0)
    denomTheor_arr = np.stack(denomTheor, axis=0)
    np.savez_compressed(
        output_dir / "Ce_t2g_momdep_outputsV4Vr.npz",
        theorSpectra=theorSpectra_arr,
        numNegTheor=numNegTheor_arr,
        denomTheor=denomTheor_arr,
        argmin=np.array(initial_guess),
    )
    omega = -558.5
    Ls = np.arange(0.0, 20.1, 0.1)
    theorSpectraM0 = []
    theorSpectraMpi2 = []
    for ind in [0]:
        dp = {labels0[i]: coordsfin[ind][i] for i in range(len(labels0))}
        emat_params = dict(
            tenDq=dp["tenDq"],
            soc=dp["soc"],
            theta=50.86,
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
        for nth, (tth, thin, L) in tqdm(enumerate(zip(tths, thins, Ls))):
            FQ_params = dict(
                Gam_c=dp["Gam_c"], omega=omega, pol=(0, 0), folder=witness.folder
            )
            build_scattering_operator(
                tth=tth,
                thin=thin,
                HKL=[-0.5, 0.0, L],
                phi=10.0,
                **FQ_params,
                ext="M_pol0"
            )
            FQ_params = dict(
                Gam_c=dp["Gam_c"],
                omega=omega,
                pol=(0, np.pi / 2),
                folder=witness.folder,
            )
            build_scattering_operator(
                tth=tth,
                thin=thin,
                HKL=[-0.5, 0.0, L],
                phi=10.0,
                **FQ_params,
                ext="M_polpi2"
            )
        FQ = np.zeros((2, len(Ls), 2), dtype=float)
        for nth, (tth, thin, L) in tqdm(enumerate(zip(tths, thins, Ls))):
            FQ_params = dict(Gam_c=dp["Gam_c"], omega=omega, folder=witness.folder)
            res = optimize_polarized_bound(
                tth=tth,
                thin=thin,
                HKL=[-0.5, 0.0, L],
                phi=10.0,
                **FQ_params,
                ext="M_pol0",
                indlist=indlist
            )
            FQ[0][nth] = [entry[0] for entry in [res]]
        FQ_MmapT55pol0 = dict(momenta=Qs, FQ=np.copy(FQ))
        for nth, (tth, thin, L) in tqdm(enumerate(zip(tths, thins, Ls))):
            FQ_params = dict(Gam_c=dp["Gam_c"], omega=omega, folder=witness.folder)
            res = optimize_polarized_bound(
                tth=tth,
                thin=thin,
                HKL=[-0.5, 0.0, L],
                phi=10.0,
                **FQ_params,
                ext="M_polpi2",
                indlist=indlist
            )
            FQ[0][nth] = [entry[0] for entry in [res]]
        FQ_MmapT55polpi2 = dict(momenta=Qs, FQ=np.copy(FQ))
        RIXS_params = dict(
            Gam_c=dp["Gam_c"],
            sigma=0.075,
            fraction=0.0,
            omega=-558.5,
            eloss=np.arange(-0.0, 10.0, 0.005),
            pol=[(0, 0), (0, np.pi / 2.0)],
            T=9.0,
            eline=None,
            folder=witness.folder,
        )
        rixs_all = []
        for tth, thin, L in zip(tths, thins, Ls):
            rixs_all.append(
                calc_RIXS_dimerCe(
                    tth=tth, thin=thin, HKL=[-0.5, 0.0, L], phi=10.0, **RIXS_params
                )
            )
        rixs_Mmap = dict(
            momenta=Qs,
            pol=RIXS_params["pol"],
            eloss=RIXS_params["eloss"],
            rixs=np.squeeze(rixs_all),
        )
        np.savez(witness.folder + "/pol/rixs_Mmap.npz", **rixs_Mmap)
        conjugate_rixs_all = []
        for tth, thin, L in zip(tths, thins, Ls):
            conjugate_rixs_all.append(
                conjugate_RIXS_dimerCe(
                    tth=tth, thin=thin, HKL=[-0.5, 0.0, L], phi=10.0, **RIXS_params
                )
            )
        conjugate_rixs_Mmap = dict(
            momenta=Qs,
            pol=RIXS_params["pol"],
            eloss=RIXS_params["eloss"],
            rixs=np.squeeze(conjugate_rixs_all),
        )
        np.savez(witness.folder + "/pol/conjugate_rixs_Mmap.npz", **conjugate_rixs_Mmap)
        T = 9.0
        kB = 8.6173303e-05
        beta = 1 / (kB * T)
        rixs = np.load(witness.folder + "/pol/rixs_Mmap.npz")
        rixs_con = np.load(witness.folder + "/pol/conjugate_rixs_Mmap.npz")
        rixs = dict(rixs)
        rixs1 = rixs["rixs"][:, 0, :]
        rixs2 = rixs["rixs"][:, 1, :]
        rixs_con = dict(rixs_con)
        rixs1_con = rixs_con["rixs"][:, 0]
        rixs2_con = rixs_con["rixs"][:, 1]
        eloss = rixs["eloss"]
        factor = 2 * np.tanh(beta * eloss / 2) * (1 - np.exp(-beta * eloss))
        given_axis = 1
        dim = np.ones((1, rixs1.ndim), int).ravel()
        dim[given_axis] = -1
        factor_reshaped = factor.reshape(dim)
        summand1 = rixs1 * factor_reshaped + rixs1_con * factor_reshaped
        summand2 = rixs2 * factor_reshaped + rixs2_con * factor_reshaped
        step = eloss[1] - eloss[0]
        chooser = (eloss > 0.2) * (eloss < 2.0)
        summand1_chooser = summand1[:, chooser]
        summand2_chooser = summand2[:, chooser]
        nominator1 = np.sum(summand1_chooser, axis=1) * step
        nominator2 = np.sum(summand2_chooser, axis=1) * step
        theorSpectraM0.append(nominator1 / FQ_MmapT55pol0["FQ"][0][:, 0])
        theorSpectraMpi2.append(nominator2 / FQ_MmapT55polpi2["FQ"][0][:, 0])
    theorSpectraM0_arr = np.stack(theorSpectraM0, axis=0)
    theorSpectraMpi2_arr = np.stack(theorSpectraMpi2, axis=0)
    np.savez_compressed(
        output_dir / "Ce_t2g_momdep_pol_outputsV4Vr.npz",
        theorSpectraM0=theorSpectraM0_arr,
        theorSpectraMpi2=theorSpectraMpi2_arr,
        nominator1=nominator1,
        nominator2=nominator2,
    )


if __name__ == "__main__":
    main()
