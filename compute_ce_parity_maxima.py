#!/usr/bin/env python3
"""Generate the Ce parity-sector QFI maxima used in Supplementary Fig. S4.

Run in the project's analysis environment::

    python compute_ce_parity_maxima.py --data-root /path/to/data

Inputs in DATA_ROOT:
    config_Ce_t2g.py
    Ba3CeIr2O9_test4_collectedV4V.txt
    Files required by that configuration and the project's witness module.

Required packages: NumPy, SciPy, edrixs.
Required project modules: optEDRIXS, witness, witness_utils.
Use --module-root when these modules are outside the script/data directories.

Outputs (by default in DATA_ROOT):
    Ce_t2g_maxwal_P.npz: FQ_list3, arg_list
    Ce_t2g_maxwal_M.npz: FQ_listM, arg_listM


The default uses a fresh NumPy generator for each restart, as in the notebook.
--seed uses a reproducible random stream. Existing historical values cannot
be reproduced bit for bit without their random starts and numerical environment.

Intermediate Hamiltonian, basis, and scattering-operator files are written to
WORK_DIR (default DATA_ROOT/calc_Ba3CeIr2O9). A separate checkpoint is written
after each momentum point; final output files are written only after completion
of their sector. Existing files at these paths are replaced when recomputed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize


LOGGER = logging.getLogger(__name__)
N_RESTARTS = 500
OMEGA = -558.5
OPTIMIZER_OPTIONS = {
    "maxiter": 2000,
    "ftol": 1e-12,
    "gtol": 1e-8,
    "maxls": 50,
    "maxcor": 20,
}
POSITIONS = np.array([
    [2.95175, 1.70419, 2.41068],
    [2.95175, 1.70419, 4.94682],
    [0.00000, 3.40839, 12.30432],
    [0.00000, 3.40839, 9.76818],
])
RECIPROCAL_MATRIX = np.array([
    [1.06431529, 0.614482718, -1.30341031e-16],
    [0.0, 1.22896544, -1.30341031e-16],
    [0.0, 0.0, 0.426991866],
])
FOCK_PERMUTATION = [0, 1, 6, 7, 10, 11, 16, 17,
                    2, 3, 4, 5, 8, 9, 12, 13, 14, 15, 18, 19]
LOCAL_PERMUTATION = [6, 7, 0, 1, 2, 3, 8, 9, 4, 5]


@dataclass(frozen=True)
class OperatorMoments:
    """First and second moments of one outgoing polarization channel."""

    real: np.ndarray
    real_squared: np.ndarray
    imag: np.ndarray
    imag_squared: np.ndarray


def build_parity_bases(work_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Construct the notebook's 66-state basis and signed exchange operator."""
    import edrixs
    from witness_utils import SparseBinary, get_fock_bin_by_N_S

    basis = SparseBinary(
        np.vstack([get_fock_bin_by_N_S(8, 0, 12, 10, dispinfo=False)]),
        shape=[None, 20],
    )
    basis.change_coordinates(np.asarray(FOCK_PERMUTATION), shape=[None, 20])
    fock_states = basis.toarray()

    for left, right in [(6, 4), (5, 5), (4, 6)]:
        local_basis = SparseBinary(
            np.vstack(get_fock_bin_by_N_S(8, 0, 6, left, 6, right,
                                        dispinfo=False)),
            shape=[None, 20],
        )
        local_basis.change_coordinates(np.asarray(FOCK_PERMUTATION),
                                       shape=[None, 20])
        sector_states = local_basis.toarray()
        embedding = np.zeros((len(sector_states), len(fock_states)))
        for i, state in enumerate(sector_states):
            for j, reference in enumerate(fock_states):
                embedding[i, j] = float(np.array_equal(state, reference))
        np.save(work_dir / f"states_{left}{right}.npy", embedding)

    weights = 1 << np.arange(19, -1, -1)
    fock_indices = fock_states @ weights
    local_states = np.concatenate([
        np.asarray(edrixs.get_fock_bin_by_N(6, occupancy, 4, 0),
                   dtype=np.int64)
        for occupancy in [6, 5, 4]
    ])[:, LOCAL_PERMUTATION]
    product_states = np.array([
        np.concatenate([a, b]) for a in local_states for b in local_states
    ])
    product_indices = product_states @ weights
    indices = np.array([
        np.argwhere(value == product_indices)[0] for value in fock_indices
    ]).ravel()
    positions = {int(value): i for i, value in enumerate(indices)}
    parity = np.zeros((len(indices), len(indices)))
    for i, full_index in enumerate(indices):
        a, b = divmod(int(full_index), 22)
        parity[i, positions[22 * b + a]] = 1.0
    parity *= (-1) ** np.sum(fock_states[:, :10], axis=1)
    eigenvalues, eigenvectors = np.linalg.eigh(
        (parity + parity.conj().T) * 0.5
    )
    return (eigenvectors[:, eigenvalues > 1.0 - 1e-8],
            eigenvectors[:, eigenvalues < -1.0 + 1e-8])


def initialize_hamiltonian(witness, parameters: dict[str, float]) -> None:
    """Run exact diagonalization with the notebook's fitted and fixed values."""
    p = parameters
    witness.load_params(
        tenDq=p["tenDq"], soc=p["soc"], theta=50.86, kappa=0.1, a0=0.0,
        Vdd_sigma=p["Vdd_sigma"], Vdd_pi=p["Vdd_pi"],
        Vdd_delta=p["Vdd_delta"], soc_c=p["soc_c"],
        U_dd=3.0, J_dd=(p["F2_dd"] + p["F4_dd"]) / 14.0, U_q=2.5,
        F2_dp=p["F2_dp"], G1_dp=p["G1_dp"], G3_dp=p["G3_dp"],
        F4F2_ratio=p["F4_dd"] / p["F2_dd"],
    )
    witness.load_hmat_4f(overwrite=True)
    witness.perform_ED()


def generate_scattering_operator(
    folder: Path, momentum: float, scattering_angle: float,
    incidence_angle: float, outgoing_angle: float, gamma_c: float,
) -> np.ndarray:
    """Construct one polarization-resolved RIXS operator in the ED eigenbasis."""
    from edrixs import scattering_mat, dipole_polvec_rixs
    from witness_utils import rmat_t2o

    eigenvalues = np.load(folder / "evals_i.npy")
    intermediate = [np.load(folder / "evals_n1.npy"),
                    np.load(folder / "evals_n2.npy")]
    with np.load(folder / "T_abs.npz") as data:
        absorption = [data["T_abs1"], data["T_abs2"]]
    size = len(eigenvalues)
    tensors = np.zeros((2, 3, 3, size, size), dtype=complex)
    for site, (levels, transition) in enumerate(zip(intermediate, absorption)):
        emission = np.stack([np.conj(np.transpose(t)) for t in transition])
        tensors[site] += scattering_mat(
            eigenvalues[:], levels[:], transition[:, :, :], emission,
            OMEGA, gamma_c,
        )
    operator = np.zeros((size, size), dtype=complex)
    for site in range(2):
        incoming, outgoing = dipole_polvec_rixs(
            incidence_angle / 180.0 * np.pi,
            (-incidence_angle + scattering_angle) / 180.0 * np.pi,
            10.0 / 180.0 * np.pi, 0.0, outgoing_angle,
        )
        q = np.dot([-0.5, 0.0, momentum], RECIPROCAL_MATRIX)
        incoming = np.dot(incoming, rmat_t2o(site + 1))
        outgoing = np.dot(outgoing, rmat_t2o(site + 1))
        phase = np.exp(-1j * np.dot(q, POSITIONS[site]))
        for m in range(3):
            for n in range(3):
                operator += (outgoing[m] * tensors[site, m, n]
                             * phase * incoming[n])
    return operator


def project_moments(
    operator: np.ndarray, eigenvectors: np.ndarray, sector: np.ndarray,
) -> OperatorMoments:
    """Project operator moments after squaring in the full Hilbert space."""
    real = (operator + operator.conj().T) * 0.5
    imag = (operator - operator.conj().T) * (-0.5j)
    transformed = [
        eigenvectors @ real @ eigenvectors.conj().T,
        eigenvectors @ real @ real @ eigenvectors.conj().T,
        eigenvectors @ imag @ eigenvectors.conj().T,
        eigenvectors @ imag @ imag @ eigenvectors.conj().T,
    ]
    return OperatorMoments(*[
        sector.conj().T @ matrix @ sector for matrix in transformed
    ])


def objective(parameters: np.ndarray, moments: OperatorMoments) -> float:
    """Negative four times the sum of real and imaginary operator variances."""
    dimension = len(parameters) // 2
    amplitude = parameters[:dimension]
    amplitude = amplitude / np.sqrt(np.sum(amplitude ** 2))
    state = (amplitude * np.exp(1j * parameters[dimension:])).reshape(1, -1)
    result = (state.conj() @ moments.real_squared @ state.T
              + state.conj() @ moments.imag_squared @ state.T)
    result -= ((state.conj() @ moments.real @ state.T) ** 2
               + (state.conj() @ moments.imag @ state.T) ** 2)
    return float((-4.0 * np.real(result)).item())


def objective_gradient(
    parameters: np.ndarray, moments: OperatorMoments,
) -> np.ndarray:
    """Analytic derivative for normalized amplitudes and unconstrained phases."""
    dimension = len(parameters) // 2
    raw = np.asarray(parameters[:dimension], dtype=float)
    phase = np.exp(1j * np.asarray(parameters[dimension:], dtype=float))
    norm = np.linalg.norm(raw)
    if norm == 0:
        raise ValueError("The amplitude vector must be nonzero.")
    amplitude = raw / norm
    state = amplitude * phase
    real_mean = np.vdot(state, moments.real @ state)
    imag_mean = np.vdot(state, moments.imag @ state)
    gradient = (moments.real_squared + moments.real_squared.conj().T) @ state
    gradient += (moments.imag_squared + moments.imag_squared.conj().T) @ state
    gradient += (-2 * real_mean * (moments.real @ state)
                 - 2 * np.conj(real_mean) * (moments.real.conj().T @ state))
    gradient += (-2 * imag_mean * (moments.imag @ state)
                 - 2 * np.conj(imag_mean) * (moments.imag.conj().T @ state))
    rotated = np.conj(phase) * gradient
    grad_amplitude = np.real(rotated)
    grad_phase = amplitude * np.imag(rotated)
    grad_raw = (grad_amplitude - amplitude * np.dot(amplitude, grad_amplitude)) / norm
    return -4.0 * np.concatenate([grad_raw, grad_phase])


def random_start(dimension: int, rng=None) -> np.ndarray:
    """Draw the same amplitude and phase distributions as the notebook."""
    if rng is None:
        rng = np.random.default_rng()
    amplitude = np.maximum(rng.random(dimension), 1e-12)
    phase = rng.uniform(0.0, 2 * np.pi, size=dimension)
    return np.concatenate([amplitude, phase])


def maximize_qfi(
    channels: tuple[OperatorMoments, OperatorMoments], rng=None,
    restarts: int = N_RESTARTS,
) -> tuple[float, np.ndarray]:
    """Keep the best result over random L-BFGS-B restarts."""
    dimension = channels[0].real.shape[0]

    def value(parameters):
        return objective(parameters, channels[0]) + objective(parameters, channels[1])

    def gradient(parameters):
        return (objective_gradient(parameters, channels[0])
                + objective_gradient(parameters, channels[1]))

    bounds = [(1e-10, 1)] * dimension + [(0, 2 * np.pi)] * dimension
    best = None
    for restart in range(restarts):
        result = minimize(
            fun=value, x0=random_start(dimension, rng), jac=gradient,
            method="L-BFGS-B", bounds=bounds, options=OPTIMIZER_OPTIONS.copy(),
        )
        if best is None or result.fun < best.fun:
            best = result
        if (restart + 1) % 50 == 0:
            LOGGER.info("  restart %d/%d; best QFI %.10g",
                        restart + 1, restarts, -2 * best.fun)
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("No finite optimization result was obtained.")
    return float(-best.fun * 2), best.x


def momentum_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the notebook's complete momentum and angle grids."""
    momenta = np.arange(0.0, 20.1, 0.1)
    magnitudes = np.sqrt(1.228965 ** 2 / 4 + (momenta * 0.426991866) ** 2)
    scattering = np.rad2deg(np.arcsin(1.1054837271511369 * magnitudes
                                    / 4.0 / np.pi) * 2)
    with np.errstate(divide="ignore"):
        incidence = scattering / 2 - np.rad2deg(
            np.arctan(1.228965 / 2 / momenta / 0.426991866)
        )
    return momenta, scattering, incidence


def write_results(path: Path, parity: str, maxima, parameters) -> None:
    """Save the original NPZ keys, replacing the file atomically."""
    if parity == "even":
        payload = {"FQ_list3": np.asarray(maxima), "arg_list": np.asarray(parameters)}
    else:
        payload = {"FQ_listM": np.asarray(maxima), "arg_listM": np.asarray(parameters)}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def compute_sector(
    witness, parameters: dict[str, float], sector: np.ndarray,
    parity: str, output_dir: Path, rng=None,
) -> None:
    """Prepare both polarization channels, then optimize the full L grid."""
    folder = Path(witness.folder)
    operator_dir = folder / "optimize"
    operator_dir.mkdir(parents=True, exist_ok=True)
    initialize_hamiltonian(witness, parameters)
    momenta, scattering, incidence = momentum_geometry()
    for index, (momentum, tth, thin) in enumerate(zip(momenta, scattering, incidence)):
        LOGGER.info("%s: preparing operators %d/%d (L=%.1f)",
                    parity, index + 1, len(momenta), momentum)
        for name, angle in [("pol0", 0.0), ("polpi2", np.pi / 2)]:
            operator = generate_scattering_operator(
                folder, momentum, tth, thin, angle, parameters["Gam_c"],
            )
            np.savez(operator_dir / f"FQ_Fmag_M_{name}_{str(momentum)}.npz", operator)

    eigenvectors = np.load(folder / "evecs_i.npy")
    maxima, maximizing_parameters = [], []
    tag = "P" if parity == "even" else "M"
    checkpoint = folder / f"Ce_t2g_maxwal_{tag}_checkpoint.npz"
    start = time.monotonic()
    for index, momentum in enumerate(momenta):
        LOGGER.info("%s: optimizing point %d/%d (L=%.1f)",
                    parity, index + 1, len(momenta), momentum)
        channels = []
        for name in ["pol0", "polpi2"]:
            with np.load(operator_dir / f"FQ_Fmag_M_{name}_{str(momentum)}.npz") as data:
                channels.append(project_moments(data["arr_0"], eigenvectors, sector))
        maximum, optimum = maximize_qfi(tuple(channels), rng=rng)
        maxima.append(maximum)
        maximizing_parameters.append(optimum)
        write_results(checkpoint, parity, maxima, maximizing_parameters)
        LOGGER.info("%s: completed %d/%d; QFI %.10g; elapsed %.1f min",
                    parity, index + 1, len(momenta), maximum,
                    (time.monotonic() - start) / 60)
    destination = output_dir / f"Ce_t2g_maxwal_{tag}.npz"
    write_results(destination, parity, maxima, maximizing_parameters)
    LOGGER.info("Saved %s", destination)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--module-root", type=Path, action="append", default=[],
                        help="Additional project-module directory; repeat as needed")
    parser.add_argument("--work-dir", type=Path,
                        help="Calculation directory; relative paths are resolved from DATA_ROOT")
    parser.add_argument("--output-dir", type=Path,
                        help="Final NPZ directory; relative paths are resolved from DATA_ROOT")
    parser.add_argument("--parity", choices=["both", "even", "odd"], default="both")
    parser.add_argument("--seed", type=int,
                        help="Optional nonnegative seed for reproducible random restarts")
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        parser.error("The data root must be an existing directory.")
    if args.seed is not None and args.seed < 0:
        parser.error("The random seed must be nonnegative.")
    for directory in args.module_root:
        sys.path.insert(0, str(directory.expanduser().resolve()))
    sys.path.insert(0, str(root))
    os.chdir(root)

    def directory_path(value, default):
        path = (value if value is not None else Path(default)).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    work_dir = directory_path(args.work_dir, "calc_Ba3CeIr2O9")
    output_dir = directory_path(args.output_dir, ".")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from optEDRIXS import import_config
    from witness import dimer_witness

    cfg = import_config("config_Ce_t2g.py")
    table = np.atleast_2d(np.loadtxt("Ba3CeIr2O9_test4_collectedV4V.txt"))
    labels = list(cfg.greedy_bounds.keys())
    if table.shape[1] < len(labels) or not np.all(np.isfinite(table[0, :len(labels)])):
        raise ValueError("Parameter row 0 does not match cfg.greedy_bounds.")
    parameters = {label: table[0, i] for i, label in enumerate(labels)}
    witness = dimer_witness(folder=str(work_dir))
    witness.construct_basis(10, eg_max=0)
    witness.produce_dipole()
    even, odd = build_parity_bases(work_dir)
    rng = None if args.seed is None else np.random.default_rng(args.seed)
    LOGGER.info("Using fitted row 0; %d points and %d restarts per point/sector",
                len(momentum_geometry()[0]), N_RESTARTS)
    LOGGER.info("Parity dimensions: even=%d, odd=%d", even.shape[1], odd.shape[1])
    for parity, sector in [("even", even), ("odd", odd)]:
        if args.parity in ["both", parity]:
            compute_sector(witness, parameters, sector, parity, output_dir, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
