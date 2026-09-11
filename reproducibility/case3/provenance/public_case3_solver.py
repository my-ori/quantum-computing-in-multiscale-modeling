#!/usr/bin/env python3
"""
Carleman K=2 backward-Euler LJ chain upgraded to a PennyLane shot-free VQLS layer.

The mechanics framework is kept the same as in Carleman_and_Baselines_VQLS_try1.ipynb:

  * 3-atom 1D Lennard-Jones Taylor expansion about r=a
  * ordered K=2 Carleman lifting Y=[z; vec(z⊗z)]
  * backward-Euler solve (I - dt*A) Y_{n+1} = Y_n at each time step
  * HDF5 trajectory output and optional XYZ conversion

The old Qiskit Statevector + dense classical objective has been replaced by a
PennyLane analytic VQLS objective.  For each right-hand side b, the optimizer
minimizes the direct signed-amplitude residual objective

    C(theta) = 1 - <y(theta)| A^T |b̂><b̂| A |y(theta)>
                   / <y(theta)| A^T A |y(theta)>

where A is the padded backward-Euler matrix, b̂=b/||b||, and |y(theta)> is a
real, signed-amplitude trial state in the padded Carleman space itself.  No
positive/negative doubled embedding is used.  The controlled-Ry amplitude tree
is kept, but its angles are no longer constrained to [0, pi]; therefore the
circuit can directly represent positive and negative real amplitudes.

The objective uses qml.expval(Pauli Hamiltonian) on analytic devices.  Dense
np.linalg.solve is not used by the VQLS loop; it is available only as an optional
diagnostic via --classical-check-interval.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np

try:
    import pennylane as qml
except ImportError as exc:  # pragma: no cover - user environment dependency
    raise SystemExit(
        "This upgraded Carleman VQLS script requires PennyLane. "
        "Use the same environment as pennylane_vqls_shot_free.py, e.g. "
        "PennyLane/lightning.gpu in analytic mode."
    ) from exc


# -----------------------------------------------------------------------------
# Mechanics: same Carleman K=2 1D LJ chain as the notebook
# -----------------------------------------------------------------------------


@dataclass
class CarlemanSetup:
    nm: float
    ps: float
    lbond0: float
    epsilon: float
    mass: float
    dt: float
    N: int
    istep: int
    a: float
    x0: np.ndarray
    x_initial: np.ndarray
    v_initial: np.ndarray
    sigma: float
    c1: float
    c2: float
    A: np.ndarray
    Ainfo: dict[str, object]
    d: int
    d2: int
    Y0: np.ndarray


def lennard_jones_U(r: float | np.ndarray, sigma: float, epsilon: float) -> float | np.ndarray:
    s_over_r = sigma / r
    s2 = s_over_r * s_over_r
    s6 = s2**3
    s12 = s6**2
    return 4.0 * epsilon * (s12 - s6)


def lennard_jones_deriv_at(r: float, order: int, sigma: float, epsilon: float) -> float:
    s = sigma
    e = epsilon
    if order == 1:
        return 4 * e * (-12 * s**12 / r**13 + 6 * s**6 / r**7)
    if order == 2:
        return 4 * e * (156 * s**12 / r**14 - 42 * s**6 / r**8)
    if order == 3:
        return 4 * e * (-2184 * s**12 / r**15 + 336 * s**6 / r**9)
    if order == 4:
        return 4 * e * (32760 * s**12 / r**16 - 3024 * s**6 / r**10)
    raise ValueError("order must be 1..4")


def generate_ordered_carleman_matrix_K2_dense(
    N: int,
    m_val: float,
    c1_val: float,
    c2_val: float,
    dtype: type = float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Same ordered/Kronecker K=2 Carleman matrix builder as the notebook."""
    N = int(N)
    d = 2 * N
    d2 = d * d
    m = dtype(m_val)
    c1 = dtype(c1_val)
    c2 = dtype(c2_val)

    # Incidence: T[i,i]=-1, T[i,i+1]=+1 ; S=-T^T as in the notebook.
    T = np.zeros((N - 1, N), dtype=dtype)
    for i in range(N - 1):
        T[i, i] = -1.0
        T[i, i + 1] = +1.0
    S = -T.T

    L1 = S @ T
    L2 = np.zeros((N, N * N), dtype=dtype)
    for i in range(N - 1):
        s_col = S[:, i : i + 1]
        t_row = T[i : i + 1, :]
        L2 += s_col @ np.kron(t_row, t_row)

    I_N = np.eye(N, dtype=dtype)
    zerosN = np.zeros((N, N), dtype=dtype)
    E_q = np.hstack([I_N, zerosN])
    Pqq = np.kron(E_q, E_q)

    A11 = np.zeros((d, d), dtype=dtype)
    A11[0:N, N : 2 * N] = np.eye(N, dtype=dtype)
    A11[N : 2 * N, 0:N] = (c1 / m) * L1

    A12_bottom = (c2 / m) * (L2 @ Pqq)
    A12 = np.vstack([np.zeros((N, d2), dtype=dtype), A12_bottom])

    I_d = np.eye(d, dtype=dtype)
    A22 = np.kron(I_d, A11) + np.kron(A11, I_d)

    A = np.block(
        [
            [A11, A12],
            [np.zeros((d2, d), dtype=dtype), A22],
        ]
    )
    info = {"sizes": {"N": N, "d": d, "y1": d, "y2": d2, "total": d + d2}}
    return A, info


def build_carleman_setup(
    N: int = 3,
    lbond0: float = 1.0,
    epsilon: float = 1.65,
    mass: float = 1.993,
    dt: float = 0.002,
    pulse_atoms: int = 2,
    pulse_amp: float = 0.01,
    pulse_wavelength: float = 80.0,
    pulse_phase: float = 0.0,
    istep: int = 1,
) -> CarlemanSetup:
    nm = 1.0
    ps = 1.0
    lbond0 = lbond0 * nm
    dt = dt * ps
    a = lbond0

    x0 = np.arange(N, dtype=float) * a
    x = x0.copy()

    if pulse_atoms > 0:
        if pulse_atoms == 1:
            smooth_factors = [1.0]
        else:
            smooth_factors = [
                0.5 * (1.0 + np.cos(np.pi * i / (pulse_atoms - 1)))
                for i in range(pulse_atoms)
            ]
        k_lo = 2.0 * np.pi / (pulse_wavelength * a)
        for index_in_window, smooth_factor in enumerate(smooth_factors):
            atom_index = index_in_window
            displacement = smooth_factor * pulse_amp * a * np.cos(k_lo * x0[atom_index] + pulse_phase)
            x[atom_index] += displacement

    v = np.zeros(N, dtype=float)

    sigma = a / (2.0 ** (1.0 / 6.0))
    r_eq = a
    c1 = lennard_jones_deriv_at(r_eq, 2, sigma, epsilon)
    c2 = 0.5 * lennard_jones_deriv_at(r_eq, 3, sigma, epsilon)

    A, Ainfo = generate_ordered_carleman_matrix_K2_dense(N, mass, c1, c2, dtype=float)
    d = int(Ainfo["sizes"]["d"])  # type: ignore[index]
    d2 = int(Ainfo["sizes"]["y2"])  # type: ignore[index]

    q0 = x - x0
    z0 = np.concatenate([q0, v])
    y2_0 = np.kron(z0, z0)
    Y0 = np.concatenate([z0, y2_0])

    return CarlemanSetup(
        nm=nm,
        ps=ps,
        lbond0=lbond0,
        epsilon=epsilon,
        mass=mass,
        dt=dt,
        N=N,
        istep=istep,
        a=a,
        x0=x0,
        x_initial=x,
        v_initial=v,
        sigma=sigma,
        c1=c1,
        c2=c2,
        A=A,
        Ainfo=Ainfo,
        d=d,
        d2=d2,
        Y0=Y0,
    )


def force_bond_K2(delta: float | np.ndarray, c1: float, c2: float) -> float | np.ndarray:
    return c1 * delta + c2 * delta**2


def potential_K2(delta: float | np.ndarray, c1: float, c2: float) -> float | np.ndarray:
    return 0.5 * c1 * delta**2 + (1.0 / 3.0) * c2 * delta**3


# -----------------------------------------------------------------------------
# PennyLane Pauli encoding utilities, following the user's PennyLane VQLS script
# -----------------------------------------------------------------------------


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def pauli_action(label: str, qb: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return basis permutation and phases for a Pauli string label."""
    basis = np.arange(dim)
    targets = basis.copy()
    phases = np.ones(dim, dtype=complex)

    for label_index, pauli in enumerate(label):
        if pauli == "I":
            continue
        qubit = qb - 1 - label_index
        mask = 1 << qubit
        bit_is_one = (basis & mask) != 0

        if pauli == "X":
            targets ^= mask
        elif pauli == "Y":
            targets ^= mask
            phases *= np.where(bit_is_one, -1.0j, 1.0j)
        elif pauli == "Z":
            phases *= np.where(bit_is_one, -1.0, 1.0)
        else:
            raise ValueError(f"Unsupported Pauli character {pauli!r} in {label!r}.")
    return targets, phases


def decompose_matrix_to_pauli_terms(
    A: np.ndarray,
    qb: int,
    coefficient_tol: float = 1e-10,
) -> tuple[list[str], np.ndarray]:
    """Encode a Hermitian matrix as a weighted Pauli sum over qb qubits."""
    A = np.asarray(A, dtype=complex)
    dim = 2**qb
    if A.shape != (dim, dim):
        raise ValueError(f"A has shape {A.shape}, but qb={qb} requires {(dim, dim)}.")

    basis = np.arange(dim)
    labels: list[str] = []
    coeffs: list[float] = []

    for paulis in product("IXYZ", repeat=qb):
        label = "".join(paulis)
        targets, phases = pauli_action(label, qb, dim)
        coeff = np.sum(phases * A[basis, targets]) / dim
        if abs(coeff) > coefficient_tol:
            coeff = complex(np.real_if_close(coeff))
            if abs(coeff.imag) > 100 * coefficient_tol:
                raise ValueError(
                    f"Non-real coefficient {coeff} found for Hermitian real matrix term {label}."
                )
            labels.append(label)
            coeffs.append(float(coeff.real))

    return labels, np.asarray(coeffs, dtype=float)


def pauli_label_to_observable(label: str):
    ops = []
    for wire, symbol in enumerate(label):
        if symbol == "I":
            continue
        if symbol == "X":
            ops.append(qml.PauliX(wire))
        elif symbol == "Y":
            ops.append(qml.PauliY(wire))
        elif symbol == "Z":
            ops.append(qml.PauliZ(wire))
        else:
            raise ValueError(f"Unsupported Pauli symbol {symbol!r} in {label!r}.")
    if not ops:
        return qml.Identity(0)
    if len(ops) == 1:
        return ops[0]
    return qml.prod(*ops)


def build_pennylane_hamiltonian(labels: Sequence[str], coeffs: np.ndarray):
    observables = [pauli_label_to_observable(label) for label in labels]
    return qml.Hamiltonian(coeffs.astype(float).tolist(), observables)


def summarize_pauli_terms(labels: Sequence[str], coeffs: np.ndarray, max_terms: int = 12) -> list[tuple[str, str]]:
    return [(label, f"{float(coeff):.8g}") for label, coeff in list(zip(labels, coeffs))[:max_terms]]


# -----------------------------------------------------------------------------
# Direct signed controlled-Ry amplitude tree
# -----------------------------------------------------------------------------


def _zero_tree_levels(size: int) -> list[list[float]]:
    """Return breadth-first zero angles for an arbitrary zero subtree."""
    if size <= 1:
        return []
    depth = int(round(math.log2(size)))
    return [[0.0] * (2**level) for level in range(depth)]


def _signed_tree_levels(block: np.ndarray, eps: float = 1e-15) -> tuple[float, list[list[float]]]:
    """Recursive signed real-amplitude state-preparation angles.

    Returns a global sign g and breadth-first angle levels such that the subtree
    circuit prepares g * block / ||block||. At every non-leaf node we choose the
    parent RY angle so the combined subtree has global sign +1, which makes the
    full tree directly prepare the requested signed vector up to the physically
    irrelevant global sign for one-dimensional subtrees.
    """
    block = np.asarray(block, dtype=float)
    size = block.size
    norm = float(np.linalg.norm(block))

    if size == 1:
        if norm <= eps:
            return 1.0, []
        # A leaf circuit prepares +1. Relative signs are pushed to the parent
        # rotation through this returned global sign.
        return (1.0 if block[0] >= 0.0 else -1.0), []

    if norm <= eps:
        return 1.0, _zero_tree_levels(size)

    midpoint = size // 2
    left = block[:midpoint]
    right = block[midpoint:]
    left_sign, left_levels = _signed_tree_levels(left, eps=eps)
    right_sign, right_levels = _signed_tree_levels(right, eps=eps)

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    cos_half = left_sign * left_norm / norm
    sin_half = right_sign * right_norm / norm
    angle = 2.0 * math.atan2(sin_half, cos_half)

    levels: list[list[float]] = [[angle]]
    for level in range(max(len(left_levels), len(right_levels))):
        row: list[float] = []
        row.extend(left_levels[level] if level < len(left_levels) else [0.0] * (2**level))
        row.extend(right_levels[level] if level < len(right_levels) else [0.0] * (2**level))
        levels.append(row)
    return 1.0, levels


def signed_state_to_tree_angles(vector: np.ndarray, qb: int, eps: float = 1e-15) -> np.ndarray:
    """Convert a real signed vector into breadth-first controlled-Ry tree angles.

    Unlike the positive-amplitude tree used in the static SPD case, these angles
    are not clipped to [0, pi]. Negative sine/cosine factors are allowed, so the
    prepared computational-basis amplitudes can be positive or negative directly.
    """
    vector = np.asarray(vector, dtype=float)
    dim = 2**qb
    if vector.shape != (dim,):
        raise ValueError(f"Initial vector has shape {vector.shape}, but qb={qb} requires {(dim,)}.")
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        raise ValueError("The signed-amplitude initial vector must be nonzero.")
    _global_sign, levels = _signed_tree_levels(vector / norm, eps=eps)
    angles = [angle for row in levels for angle in row]
    if len(angles) != dim - 1:
        raise RuntimeError("Internal error while constructing signed amplitude-tree angles.")
    return np.asarray(angles, dtype=float)


def project_tree_angles(theta: np.ndarray) -> np.ndarray:
    """Return finite real tree angles without imposing positive-amplitude bounds."""
    theta = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(theta)):
        raise ValueError("The optimizer produced a non-finite angle vector.")
    return theta


def apply_controlled_ry_tree(theta: np.ndarray, qb: int) -> None:
    expected = 2**qb - 1
    if len(theta) != expected:
        raise ValueError(f"The amplitude-tree ansatz for {qb} qubits needs {expected} parameters.")

    param_index = 0
    for level in range(qb):
        target = level
        controls = list(range(level))
        for branch in range(2**level):
            angle = theta[param_index]
            if controls:
                control_values = [bool((branch >> (level - 1 - bit)) & 1) for bit in range(level)]
                qml.ctrl(qml.RY, control=controls, control_values=control_values)(angle, wires=target)
            else:
                qml.RY(angle, wires=target)
            param_index += 1


def make_device(
    device_name: str,
    wires: int,
    seed: int | None = None,
    allow_fallback: bool = True,
):
    """Create a shot-free PennyLane device, with graceful fallback."""

    def _try_create(name: str):
        attempts: list[dict[str, object]] = []
        if seed is not None:
            attempts.append({"wires": wires, "seed": seed})
        attempts.append({"wires": wires})
        if seed is not None:
            attempts.append({"wires": wires, "shots": None, "seed": seed})
        attempts.append({"wires": wires, "shots": None})

        last_exc: Exception | None = None
        for kwargs in attempts:
            try:
                return qml.device(name, **kwargs)
            except TypeError as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Could not create PennyLane device {name!r}.")

    try:
        return _try_create(device_name)
    except Exception as exc:
        if not allow_fallback:
            raise
        for fallback in ("lightning.qubit", "default.qubit"):
            if fallback == device_name:
                continue
            try:
                warnings.warn(
                    f"Could not create PennyLane device {device_name!r} ({exc}). "
                    f"Falling back to {fallback!r} in analytic shot-free mode.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return _try_create(fallback)
            except Exception:
                pass
        raise


def normalize_probabilities(probs: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.maximum(probs, 0.0)
    total = float(np.sum(probs))
    if total <= eps:
        raise ValueError("Probability vector is numerically zero.")
    return probs / total


def real_amplitudes_from_state(state: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Return normalized real amplitudes from a PennyLane statevector."""
    state = np.asarray(state, dtype=complex).reshape(-1)
    max_imag = float(np.max(np.abs(state.imag))) if state.size else 0.0
    if max_imag > 1e-8:
        raise ValueError(
            "The direct Carleman ansatz is expected to be real-valued, but the "
            f"statevector has imaginary components up to {max_imag:.3e}."
        )
    amps = np.asarray(np.real(state), dtype=float)
    norm = float(np.linalg.norm(amps))
    if norm <= eps:
        raise ValueError("Cannot infer amplitudes from a zero statevector.")
    return amps / norm


# -----------------------------------------------------------------------------
# Optimizers from the PennyLane VQLS workflow
# -----------------------------------------------------------------------------


@dataclass
class OptimizerResult:
    x: np.ndarray
    fun: float
    nit: int
    success: bool
    message: str
    history: list[dict[str, float | int]]


def spsa_minimize(
    objective: Callable[[np.ndarray], float],
    theta0: np.ndarray,
    rng: np.random.Generator,
    maxiter: int = 100,
    a: float = 0.03,
    c: float = 0.08,
    A: float | None = None,
    alpha: float = 0.602,
    gamma: float = 0.101,
    progress_interval: int = 10,
    progress_callback: Callable[[int, np.ndarray, float, float], None] | None = None,
) -> OptimizerResult:
    theta = project_tree_angles(theta0)
    n_params = theta.size
    maxiter = int(maxiter)
    if A is None:
        A = max(1.0, 0.1 * maxiter)

    history: list[dict[str, float | int]] = []
    current_cost = float(objective(theta))
    best_theta = theta.copy()
    best_cost = current_cost
    history.append({"iteration": 0, "cost": current_cost, "best_cost": best_cost})
    if progress_callback is not None:
        progress_callback(0, theta.copy(), current_cost, best_cost)

    for k in range(1, maxiter + 1):
        ak = a / ((k + A) ** alpha)
        ck = c / (k**gamma)
        delta = rng.choice(np.array([-1.0, 1.0]), size=n_params)

        theta_plus = project_tree_angles(theta + ck * delta)
        theta_minus = project_tree_angles(theta - ck * delta)
        y_plus = float(objective(theta_plus))
        y_minus = float(objective(theta_minus))

        ghat = ((y_plus - y_minus) / (2.0 * ck)) * delta
        theta = project_tree_angles(theta - ak * ghat)

        if y_plus < best_cost:
            best_cost = y_plus
            best_theta = theta_plus.copy()
        if y_minus < best_cost:
            best_cost = y_minus
            best_theta = theta_minus.copy()

        if progress_interval > 0 and (k == 1 or k % progress_interval == 0 or k == maxiter):
            current_cost = float(objective(theta))
            if current_cost < best_cost:
                best_cost = current_cost
                best_theta = theta.copy()
            history.append({"iteration": k, "cost": current_cost, "best_cost": best_cost})
            if progress_callback is not None:
                progress_callback(k, theta.copy(), current_cost, best_cost)

    best_cost = float(objective(best_theta))
    history.append({"iteration": maxiter, "cost": best_cost, "best_cost": best_cost})
    return OptimizerResult(
        x=best_theta,
        fun=best_cost,
        nit=maxiter,
        success=True,
        message="SPSA completed.",
        history=history,
    )


def scipy_minimize(
    objective: Callable[[np.ndarray], float],
    theta0: np.ndarray,
    method: str = "L-BFGS-B",
    maxiter: int = 100,
    progress_interval: int = 10,
    progress_callback: Callable[[int, np.ndarray, float, float], None] | None = None,
) -> OptimizerResult:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("The selected optimizer requires SciPy. Install SciPy or use --optimizer spsa.") from exc

    method = method.upper()
    theta0 = project_tree_angles(theta0)
    best_theta = theta0.copy()
    best_cost = float(objective(theta0))
    history: list[dict[str, float | int]] = [{"iteration": 0, "cost": best_cost, "best_cost": best_cost}]
    if progress_callback is not None:
        progress_callback(0, theta0.copy(), best_cost, best_cost)

    iteration = 0

    def fun(theta: np.ndarray) -> float:
        nonlocal best_theta, best_cost
        theta_projected = project_tree_angles(theta)
        value = float(objective(theta_projected))
        if value < best_cost:
            best_cost = value
            best_theta = theta_projected.copy()
        return value

    def callback(xk: np.ndarray) -> None:
        nonlocal iteration, best_theta, best_cost
        iteration += 1
        theta_projected = project_tree_angles(xk)
        current_cost = float(objective(theta_projected))
        if current_cost < best_cost:
            best_cost = current_cost
            best_theta = theta_projected.copy()
        if progress_interval > 0 and (
            iteration == 1 or iteration % progress_interval == 0 or iteration == maxiter
        ):
            history.append({"iteration": iteration, "cost": current_cost, "best_cost": best_cost})
            if progress_callback is not None:
                progress_callback(iteration, theta_projected.copy(), current_cost, best_cost)

    bounds = [(-2.0 * math.pi, 2.0 * math.pi)] * theta0.size
    options: dict[str, object] = {"maxiter": int(maxiter), "disp": False}
    if method == "POWELL":
        options.update({"xtol": 1e-4, "ftol": 1e-8})
    if method == "L-BFGS-B":
        options.update(
            {
                "maxfun": max(150000, 4 * theta0.size * int(maxiter)),
                "maxls": 50,
                "ftol": 1e-12,
                "gtol": 1e-8,
            }
        )

    result = minimize(
        fun,
        theta0,
        method=method,
        bounds=bounds,
        callback=callback,
        options=options,
    )
    final_theta = project_tree_angles(result.x)
    final_cost = float(objective(final_theta))
    if final_cost < best_cost:
        best_cost = final_cost
        best_theta = final_theta.copy()
    history.append({"iteration": int(result.nit), "cost": final_cost, "best_cost": best_cost})

    return OptimizerResult(
        x=best_theta,
        fun=best_cost,
        nit=int(result.nit),
        success=bool(result.success),
        message=f"SciPy {method} completed: {result.message}",
        history=history,
    )


# -----------------------------------------------------------------------------
# PennyLane direct signed-amplitude VQLS for Carleman backward-Euler systems
# -----------------------------------------------------------------------------


@dataclass
class VQLSSolveResult:
    theta: np.ndarray
    cost: float
    alpha: float
    physical_direction: np.ndarray
    x_est: np.ndarray
    probabilities: np.ndarray
    signed_amplitudes: np.ndarray
    numerator_expectation: float
    denominator_expectation: float
    signed_overlap: float
    relative_residual: float | None
    optimizer_result: OptimizerResult
    pauli_term_count_numerator: int


class PennyLaneDirectSignedVQLS:
    """Shot-free PennyLane VQLS for real signed linear systems A x = b.

    The quantum ansatz is a controlled-Ry amplitude tree whose angles are allowed
    outside [0, pi]. Its computational-basis amplitudes are therefore used
    directly as the signed physical trial direction y(theta), matching the
    manuscript's Case III direct signed-amplitude formulation. No doubled
    positive/negative lifting is applied.
    """

    def __init__(
        self,
        A: np.ndarray,
        device_name: str = "lightning.gpu",
        seed: int | None = 7,
        coefficient_tol: float = 1e-10,
        allow_device_fallback: bool = True,
    ) -> None:
        self.A = np.asarray(A, dtype=float)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be a square matrix.")
        self.physical_dim = self.A.shape[0]
        if self.physical_dim & (self.physical_dim - 1):
            raise ValueError("A dimension must be a power of two for direct amplitude encoding.")

        self.qb = int(round(math.log2(self.physical_dim)))
        if 2**self.qb != self.physical_dim:
            raise ValueError("Direct signed ansatz dimension must be a power of two.")
        self.coefficient_tol = coefficient_tol
        self.device_name = device_name
        self.seed = seed

        self.dev_expvals = make_device(device_name, self.qb, seed, allow_fallback=allow_device_fallback)
        self.dev_state = make_device(
            device_name,
            self.qb,
            None if seed is None else seed + 17,
            allow_fallback=allow_device_fallback,
        )

        H_den = self.A.T @ self.A
        self.den_labels, self.den_coeffs = decompose_matrix_to_pauli_terms(
            H_den, self.qb, coefficient_tol=coefficient_tol
        )
        self.den_hamiltonian = build_pennylane_hamiltonian(self.den_labels, self.den_coeffs)

        def state_circuit(theta):
            apply_controlled_ry_tree(theta, self.qb)
            return qml.state()

        try:
            self.state_qnode = qml.QNode(state_circuit, self.dev_state, diff_method=None)
        except TypeError:
            self.state_qnode = qml.QNode(state_circuit, self.dev_state)

    def make_initial_theta(self, physical_guess: np.ndarray) -> np.ndarray:
        return project_tree_angles(signed_state_to_tree_angles(physical_guess, self.qb))

    def signed_amplitudes(self, theta: np.ndarray) -> np.ndarray:
        theta = project_tree_angles(theta)
        return real_amplitudes_from_state(self.state_qnode(theta))

    def physical_direction(self, theta: np.ndarray) -> np.ndarray:
        return self.signed_amplitudes(theta)

    def _build_numerator_hamiltonian(self, b: np.ndarray) -> tuple[object, int, np.ndarray, np.ndarray]:
        b = np.asarray(b, dtype=float)
        b_norm = np.linalg.norm(b)
        if b_norm <= 1e-15:
            raise ValueError("Right-hand side b is numerically zero.")
        b_hat = b / b_norm
        g = self.A.T @ b_hat
        H_num = np.outer(g, g)
        num_labels, num_coeffs = decompose_matrix_to_pauli_terms(
            H_num, self.qb, coefficient_tol=self.coefficient_tol
        )
        return build_pennylane_hamiltonian(num_labels, num_coeffs), len(num_labels), b_hat, g

    def solve(
        self,
        b: np.ndarray,
        theta0: np.ndarray,
        optimizer: str = "scipy-lbfgsb",
        maxiter: int = 100,
        rng: np.random.Generator | None = None,
        spsa_a: float = 0.03,
        spsa_c: float = 0.08,
        progress_interval: int = 10,
        step_label: str = "",
        compute_residual: bool = True,
    ) -> VQLSSolveResult:
        b = np.asarray(b, dtype=float)
        if b.shape != (self.physical_dim,):
            raise ValueError(f"b has shape {b.shape}; expected {(self.physical_dim,)}")
        if rng is None:
            rng = np.random.default_rng(self.seed)

        num_hamiltonian, num_term_count, _b_hat, _g = self._build_numerator_hamiltonian(b)

        def expval_circuit(theta):
            apply_controlled_ry_tree(theta, self.qb)
            return [qml.expval(num_hamiltonian), qml.expval(self.den_hamiltonian)]

        try:
            expval_qnode = qml.QNode(expval_circuit, self.dev_expvals, diff_method=None)
        except TypeError:
            expval_qnode = qml.QNode(expval_circuit, self.dev_expvals)

        def metrics_from_expvals(theta: np.ndarray) -> tuple[float, float, float]:
            theta = project_tree_angles(theta)
            values = np.asarray(expval_qnode(theta), dtype=float)
            numerator = float(values[0])
            denominator = float(values[1])
            if denominator <= 1e-14:
                return 1.0e6 + float((1e-14 - denominator) ** 2), numerator, denominator
            ratio = numerator / denominator
            # Cauchy-Schwarz gives ratio <= 1 analytically. Keep small numerical overshoots harmless.
            if ratio > 1.0 and ratio < 1.0 + 1e-8:
                ratio = 1.0
            cost = 1.0 - ratio
            return float(cost), numerator, denominator

        def objective(theta: np.ndarray) -> float:
            return metrics_from_expvals(theta)[0]

        progress_records: list[dict[str, float | int]] = []

        def progress_callback(iteration: int, theta: np.ndarray, current_cost: float, best_cost: float) -> None:
            cost_val, numerator, denominator = metrics_from_expvals(theta)
            progress_records.append(
                {
                    "iteration": int(iteration),
                    "cost": float(cost_val),
                    "best_cost": float(best_cost),
                    "numerator": float(numerator),
                    "denominator": float(denominator),
                }
            )
            if progress_interval > 0 and (
                iteration == 0 or iteration == 1 or iteration % progress_interval == 0
            ):
                prefix = f"{step_label} " if step_label else ""
                print(
                    f"{prefix}iter {iteration}: cost={cost_val:.8e}, "
                    f"best={best_cost:.8e}, num={numerator:.8e}, den={denominator:.8e}"
                )

        optimizer_key = optimizer.lower().replace("_", "-")
        if optimizer_key == "spsa":
            opt = spsa_minimize(
                objective=objective,
                theta0=theta0,
                rng=rng,
                maxiter=maxiter,
                a=spsa_a,
                c=spsa_c,
                progress_interval=progress_interval,
                progress_callback=progress_callback,
            )
        elif optimizer_key == "scipy-powell":
            opt = scipy_minimize(
                objective=objective,
                theta0=theta0,
                method="Powell",
                maxiter=maxiter,
                progress_interval=progress_interval,
                progress_callback=progress_callback,
            )
        elif optimizer_key == "scipy-lbfgsb":
            opt = scipy_minimize(
                objective=objective,
                theta0=theta0,
                method="L-BFGS-B",
                maxiter=maxiter,
                progress_interval=progress_interval,
                progress_callback=progress_callback,
            )
        else:
            raise ValueError("optimizer must be 'spsa', 'scipy-powell', or 'scipy-lbfgsb'.")
        opt.history.extend(progress_records)

        theta_opt = project_tree_angles(opt.x)
        cost, numerator, denominator = metrics_from_expvals(theta_opt)
        signed_amps = self.signed_amplitudes(theta_opt)
        probs = normalize_probabilities(np.abs(signed_amps) ** 2)
        y_dir = signed_amps

        Ay = self.A @ y_dir
        den_classical = float(Ay @ Ay)
        signed_overlap = float(Ay @ b)
        if den_classical <= 1e-14:
            alpha = 0.0
            x_est = np.zeros_like(y_dir)
        else:
            alpha = signed_overlap / den_classical
            x_est = alpha * y_dir

        residual = None
        if compute_residual:
            b_norm = np.linalg.norm(b)
            residual = float(np.linalg.norm(self.A @ x_est - b) / max(b_norm, 1e-15))

        return VQLSSolveResult(
            theta=theta_opt,
            cost=float(cost),
            alpha=float(alpha),
            physical_direction=y_dir,
            x_est=x_est,
            probabilities=probs,
            signed_amplitudes=signed_amps,
            numerator_expectation=float(numerator),
            denominator_expectation=float(denominator),
            signed_overlap=signed_overlap,
            relative_residual=residual,
            optimizer_result=opt,
            pauli_term_count_numerator=num_term_count,
        )

# -----------------------------------------------------------------------------
# Integration and output
# -----------------------------------------------------------------------------


def pad_backward_euler_matrix(M: np.ndarray, qdim: int, mode: str = "identity") -> np.ndarray:
    """Pad M to qdim x qdim.

    mode='identity' is preferred: dummy basis states satisfy x_dummy=0 because b_dummy=0.
    mode='zero' reproduces the original notebook's zero-padded matrix but leaves a nullspace.
    """
    n = M.shape[0]
    if qdim < n:
        raise ValueError("qdim must be >= matrix dimension.")
    if qdim == n:
        return M.copy()
    out = np.zeros((qdim, qdim), dtype=M.dtype)
    out[:n, :n] = M
    if mode == "identity":
        out[n:, n:] = np.eye(qdim - n, dtype=M.dtype)
    elif mode == "zero":
        pass
    else:
        raise ValueError("padding mode must be 'identity' or 'zero'.")
    return out


def pad_vector(v: np.ndarray, qdim: int) -> np.ndarray:
    out = np.zeros(qdim, dtype=float)
    out[: len(v)] = v
    return out


def make_initial_physical_guess(dim: int, mode: str, rhs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mode = mode.lower()
    if mode == "rhs":
        return np.asarray(rhs, dtype=float)
    if mode == "index_ramp":
        return np.linspace(1.0, float(dim), dim, dtype=float)
    if mode == "uniform":
        return np.ones(dim, dtype=float)
    if mode == "random":
        return rng.standard_normal(dim)
    raise ValueError("initial_state_guess must be one of: rhs, index_ramp, uniform, random.")


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-30:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def write_xyz_frames(src_h5: Path, outdir: Path, element: str = "C") -> None:
    outdir.mkdir(exist_ok=True, parents=True)
    with h5py.File(src_h5) as h5:
        X = h5["x"][:]
        Y = h5["y"][:]
        T = h5["t"][:]
    N = X.shape[1]
    for k in range(X.shape[0]):
        filename = outdir / f"wave_{k:04d}.xyz"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{N}\n")
            f.write(f"t = {T[k]*1e12:.4f} ps\n")
            for x_i, y_i in zip(X[k], Y[k]):
                f.write(f"{element:2s}  {x_i*1:15.6f}  {y_i*1e3:15.6f}  0.0\n")


def solve_carleman_with_pennylane_vqls(
    n_steps: int = 200,
    output_prefix: str = "3atom_LJ_newunit_Taylor_Carleman_PennyLaneVQLS",
    N: int = 3,
    lbond0: float = 1.0,
    epsilon: float = 1.65,
    mass: float = 1.993,
    dt: float = 0.002,
    pulse_atoms: int = 2,
    pulse_amp: float = 0.01,
    pulse_wavelength: float = 80.0,
    pulse_phase: float = 0.0,
    istep: int = 1,
    device: str = "lightning.gpu",
    optimizer: str = "scipy-lbfgsb",
    maxiter: int = 100,
    initial_state_guess: str = "rhs",
    coefficient_tol: float = 1e-10,
    rng_seed: int = 7,
    spsa_a: float = 0.03,
    spsa_c: float = 0.08,
    progress_interval: int = 10,
    classical_check_interval: int = 0,
    padding_mode: str = "identity",
    allow_device_fallback: bool = True,
    write_xyz: bool = False,
) -> dict[str, object]:
    setup = build_carleman_setup(
        N=N,
        lbond0=lbond0,
        epsilon=epsilon,
        mass=mass,
        dt=dt,
        pulse_atoms=pulse_atoms,
        pulse_amp=pulse_amp,
        pulse_wavelength=pulse_wavelength,
        pulse_phase=pulse_phase,
        istep=istep,
    )

    print("PennyLane version:", qml.__version__)
    print("VQLS level: shot-free analytic PennyLane Pauli-Hamiltonian expectations")
    print("Mechanics case: 1D LJ Carleman K=2 backward Euler")
    print(f"Atoms: {setup.N}; lifted dimension: {len(setup.Y0)}")
    print(f"Taylor force coefficients: c1={setup.c1:.8e}, c2={setup.c2:.8e}")

    M = np.eye(setup.A.shape[0]) - setup.dt * setup.A
    n_orig = len(setup.Y0)
    qdim = next_pow2(n_orig)
    M_pad = pad_backward_euler_matrix(M, qdim, mode=padding_mode)
    qb_physical = int(round(math.log2(qdim)))

    print(f"Backward-Euler matrix dimension: {M.shape[0]}")
    print(f"Padded physical dimension: {qdim} = 2**{qb_physical}")
    print(f"Direct signed quantum dimension: {qdim} = 2**{qb_physical}")
    print("Device requested:", device)
    print("Optimizer:", optimizer)
    print("Initial state guess:", initial_state_guess)
    print("Padding mode:", padding_mode)
    print("Shots: analytic mode; no finite shot count is set")

    solver = PennyLaneDirectSignedVQLS(
        A=M_pad,
        device_name=device,
        seed=rng_seed,
        coefficient_tol=coefficient_tol,
        allow_device_fallback=allow_device_fallback,
    )
    print(f"Denominator Pauli encoding uses {len(solver.den_labels)} terms.")
    print("First denominator terms:")
    for label, coeff_text in summarize_pauli_terms(solver.den_labels, solver.den_coeffs, max_terms=8):
        print(f"  {label}: {coeff_text}")

    rng = np.random.default_rng(rng_seed)
    Y = setup.Y0.copy()
    first_rhs = pad_vector(Y / np.linalg.norm(Y), qdim)
    theta0 = solver.make_initial_theta(make_initial_physical_guess(qdim, initial_state_guess, first_rhs, rng))

    out_h5 = Path(output_prefix).with_suffix(".h5")
    diagnostics_path = Path(output_prefix).with_name(Path(output_prefix).name + "_diagnostics.npz")

    diagnostics: list[dict[str, float | int | bool | str | None]] = []

    with h5py.File(out_h5, "w") as h5:
        dset_x = h5.create_dataset("x", (n_steps // istep + 1, setup.N), dtype=np.float64)
        dset_y = h5.create_dataset("y", (n_steps // istep + 1, setup.N), dtype=np.float64)
        dset_t = h5.create_dataset("t", (n_steps // istep + 1,), dtype=np.float64)

        dset_x[0] = setup.x_initial
        dset_y[0] = setup.x_initial - setup.x0
        dset_t[0] = 0.0
        frame = 0

        for step in range(1, n_steps + 1):
            norm_Y = np.linalg.norm(Y)
            if norm_Y <= 1e-15:
                raise ValueError(f"Y became numerically zero at step {step}.")
            b_orig = Y / norm_Y
            b_pad = pad_vector(b_orig, qdim)

            result = solver.solve(
                b=b_pad,
                theta0=theta0,
                optimizer=optimizer,
                maxiter=maxiter,
                rng=rng,
                spsa_a=spsa_a,
                spsa_c=spsa_c,
                progress_interval=progress_interval,
                step_label=f"step {step}",
                compute_residual=True,
            )
            theta0 = result.theta
            Y_vqls_normalized = result.x_est[:n_orig]
            Y = Y_vqls_normalized * norm_Y

            classical_relative_error = None
            r2_q = None
            if classical_check_interval > 0 and (
                step == 1 or step % classical_check_interval == 0 or step == n_steps
            ):
                Y_class_normalized = np.linalg.solve(M, b_orig)
                Y_class = Y_class_normalized * norm_Y
                classical_relative_error = float(
                    np.linalg.norm(Y - Y_class) / max(np.linalg.norm(Y_class), 1e-15)
                )
                r2_q = r2_score_np(Y_class[: setup.N], Y[: setup.N])
                print(f"step {step}: classical relative error={classical_relative_error:.8e}, R²(q)={r2_q:.8e}")

            z = Y[: setup.d]
            q = z[: setup.N]
            v = z[setup.N :]
            x = setup.x0 + q

            if (step % istep) == 0:
                frame += 1
                dset_x[frame] = setup.x0
                dset_y[frame] = q
                dset_t[frame] = step * setup.dt

            diagnostics.append(
                {
                    "step": int(step),
                    "time": float(step * setup.dt),
                    "vqls_cost": float(result.cost),
                    "alpha": float(result.alpha),
                    "numerator_expectation": float(result.numerator_expectation),
                    "denominator_expectation": float(result.denominator_expectation),
                    "signed_overlap": float(result.signed_overlap),
                    "relative_residual_padded": result.relative_residual,
                    "classical_relative_error": classical_relative_error,
                    "r2_q": r2_q,
                    "optimizer_success": bool(result.optimizer_result.success),
                    "optimizer_iterations": int(result.optimizer_result.nit),
                    "numerator_pauli_terms": int(result.pauli_term_count_numerator),
                }
            )
            print(
                f"step {step}: cost={result.cost:.8e}, residual={result.relative_residual:.8e}, "
                f"alpha={result.alpha:.8e}, q={q}"
            )

    metadata = {
        "vqls_level": "shot-free analytic PennyLane Pauli-Hamiltonian VQLS",
        "ansatz_type": "direct signed controlled-Ry real-amplitude tree",
        "pennylane_version": qml.__version__,
        "device_requested": device,
        "optimizer": optimizer,
        "maxiter": maxiter,
        "coefficient_tol": coefficient_tol,
        "rng_seed": rng_seed,
        "n_steps": n_steps,
        "N": setup.N,
        "dt": setup.dt,
        "c1": setup.c1,
        "c2": setup.c2,
        "n_orig": n_orig,
        "qdim": qdim,
        "direct_trial_dim": qdim,
        "qb_physical": qb_physical,
        "qb_quantum": qb_physical,
        "denominator_pauli_terms": len(solver.den_labels),
        "padding_mode": padding_mode,
        "classical_check_interval": classical_check_interval,
    }

    # Save compact diagnostics as JSON inside NPZ for convenient reloading.
    np.savez_compressed(
        diagnostics_path,
        A=setup.A,
        M=M,
        M_pad=M_pad,
        Y_final=Y,
        x0=setup.x0,
        x_final=x,
        metadata=json.dumps(metadata, indent=2),
        diagnostics=json.dumps(diagnostics, indent=2),
        theta_final=theta0,
    )
    print(f"Saved HDF5 trajectory to {out_h5}")
    print(f"Saved diagnostics to {diagnostics_path}")

    xyz_dir = None
    if write_xyz:
        xyz_dir = Path(output_prefix).with_suffix("")
        write_xyz_frames(out_h5, xyz_dir)
        print(f"Saved XYZ frames to {xyz_dir}")

    return {
        "metadata": metadata,
        "diagnostics": diagnostics,
        "h5_path": str(out_h5),
        "diagnostics_path": str(diagnostics_path),
        "xyz_dir": None if xyz_dir is None else str(xyz_dir),
        "Y_final": Y,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PennyLane shot-free VQLS upgrade for the Carleman K=2 LJ backward-Euler notebook."
    )
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--output-prefix", type=str, default="3atom_LJ_newunit_Taylor_Carleman_PennyLaneVQLS")

    parser.add_argument("--atoms", type=int, default=3)
    parser.add_argument("--lbond0", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1.65)
    parser.add_argument("--mass", type=float, default=1.993)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--pulse-atoms", type=int, default=2)
    parser.add_argument("--pulse-amp", type=float, default=0.01)
    parser.add_argument("--pulse-wavelength", type=float, default=80.0)
    parser.add_argument("--pulse-phase", type=float, default=0.0)
    parser.add_argument("--istep", type=int, default=1)

    parser.add_argument("--device", type=str, default="lightning.gpu")
    parser.add_argument(
        "--energy-estimator",
        type=str,
        default="pauli_hamiltonian",
        choices=["pauli_hamiltonian"],
        help="Kept for command-line consistency with pennylane_vqls_shot_free.py.",
    )
    parser.add_argument("--coefficient-tol", type=float, default=1e-10)
    parser.add_argument("--no-device-fallback", action="store_true")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="scipy-lbfgsb",
        choices=["spsa", "scipy-powell", "scipy-lbfgsb"],
    )
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--spsa-a", type=float, default=0.03)
    parser.add_argument("--spsa-c", type=float, default=0.08)
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument(
        "--initial-state-guess",
        type=str,
        default="rhs",
        choices=["rhs", "index_ramp", "uniform", "random"],
    )
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--classical-check-interval",
        type=int,
        default=0,
        help="0 disables np.linalg.solve diagnostics; 1 checks every step.",
    )
    parser.add_argument(
        "--padding-mode",
        type=str,
        default="identity",
        choices=["identity", "zero"],
        help="identity is recommended; zero reproduces the original nullspace padding.",
    )
    parser.add_argument("--write-xyz", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solve_carleman_with_pennylane_vqls(
        n_steps=args.n_steps,
        output_prefix=args.output_prefix,
        N=args.atoms,
        lbond0=args.lbond0,
        epsilon=args.epsilon,
        mass=args.mass,
        dt=args.dt,
        pulse_atoms=args.pulse_atoms,
        pulse_amp=args.pulse_amp,
        pulse_wavelength=args.pulse_wavelength,
        pulse_phase=args.pulse_phase,
        istep=args.istep,
        device=args.device,
        optimizer=args.optimizer,
        maxiter=args.maxiter,
        initial_state_guess=args.initial_state_guess,
        coefficient_tol=args.coefficient_tol,
        rng_seed=args.rng_seed,
        spsa_a=args.spsa_a,
        spsa_c=args.spsa_c,
        progress_interval=args.progress_interval,
        classical_check_interval=args.classical_check_interval,
        padding_mode=args.padding_mode,
        allow_device_fallback=not args.no_device_fallback,
        write_xyz=args.write_xyz,
    )


if __name__ == "__main__":
    main()
