#!/usr/bin/env python3
"""
BCOW + VQLS PennyLane simulation for the 1D dynamic MD-FE ODE problem.

This script is the next quantum-simulation layer after 1Ddynamic_BCOW_ODE_solver.py:

    y_dot = A y
        -> BCOW sparse linear system C |history> = |rhs>
        -> PennyLane VQLS residual minimization for C |x> = |rhs>

The VQLS part is intentionally more general than the static SPD VQLS script.
The static Kff u = ff problem can use a positive-amplitude tree and an energy
functional.  The BCOW history vector can contain signs/phases and C is generally
non-Hermitian, so this file uses the standard normalized VQLS residual objective

    cost(theta) = 1 - |<b_hat| A_target |psi(theta)>|^2
                    / <psi(theta)| A_target^dagger A_target |psi(theta)>,

where b_hat = b / ||b||.  The optimized state |psi(theta)> gives the direction
of the solution and the scalar is recovered classically as

    alpha = <A psi | b> / <A psi | A psi>,       x = alpha psi.

For direct BCOW solving A_target = C.  For the optional Hermitian-dilation mode,
A_target = [[0, C], [C^dagger, 0]] and b = [rhs, 0]; the BCOW history is then the
lower half of the optimized solution.

Important simulator limitation:
    This is a circuit-level, shot-free PennyLane VQLS simulation using Pauli
    Hamiltonian expectation values for A^dagger A and A^dagger |b_hat><b_hat| A.
    The Pauli expansion scales as O(4^q), so this file is for small BCOW
    instances and algorithm validation.  For large production BCOW systems, use
    a block-encoding/local-cost VQLS or QLSA/QSVT implementation instead of
    dense Pauli expansion.

Example smoke test:
    python 1Ddynamic_BCOW_VQLS_pennylane.py \
        --n1 3 --n2 1 --t 0.5 \
        --segments 1 --taylor-order 3 --padding-steps 1 \
        --target direct-c \
        --device lightning.gpu \
        --energy-estimator pauli_hamiltonian \
        --optimizer scipy-lbfgsb \
        --initial-state-guess index_ramp \
        --layers 3 --maxiter 50 \
        --output-prefix bcow_vqls_small

Hermitian-dilation smoke test:
    python 1Ddynamic_BCOW_VQLS_pennylane.py \
        --n1 2 --n2 1 --t 0.5 \
        --segments 1 --taylor-order 3 --padding-steps 1 \
        --target hermitian-dilation \
        --device lightning.gpu \
        --optimizer spsa --maxiter 100 \
        --output-prefix bcow_vqls_dilation_small
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
import warnings
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires SciPy. Install with: pip install scipy") from exc

try:
    import pennylane as qml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires PennyLane. The target environment is PennyLane 0.45+ "
        "with lightning.gpu/lightning.qubit/default.qubit available."
    ) from exc


# -----------------------------------------------------------------------------
# Dynamic import of the uploaded BCOW construction
# -----------------------------------------------------------------------------


def load_bcow_module(path: str | Path):
    """Load 1Ddynamic_BCOW_ODE_solver.py despite the leading digit in the file name."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find BCOW module at {path}")
    spec = importlib.util.spec_from_file_location("bcow_dynamic_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# -----------------------------------------------------------------------------
# Generic Pauli decomposition helpers
# -----------------------------------------------------------------------------


def pauli_action(label: str, qb: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return target basis indices and phases for a Pauli label.

    The leftmost Pauli character acts on PennyLane wire 0, the most significant
    bit in qml.probs/qml.state computational-basis ordering.
    """
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
        else:  # pragma: no cover
            raise ValueError(f"Unsupported Pauli character {pauli!r} in {label!r}")
    return targets, phases


def decompose_hermitian_to_pauli_terms(
    H: np.ndarray,
    qb: int,
    coefficient_tol: float = 1e-10,
) -> tuple[list[str], np.ndarray]:
    """Encode a Hermitian 2**qb x 2**qb matrix as sum_i coeff_i P_i."""
    H = np.asarray(H, dtype=complex)
    dim = 2**qb
    if H.shape != (dim, dim):
        raise ValueError(f"H has shape {H.shape}, but qb={qb} requires {(dim, dim)}")
    hermitian_error = np.linalg.norm(H - H.conjugate().T) / max(np.linalg.norm(H), 1e-15)
    if hermitian_error > 1e-8:
        raise ValueError(f"Matrix is not Hermitian enough for Pauli decomposition: {hermitian_error:.3e}")

    basis = np.arange(dim)
    labels: list[str] = []
    coeffs: list[float] = []

    for paulis in product("IXYZ", repeat=qb):
        label = "".join(paulis)
        targets, phases = pauli_action(label, qb, dim)
        coeff = np.sum(phases * H[basis, targets]) / dim
        coeff = complex(np.real_if_close(coeff, tol=1000))
        if abs(coeff) > coefficient_tol:
            if abs(coeff.imag) > max(100 * coefficient_tol, 1e-10):
                raise ValueError(
                    f"Hermitian Pauli coefficient for {label} has non-negligible imaginary part {coeff.imag:.3e}."
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
        else:  # pragma: no cover
            raise ValueError(f"Unsupported Pauli symbol {symbol!r} in {label!r}")
    if not ops:
        return qml.Identity(0)
    if len(ops) == 1:
        return ops[0]
    return qml.prod(*ops)


def build_pennylane_hamiltonian(labels: Sequence[str], coeffs: np.ndarray):
    observables = [pauli_label_to_observable(label) for label in labels]
    return qml.Hamiltonian(np.asarray(coeffs, dtype=float).tolist(), observables)


def summarize_terms(labels: Sequence[str], coeffs: np.ndarray, max_terms: int = 12) -> list[tuple[str, str]]:
    return [(label, f"{float(coeff):.8g}") for label, coeff in list(zip(labels, coeffs, strict=False))[:max_terms]]


# -----------------------------------------------------------------------------
# PennyLane device, ansatz, and optimizers
# -----------------------------------------------------------------------------


def make_device(device_name: str, wires: int, seed: int | None = None, allow_fallback: bool = True):
    """Create a shot-free PennyLane device, with optional graceful fallback."""

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
        raise RuntimeError(f"Could not create PennyLane device {name!r}")

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


def normalized_state(vector: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= eps:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def make_initial_state(
    dim: int,
    mode: str,
    rng: np.random.Generator,
    rhs: np.ndarray,
    recurrence_history: np.ndarray | None = None,
    target: str = "direct-c",
) -> np.ndarray:
    """Fixed state preparation before the trainable ansatz."""
    mode = mode.lower().replace("-", "_")
    if mode in {"zero", "basis_zero"}:
        vec = np.zeros(dim, dtype=complex)
        vec[0] = 1.0
        return vec
    if mode == "uniform":
        return np.ones(dim, dtype=complex) / math.sqrt(dim)
    if mode == "index_ramp":
        return normalized_state(np.linspace(1.0, float(dim), dim, dtype=float).astype(complex))
    if mode in {"random", "random_complex"}:
        return normalized_state(rng.standard_normal(dim) + 1j * rng.standard_normal(dim))
    if mode == "rhs":
        return normalized_state(rhs)
    if mode in {"recurrence", "reference", "reference_solution"}:
        if recurrence_history is None:
            raise ValueError("initial_state_guess='recurrence' requires recurrence_history.")
        hist_vec = recurrence_history.reshape(-1).astype(complex)
        if target == "hermitian-dilation":
            vec = np.concatenate([np.zeros_like(hist_vec), hist_vec])
        else:
            vec = hist_vec
        if vec.shape != (dim,):
            raise ValueError("Recurrence initial state has wrong dimension for VQLS target.")
        warnings.warn(
            "initial_state_guess='recurrence' uses the classical BCOW recurrence solution as "
            "state preparation. Use only as a VQLS smoke test, not as a quantum advantage claim.",
            RuntimeWarning,
            stacklevel=2,
        )
        return normalized_state(vec)
    raise ValueError(
        "Unsupported initial_state_guess. Choose zero, uniform, index_ramp, random_complex, rhs, or recurrence."
    )


def parameter_count(ansatz: str, layers: int, n_qubits: int) -> int:
    ansatz_key = ansatz.lower().replace("-", "_")
    if ansatz_key in {"real_ry", "ry"}:
        return int(layers) * int(n_qubits)
    if ansatz_key in {"hardware_efficient", "he", "rot"}:
        return int(layers) * int(n_qubits) * 3
    raise ValueError("ansatz must be 'hardware_efficient' or 'real_ry'.")


def apply_ansatz(theta: np.ndarray, ansatz: str, layers: int, n_qubits: int) -> None:
    theta = np.asarray(theta, dtype=float)
    wires = list(range(n_qubits))
    ansatz_key = ansatz.lower().replace("-", "_")
    if ansatz_key in {"real_ry", "ry"}:
        params = theta.reshape((layers, n_qubits))
        for layer in range(layers):
            for wire in wires:
                qml.RY(params[layer, wire], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])
        return

    if ansatz_key in {"hardware_efficient", "he", "rot"}:
        params = theta.reshape((layers, n_qubits, 3))
        for layer in range(layers):
            for wire in wires:
                qml.Rot(params[layer, wire, 0], params[layer, wire, 1], params[layer, wire, 2], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])
        return

    raise ValueError("ansatz must be 'hardware_efficient' or 'real_ry'.")


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
    a: float = 0.05,
    c: float = 0.08,
    A: float | None = None,
    alpha: float = 0.602,
    gamma: float = 0.101,
    progress_interval: int = 10,
    progress_callback: Callable[[int, np.ndarray, float, float], None] | None = None,
) -> OptimizerResult:
    theta = np.asarray(theta0, dtype=float).copy()
    n_params = theta.size
    maxiter = int(maxiter)
    if A is None:
        A = max(1.0, 0.1 * maxiter)

    current_cost = float(objective(theta))
    best_theta = theta.copy()
    best_cost = current_cost
    history: list[dict[str, float | int]] = [{"iteration": 0, "cost": current_cost, "best_cost": best_cost}]
    if progress_callback is not None:
        progress_callback(0, theta.copy(), current_cost, best_cost)

    for k in range(1, maxiter + 1):
        ak = a / ((k + A) ** alpha)
        ck = c / (k**gamma)
        delta = rng.choice(np.array([-1.0, 1.0]), size=n_params)
        theta_plus = theta + ck * delta
        theta_minus = theta - ck * delta
        y_plus = float(objective(theta_plus))
        y_minus = float(objective(theta_minus))
        ghat = ((y_plus - y_minus) / (2.0 * ck)) * delta
        theta = theta - ak * ghat

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

    final_cost = float(objective(best_theta))
    return OptimizerResult(
        x=best_theta,
        fun=final_cost,
        nit=maxiter,
        success=True,
        message="SPSA completed.",
        history=history,
    )


def scipy_minimize(
    objective: Callable[[np.ndarray], float],
    theta0: np.ndarray,
    method: str,
    maxiter: int,
    progress_interval: int,
    progress_callback: Callable[[int, np.ndarray, float, float], None] | None = None,
) -> OptimizerResult:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("SciPy optimize is required for scipy-powell/scipy-lbfgsb.") from exc

    theta0 = np.asarray(theta0, dtype=float)
    method_name = "L-BFGS-B" if method.lower().replace("_", "-") in {"scipy-lbfgsb", "lbfgsb", "l-bfgs-b"} else "Powell"
    best_theta = theta0.copy()
    best_cost = float(objective(theta0))
    history: list[dict[str, float | int]] = [{"iteration": 0, "cost": best_cost, "best_cost": best_cost}]
    if progress_callback is not None:
        progress_callback(0, theta0.copy(), best_cost, best_cost)

    call_count = 0
    callback_count = 0

    def fun(theta: np.ndarray) -> float:
        nonlocal best_theta, best_cost, call_count
        call_count += 1
        value = float(objective(theta))
        if value < best_cost:
            best_cost = value
            best_theta = np.asarray(theta, dtype=float).copy()
        return value

    def callback(xk: np.ndarray) -> None:
        nonlocal callback_count, best_theta, best_cost
        callback_count += 1
        current_cost = float(objective(xk))
        if current_cost < best_cost:
            best_cost = current_cost
            best_theta = np.asarray(xk, dtype=float).copy()
        if progress_interval > 0 and (
            callback_count == 1 or callback_count % progress_interval == 0 or callback_count == maxiter
        ):
            history.append({"iteration": callback_count, "cost": current_cost, "best_cost": best_cost})
            if progress_callback is not None:
                progress_callback(callback_count, np.asarray(xk, dtype=float).copy(), current_cost, best_cost)

    options: dict[str, object] = {"maxiter": int(maxiter), "disp": False}
    if method_name == "Powell":
        options.update({"xtol": 1e-4, "ftol": 1e-8})
    elif method_name == "L-BFGS-B":
        options.update({
            "maxfun": max(20000, 4 * len(theta0) * int(maxiter)),
            "maxls": 50,
            "ftol": 1e-12,
            "gtol": 1e-8,
        })

    result = minimize(fun, theta0, method=method_name, callback=callback, options=options)
    final_theta = np.asarray(result.x, dtype=float)
    final_cost = float(objective(final_theta))
    if final_cost < best_cost:
        best_cost = final_cost
        best_theta = final_theta.copy()
    history.append({"iteration": int(getattr(result, "nit", callback_count)), "cost": final_cost, "best_cost": best_cost})
    return OptimizerResult(
        x=best_theta,
        fun=best_cost,
        nit=int(getattr(result, "nit", callback_count)),
        success=bool(result.success),
        message=f"SciPy {method_name} completed after {call_count} objective calls: {result.message}",
        history=history,
    )


# -----------------------------------------------------------------------------
# VQLS estimator for a generic linear system A_target x = rhs
# -----------------------------------------------------------------------------


@dataclass
class VQLSTarget:
    target: str
    A_target: sp.csr_matrix
    rhs_target: np.ndarray
    history_dim: int
    history_slice: slice
    n_qubits: int
    dim: int


@dataclass
class VQLSDiagnostics:
    final_cost: float
    sqrt_cost: float
    denominator_expectation: float
    numerator_expectation: float
    alpha_real: float
    alpha_imag: float
    relative_residual: float | None
    relative_history_error_vs_direct: float | None
    relative_final_error_vs_expm: float | None
    final_state_distance_vs_expm: float | None
    final_overlap_abs_vs_expm: float | None
    pauli_terms_denominator: int
    pauli_terms_numerator: int
    optimizer_success: bool
    optimizer_message: str
    optimizer_iterations: int
    wall_time_seconds: float


def build_vqls_target(C: sp.spmatrix, rhs: np.ndarray, target: str, bcow_module) -> VQLSTarget:
    target_key = target.lower().replace("_", "-")
    C = sp.csr_matrix(C, dtype=complex)
    rhs = np.asarray(rhs, dtype=complex)
    if target_key == "direct-c":
        dim = C.shape[0]
        rhs_target = rhs
        A_target = C
        history_slice = slice(0, dim)
    elif target_key == "hermitian-dilation":
        A_target = bcow_module.build_hermitian_dilation(C)
        rhs_target = bcow_module.build_hermitian_rhs(rhs)
        dim = A_target.shape[0]
        history_slice = slice(C.shape[0], 2 * C.shape[0])
    else:
        raise ValueError("target must be 'direct-c' or 'hermitian-dilation'.")
    if dim & (dim - 1):
        raise ValueError(f"VQLS target dimension {dim} is not a power of two.")
    return VQLSTarget(
        target=target_key,
        A_target=sp.csr_matrix(A_target, dtype=complex),
        rhs_target=np.asarray(rhs_target, dtype=complex),
        history_dim=C.shape[0],
        history_slice=history_slice,
        n_qubits=int(math.log2(dim)),
        dim=dim,
    )


def build_residual_hamiltonians(
    A_target: sp.spmatrix,
    rhs_target: np.ndarray,
    coefficient_tol: float,
    max_qubits: int,
) -> tuple[object, object, dict[str, object]]:
    """Build PennyLane Hamiltonians for denominator and numerator of VQLS residual."""
    A = sp.csr_matrix(A_target, dtype=complex)
    b = normalized_state(rhs_target)
    dim = A.shape[0]
    n_qubits = int(math.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(f"Target dimension {dim} is not a power of two.")
    if n_qubits > max_qubits:
        raise ValueError(
            f"This Pauli-expanded simulator would require decomposing matrices on {n_qubits} qubits. "
            f"The current --max-vqls-qubits is {max_qubits}. Increase it only for small validation runs; "
            "for the full BCOW system use a block-encoded/local-cost VQLS or QLSA/QSVT route."
        )

    t0 = time.perf_counter()
    H_den_sparse = (A.conjugate().transpose() @ A).tocsr()
    v = A.conjugate().transpose() @ b
    H_num_dense = np.outer(v, v.conjugate())
    H_den_dense = H_den_sparse.toarray()

    labels_den, coeffs_den = decompose_hermitian_to_pauli_terms(H_den_dense, n_qubits, coefficient_tol)
    labels_num, coeffs_num = decompose_hermitian_to_pauli_terms(H_num_dense, n_qubits, coefficient_tol)

    H_den_qml = build_pennylane_hamiltonian(labels_den, coeffs_den)
    H_num_qml = build_pennylane_hamiltonian(labels_num, coeffs_num)

    metadata = {
        "n_qubits": n_qubits,
        "dim": dim,
        "pauli_terms_denominator": len(labels_den),
        "pauli_terms_numerator": len(labels_num),
        "coefficient_tol": coefficient_tol,
        "build_seconds": time.perf_counter() - t0,
        "denominator_terms_preview": summarize_terms(labels_den, coeffs_den, max_terms=10),
        "numerator_terms_preview": summarize_terms(labels_num, coeffs_num, max_terms=10),
    }
    return H_den_qml, H_num_qml, metadata


class PennyLaneBCOWVQLS:
    """Shot-free VQLS residual estimator for BCOW linear systems."""

    def __init__(
        self,
        n_qubits: int,
        initial_state: np.ndarray,
        ansatz: str,
        layers: int,
        H_denominator,
        H_numerator,
        device_name: str,
        seed: int | None,
        allow_device_fallback: bool = True,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.dim = 2**self.n_qubits
        self.initial_state = normalized_state(initial_state)
        if self.initial_state.shape != (self.dim,):
            raise ValueError("Initial state has wrong dimension.")
        self.ansatz = ansatz
        self.layers = int(layers)
        self.H_denominator = H_denominator
        self.H_numerator = H_numerator
        self.device_name = device_name
        self.seed = seed

        self.dev = make_device(device_name, self.n_qubits, seed, allow_fallback=allow_device_fallback)
        wires = list(range(self.n_qubits))

        def residual_circuit(theta):
            qml.StatePrep(self.initial_state, wires=wires)
            apply_ansatz(theta, self.ansatz, self.layers, self.n_qubits)
            return [qml.expval(self.H_denominator), qml.expval(self.H_numerator)]

        try:
            self.residual_qnode = qml.QNode(residual_circuit, self.dev, diff_method=None)
        except TypeError:
            self.residual_qnode = qml.QNode(residual_circuit, self.dev)

        self.state_dev = make_device(device_name, self.n_qubits, None if seed is None else seed + 7919, allow_fallback=allow_device_fallback)

        def state_circuit(theta):
            qml.StatePrep(self.initial_state, wires=wires)
            apply_ansatz(theta, self.ansatz, self.layers, self.n_qubits)
            return qml.state()

        try:
            self.state_qnode = qml.QNode(state_circuit, self.state_dev, diff_method=None)
        except TypeError:
            self.state_qnode = qml.QNode(state_circuit, self.state_dev)

    def expectations(self, theta: np.ndarray) -> tuple[float, float]:
        vals = self.residual_qnode(np.asarray(theta, dtype=float))
        denominator = float(np.real(np.asarray(vals[0])))
        numerator = float(np.real(np.asarray(vals[1])))
        # Numerical tolerance: both are PSD expectations, so small negatives are clipping artifacts.
        if denominator < 0 and denominator > -1e-10:
            denominator = 0.0
        if numerator < 0 and numerator > -1e-10:
            numerator = 0.0
        return denominator, numerator

    def cost_metrics(self, theta: np.ndarray, min_denominator: float = 1e-14) -> dict[str, float]:
        denominator, numerator = self.expectations(theta)
        if denominator <= min_denominator:
            cost = 1.0 + (min_denominator - denominator) ** 2
            ratio = 0.0
        else:
            ratio = min(1.0, max(0.0, numerator / denominator))
            cost = 1.0 - ratio
        return {
            "cost": float(cost),
            "sqrt_cost": float(math.sqrt(max(cost, 0.0))),
            "denominator": float(denominator),
            "numerator": float(numerator),
            "ratio": float(ratio),
        }

    def state(self, theta: np.ndarray) -> np.ndarray:
        return normalized_state(np.asarray(self.state_qnode(np.asarray(theta, dtype=float)), dtype=complex))


# -----------------------------------------------------------------------------
# Solver orchestration and outputs
# -----------------------------------------------------------------------------


def optimal_scale(A_target: sp.spmatrix, psi: np.ndarray, rhs: np.ndarray) -> complex:
    Apsi = A_target @ psi
    denom = np.vdot(Apsi, Apsi)
    if abs(denom) <= 1e-15:
        return 0.0 + 0.0j
    return complex(np.vdot(Apsi, rhs) / denom)


def normalized_state_metrics(approx: np.ndarray, exact: np.ndarray) -> tuple[float, float]:
    approx = np.asarray(approx, dtype=complex)
    exact = np.asarray(exact, dtype=complex)
    na = np.linalg.norm(approx)
    ne = np.linalg.norm(exact)
    if na <= 1e-15 or ne <= 1e-15:
        return float("nan"), float("nan")
    a = approx / na
    e = exact / ne
    overlap_abs = float(min(1.0, abs(np.vdot(e, a))))
    distance = math.sqrt(max(0.0, 2.0 - 2.0 * overlap_abs))
    return distance, overlap_abs


def extract_history_from_target_solution(solution_target: np.ndarray, target_info: VQLSTarget) -> np.ndarray:
    hist = np.asarray(solution_target[target_info.history_slice], dtype=complex)
    if hist.shape != (target_info.history_dim,):
        raise ValueError("Extracted history vector has wrong shape.")
    return hist


def save_outputs(
    output_prefix: str,
    system,
    layout,
    target_info: VQLSTarget,
    theta0: np.ndarray,
    theta_opt: np.ndarray,
    psi_opt: np.ndarray,
    solution_target: np.ndarray,
    history_vec: np.ndarray,
    final_approx_padded: np.ndarray,
    x_exact_original: np.ndarray | None,
    metadata: dict[str, object],
    optimization_history: list[dict[str, float | int]],
) -> tuple[Path, Path, Path | None]:
    prefix = Path(output_prefix)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")

    history = history_vec.reshape((layout.padded_clock_dim, layout.system_dim))
    final_original = final_approx_padded[: system.state_dim_original]
    n_mech = system.n_mech
    u_approx = np.real_if_close(final_original[:n_mech], tol=1e8)
    v_approx = np.real_if_close(final_original[n_mech:], tol=1e8)

    payload: dict[str, object] = {
        "metadata": json.dumps(metadata, indent=2),
        "layout_json": json.dumps(asdict(layout), indent=2),
        "theta_initial": np.asarray(theta0),
        "theta_opt": np.asarray(theta_opt),
        "psi_opt_normalized": np.asarray(psi_opt),
        "solution_target_scaled": np.asarray(solution_target),
        "bcow_history_vqls": np.asarray(history),
        "x_approx_padded": np.asarray(final_approx_padded),
        "x_approx": np.asarray(final_original),
        "u_approx": np.asarray(u_approx),
        "v_approx": np.asarray(v_approx),
        "A_original": np.asarray(system.A_original),
        "y0_original": np.asarray(system.y0_original),
        "X_all": np.asarray(system.X_all),
    }

    csv_path: Path | None = None
    if x_exact_original is not None:
        payload["x_exact"] = np.asarray(x_exact_original)
        payload["u_exact"] = np.real_if_close(x_exact_original[:n_mech], tol=1e8)
        payload["v_exact"] = np.real_if_close(x_exact_original[n_mech:], tol=1e8)
        table = np.column_stack([system.X_all, payload["u_exact"], u_approx])
        csv_path = prefix.with_name(prefix.name + "_displacement.csv")
        np.savetxt(csv_path, table, delimiter=",", header="X_all,u_exact,u_bcow_vqls", comments="")

    payload["optimization_history"] = json.dumps(optimization_history, indent=2)
    np.savez_compressed(npz_path, **payload)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return npz_path, json_path, csv_path


def solve_bcow_with_pennylane_vqls(args: argparse.Namespace) -> dict[str, object]:
    bcow_module = load_bcow_module(args.bcow_module_path)

    if args.taylor_order < 5:
        warnings.warn(
            "BCOW's condition/error analysis is usually stated for k >= 5. "
            "Small k is useful for VQLS smoke tests but not recommended for final physics runs.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.u_right is not None:
        warnings.warn(
            "--u-right is accepted only for CLI compatibility with the static VQLS script; "
            "the dynamic BCOW ODE builder does not use it.",
            RuntimeWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(args.rng_seed)
    total_start = time.perf_counter()

    system = bcow_module.build_dynamic_md_fe_system(
        n1=args.n1,
        n2=args.n2,
        a=args.a,
        h_factor=args.h_factor,
        k_spring=args.k_spring,
        mass_atom=args.mass_atom,
        pulse_atoms=args.pulse_atoms,
        pulse_amp_factor=args.pulse_amp_factor,
        pulse_wavelength_factor=args.pulse_wavelength_factor,
        pad_to_power2=not args.no_pad,
    )

    if args.segments == 0:
        segments, norm_A_for_segments = bcow_module.choose_segments_from_norm(
            system.A_sparse_padded,
            args.t,
            segment_norm_bound=args.segment_norm_bound,
            exact_norm_dim_limit=args.norm_exact_dim_limit,
        )
    else:
        segments = int(args.segments)
        norm_A_for_segments = bcow_module.estimate_operator_norm(
            system.A_sparse_padded,
            exact_dim_limit=args.norm_exact_dim_limit,
        )
    if segments < 1:
        raise ValueError("segments must be >= 1, or 0 for automatic selection.")

    padding_steps = int(args.padding_steps)
    if padding_steps < 0:
        padding_steps = segments

    layout = bcow_module.build_bcow_layout(
        system_dim=system.state_dim_padded,
        t_final=args.t,
        segments=segments,
        taylor_order=args.taylor_order,
        padding_steps=padding_steps,
        pad_clock_to_power2=not args.no_pad_clock,
    )

    print("PennyLane version:", qml.__version__)
    print("Algorithm: BCOW + PennyLane VQLS")
    print("Mechanics case: unchanged 1D dynamic MD-FE coupling")
    print("Mechanical DOFs:", system.n_mech)
    print("Original state dimension:", system.state_dim_original)
    print("Padded ODE system dimension:", system.state_dim_padded)
    print("System qubits:", layout.system_qubits)
    print("BCOW segments m:", layout.segments)
    print("BCOW Taylor order k:", layout.taylor_order)
    print("BCOW padding steps p:", layout.padding_steps)
    print("Time step h = T/m:", f"{layout.h_step:.8e}")
    print("Estimated ||A||:", f"{norm_A_for_segments:.8e}")
    print("Estimated ||A h||:", f"{norm_A_for_segments * abs(layout.h_step):.8e}")
    print("Padded clock blocks:", layout.padded_clock_dim)
    print("Clock qubits:", layout.clock_qubits)
    print("Direct BCOW VQLS qubits:", layout.total_qubits_without_dilation)
    print("Hermitian-dilation VQLS qubits:", layout.total_qubits_with_hermitian_dilation)

    print("Building sparse BCOW C and rhs ...")
    t0 = time.perf_counter()
    C = bcow_module.build_bcow_sparse_matrix(system.A_sparse_padded, layout)
    rhs = bcow_module.build_bcow_rhs(system.y0_padded, layout)
    print(f"Sparse C: shape={C.shape}, nnz={C.nnz}, build_time={time.perf_counter() - t0:.3f}s")

    target_info = build_vqls_target(C, rhs, args.target, bcow_module)
    print("VQLS target:", target_info.target)
    print("VQLS target dimension:", target_info.dim)
    print("VQLS qubits:", target_info.n_qubits)
    print("PennyLane device requested:", args.device)
    print("Energy estimator:", args.energy_estimator)
    print("Optimizer:", args.optimizer)
    print("Shots: analytic mode; no finite shot count is set")

    if args.energy_estimator != "pauli_hamiltonian":
        raise ValueError("For BCOW+VQLS this script currently supports --energy-estimator pauli_hamiltonian only.")

    recurrence_history: np.ndarray | None = None
    if args.initial_state_guess.lower().replace("-", "_") in {"recurrence", "reference", "reference_solution"} or args.compare_direct:
        print("Computing classical BCOW recurrence for initialization/diagnostics ...")
        recurrence_history = bcow_module.solve_bcow_by_recurrence(system.A_sparse_padded, system.y0_padded, layout)

    print("Building VQLS residual Hamiltonians A†A and A†|b><b|A ...")
    H_den_qml, H_num_qml, hamiltonian_metadata = build_residual_hamiltonians(
        target_info.A_target,
        target_info.rhs_target,
        coefficient_tol=args.coefficient_tol,
        max_qubits=args.max_vqls_qubits,
    )
    print("Denominator Pauli terms:", hamiltonian_metadata["pauli_terms_denominator"])
    print("Numerator Pauli terms:", hamiltonian_metadata["pauli_terms_numerator"])
    print("Hamiltonian build time:", f"{float(hamiltonian_metadata['build_seconds']):.3f}s")
    print("First denominator terms:")
    for label, coeff in hamiltonian_metadata["denominator_terms_preview"]:
        print(f"  {label}: {coeff}")
    print("First numerator terms:")
    for label, coeff in hamiltonian_metadata["numerator_terms_preview"]:
        print(f"  {label}: {coeff}")

    init_state = make_initial_state(
        target_info.dim,
        args.initial_state_guess,
        rng,
        target_info.rhs_target,
        recurrence_history=recurrence_history,
        target=target_info.target,
    )
    n_params = parameter_count(args.ansatz, args.layers, target_info.n_qubits)
    theta0 = args.init_scale * rng.standard_normal(n_params)

    estimator = PennyLaneBCOWVQLS(
        n_qubits=target_info.n_qubits,
        initial_state=init_state,
        ansatz=args.ansatz,
        layers=args.layers,
        H_denominator=H_den_qml,
        H_numerator=H_num_qml,
        device_name=args.device,
        seed=args.rng_seed,
        allow_device_fallback=not args.no_device_fallback,
    )

    def objective(theta: np.ndarray) -> float:
        return float(estimator.cost_metrics(theta)["cost"])

    initial_metrics = estimator.cost_metrics(theta0)
    print("Ansatz:", args.ansatz)
    print("Ansatz layers:", args.layers)
    print("Ansatz parameters:", n_params)
    print("Initial state guess:", args.initial_state_guess)
    print(f"Initial VQLS cost = {initial_metrics['cost']:.8e}, sqrt(cost) = {initial_metrics['sqrt_cost']:.8e}")
    print(f"Initial denominator = {initial_metrics['denominator']:.8e}, numerator = {initial_metrics['numerator']:.8e}")

    progress_records: list[dict[str, float | int]] = []

    def progress_callback(iteration: int, theta: np.ndarray, current_cost: float, best_cost: float) -> None:
        metrics = estimator.cost_metrics(theta)
        record = {
            "iteration": int(iteration),
            "cost": float(metrics["cost"]),
            "sqrt_cost": float(metrics["sqrt_cost"]),
            "denominator": float(metrics["denominator"]),
            "numerator": float(metrics["numerator"]),
            "best_cost": float(best_cost),
        }
        progress_records.append(record)
        if iteration == 0 or iteration == 1 or (args.progress_interval > 0 and iteration % args.progress_interval == 0):
            print(
                f"Iteration {iteration}: cost={record['cost']:.8e}, "
                f"sqrt={record['sqrt_cost']:.8e}, best={record['best_cost']:.8e}, "
                f"den={record['denominator']:.8e}, num={record['numerator']:.8e}"
            )

    opt_start = time.perf_counter()
    opt_key = args.optimizer.lower().replace("_", "-")
    if opt_key == "spsa":
        opt = spsa_minimize(
            objective=objective,
            theta0=theta0,
            rng=rng,
            maxiter=args.maxiter,
            a=args.spsa_a,
            c=args.spsa_c,
            progress_interval=args.progress_interval,
            progress_callback=progress_callback,
        )
    elif opt_key in {"scipy-powell", "powell"}:
        opt = scipy_minimize(
            objective=objective,
            theta0=theta0,
            method="Powell",
            maxiter=args.maxiter,
            progress_interval=args.progress_interval,
            progress_callback=progress_callback,
        )
    elif opt_key in {"scipy-lbfgsb", "lbfgsb", "l-bfgs-b"}:
        opt = scipy_minimize(
            objective=objective,
            theta0=theta0,
            method="L-BFGS-B",
            maxiter=args.maxiter,
            progress_interval=args.progress_interval,
            progress_callback=progress_callback,
        )
    else:
        raise ValueError("optimizer must be 'spsa', 'scipy-powell', or 'scipy-lbfgsb'.")
    opt_seconds = time.perf_counter() - opt_start

    final_metrics = estimator.cost_metrics(opt.x)
    psi_opt = estimator.state(opt.x)
    alpha = optimal_scale(target_info.A_target, psi_opt, target_info.rhs_target)
    solution_target = alpha * psi_opt
    history_vec = extract_history_from_target_solution(solution_target, target_info)
    history = history_vec.reshape((layout.padded_clock_dim, layout.system_dim))
    final_approx_padded, history_norm, final_subspace_probability, final_copy_relative_mismatch = bcow_module.extract_final_vector(
        history,
        layout,
        mode=args.extract_final,
    )

    residual_vec = target_info.A_target @ solution_target - target_info.rhs_target
    relative_residual = float(np.linalg.norm(residual_vec) / max(np.linalg.norm(target_info.rhs_target), 1e-15))

    relative_history_error_vs_direct: float | None = None
    if recurrence_history is not None:
        reference_history_vec = recurrence_history.reshape(-1)
        relative_history_error_vs_direct = float(
            np.linalg.norm(history_vec - reference_history_vec) / max(np.linalg.norm(reference_history_vec), 1e-15)
        )

    x_exact_original: np.ndarray | None = None
    relative_final_error_vs_expm: float | None = None
    final_state_distance_vs_expm: float | None = None
    final_overlap_abs_vs_expm: float | None = None
    if not args.skip_exact:
        print("Computing expm_multiply reference for final block ...")
        x_exact_original = bcow_module.expm_times_vec(system.A_sparse_original, args.t, system.y0_original)
        final_original = final_approx_padded[: system.state_dim_original]
        relative_final_error_vs_expm = float(
            np.linalg.norm(final_original - x_exact_original) / max(np.linalg.norm(x_exact_original), 1e-15)
        )
        final_state_distance_vs_expm, final_overlap_abs_vs_expm = normalized_state_metrics(final_original, x_exact_original)

    diagnostics = VQLSDiagnostics(
        final_cost=float(final_metrics["cost"]),
        sqrt_cost=float(final_metrics["sqrt_cost"]),
        denominator_expectation=float(final_metrics["denominator"]),
        numerator_expectation=float(final_metrics["numerator"]),
        alpha_real=float(np.real(alpha)),
        alpha_imag=float(np.imag(alpha)),
        relative_residual=relative_residual,
        relative_history_error_vs_direct=relative_history_error_vs_direct,
        relative_final_error_vs_expm=relative_final_error_vs_expm,
        final_state_distance_vs_expm=final_state_distance_vs_expm,
        final_overlap_abs_vs_expm=final_overlap_abs_vs_expm,
        pauli_terms_denominator=int(hamiltonian_metadata["pauli_terms_denominator"]),
        pauli_terms_numerator=int(hamiltonian_metadata["pauli_terms_numerator"]),
        optimizer_success=opt.success,
        optimizer_message=opt.message,
        optimizer_iterations=opt.nit,
        wall_time_seconds=time.perf_counter() - total_start,
    )

    metadata: dict[str, object] = {
        "algorithm": "BCOW + shot-free PennyLane VQLS residual simulation",
        "mechanics_case": "1D dynamic MD-FE coupling, y_dot = A y, y=[u;v]",
        "mechanics_builder_changed": False,
        "bcow_module_path": str(Path(args.bcow_module_path).resolve()),
        "pennylane_version": qml.__version__,
        "device_requested": args.device,
        "shot_mode": "analytic_shots_none",
        "vqls_target": target_info.target,
        "energy_estimator": args.energy_estimator,
        "optimizer": args.optimizer,
        "ansatz": args.ansatz,
        "ansatz_layers": args.layers,
        "ansatz_parameter_count": n_params,
        "initial_state_guess": args.initial_state_guess,
        "n1": args.n1,
        "n2": args.n2,
        "mechanical_dofs": system.n_mech,
        "state_dim_original": system.state_dim_original,
        "state_dim_padded": system.state_dim_padded,
        "sparsity_A_original_nnz": int(system.A_sparse_original.nnz),
        "sparsity_A_padded_nnz": int(system.A_sparse_padded.nnz),
        "estimated_A_norm": norm_A_for_segments,
        "estimated_Ah_norm": norm_A_for_segments * abs(layout.h_step),
        "t_final": args.t,
        "layout": asdict(layout),
        "vqls_target_dim": target_info.dim,
        "vqls_qubits": target_info.n_qubits,
        "sparse_C_shape": list(C.shape),
        "sparse_C_nnz": int(C.nnz),
        "hamiltonian_metadata": hamiltonian_metadata,
        "initial_metrics": initial_metrics,
        "diagnostics": asdict(diagnostics),
        "optimizer_wall_time_seconds": opt_seconds,
        "progress_records": progress_records,
    }

    npz_path, json_path, csv_path = save_outputs(
        args.output_prefix,
        system,
        layout,
        target_info,
        theta0,
        opt.x,
        psi_opt,
        solution_target,
        history_vec,
        final_approx_padded,
        x_exact_original,
        metadata,
        opt.history,
    )

    print("\nFinal BCOW+VQLS results")
    print("VQLS target:", target_info.target)
    print("Final VQLS cost:", f"{diagnostics.final_cost:.8e}")
    print("Final sqrt(cost):", f"{diagnostics.sqrt_cost:.8e}")
    print("Relative linear-system residual:", f"{relative_residual:.8e}")
    print("Alpha:", f"{diagnostics.alpha_real:.8e} + {diagnostics.alpha_imag:.8e}j")
    print("History norm:", f"{history_norm:.8e}")
    print("Final clock-subspace probability:", f"{final_subspace_probability:.8e}")
    print("Final copy relative mismatch:", f"{final_copy_relative_mismatch:.8e}")
    if relative_history_error_vs_direct is not None:
        print("Relative history error vs BCOW recurrence:", f"{relative_history_error_vs_direct:.8e}")
    if relative_final_error_vs_expm is not None:
        print("Relative final-block error vs expm_multiply:", f"{relative_final_error_vs_expm:.8e}")
        print("Final normalized state distance vs expm_multiply:", f"{final_state_distance_vs_expm:.8e}")
        print("Final |normalized overlap| vs expm_multiply:", f"{final_overlap_abs_vs_expm:.8e}")
    print("Optimizer success:", opt.success, "|", opt.message)
    print("Saved numerical results to", npz_path)
    print("Saved metadata JSON to", json_path)
    if csv_path is not None:
        print("Saved displacement CSV to", csv_path)

    return {
        "metadata": metadata,
        "npz_path": npz_path,
        "json_path": json_path,
        "csv_path": csv_path,
        "diagnostics": diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BCOW + shot-free PennyLane VQLS for the 1D dynamic MD-FE ODE system.")

    default_bcow_path = Path(__file__).with_name("1Ddynamic_BCOW_ODE_solver.py")
    parser.add_argument("--bcow-module-path", type=str, default=str(default_bcow_path))

    # Dynamic mechanics arguments.  --md-atoms/--fe-elems are aliases for command-line continuity with the static VQLS script.
    parser.add_argument("--n1", "--md-atoms", dest="n1", type=int, default=6, help="Number of MD atoms.")
    parser.add_argument("--n2", "--fe-elems", dest="n2", type=int, default=1, help="Number of FE elements.")
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--h-factor", type=float, default=5.0)
    parser.add_argument("--k-spring", type=float, default=1.0)
    parser.add_argument("--mass-atom", type=float, default=1.0)
    parser.add_argument("--pulse-atoms", type=int, default=50)
    parser.add_argument("--pulse-amp-factor", type=float, default=0.02)
    parser.add_argument("--pulse-wavelength-factor", type=float, default=200.0)
    parser.add_argument("--no-pad", action="store_true")
    parser.add_argument("--u-right", type=float, default=None, help="Accepted for static-VQLS CLI compatibility; ignored here.")

    # BCOW layout arguments.
    parser.add_argument("--t", type=float, default=1.0, help="Final time T.")
    parser.add_argument("--segments", type=int, default=1, help="BCOW segment count m. Use 0 for automatic selection.")
    parser.add_argument("--segment-norm-bound", type=float, default=1.0)
    parser.add_argument("--norm-exact-dim-limit", type=int, default=0)
    parser.add_argument("--taylor-order", type=int, default=3)
    parser.add_argument("--padding-steps", type=int, default=1, help="Final-time copy count p. Use -1 to set p=m.")
    parser.add_argument("--no-pad-clock", action="store_true")
    parser.add_argument("--extract-final", choices=["first", "last", "average"], default="average")

    # VQLS arguments.
    parser.add_argument("--target", choices=["direct-c", "hermitian-dilation"], default="direct-c")
    parser.add_argument("--device", type=str, default="lightning.gpu")
    parser.add_argument("--energy-estimator", choices=["pauli_hamiltonian"], default="pauli_hamiltonian")
    parser.add_argument("--coefficient-tol", type=float, default=1e-10)
    parser.add_argument("--max-vqls-qubits", type=int, default=8)
    parser.add_argument("--no-device-fallback", action="store_true")

    parser.add_argument("--ansatz", choices=["hardware_efficient", "real_ry"], default="hardware_efficient")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--init-scale", type=float, default=0.05)
    parser.add_argument(
        "--initial-state-guess",
        type=str,
        default="index_ramp",
        choices=["zero", "uniform", "index_ramp", "random_complex", "rhs", "recurrence", "reference_solution"],
    )

    parser.add_argument("--optimizer", choices=["spsa", "scipy-powell", "scipy-lbfgsb"], default="spsa")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--spsa-a", type=float, default=0.05)
    parser.add_argument("--spsa-c", type=float, default=0.08)
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--progress-interval", type=int, default=10)

    # Diagnostics/output.
    parser.add_argument("--compare-direct", action="store_true", help="Compute classical recurrence for history-error diagnostics.")
    parser.add_argument("--skip-exact", action="store_true", help="Skip expm_multiply final-block reference.")
    parser.add_argument("--output-prefix", type=str, default="bcow_vqls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solve_bcow_with_pennylane_vqls(args)


if __name__ == "__main__":
    main()
