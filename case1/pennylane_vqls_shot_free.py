#!/usr/bin/env python3
"""
Shot-free PennyLane VQLS simulation for the 1D static MD-FE coupling case.

This is the analytic, no-sampling counterpart of the finite-shot PennyLane
version for the case

    md_atoms = 200
    fe_elems = 59
    reduced free DOFs = 256 = 2**8

The mechanics model is unchanged: the coupled static MD-FE system is assembled as
Kff u_free = ff with a weighted atomistic/continuum handshake at the overlap.

The quantum layer uses PennyLane analytic QNodes:

* PennyLane is used instead of Qiskit.
* Devices are created without a finite shot count, so qml.probs and qml.expval
  return exact simulator probabilities/expectation values.
* No qml.state(), PennyLane statevector return, Qiskit Statevector, or finite
  shot sampling is used.
* The default device is lightning.gpu in analytic mode.

The default ansatz is the same positive controlled-Ry amplitude tree used by the
notebook and the finite-shot script.  For this SPD mechanics problem the
objective is the scaled energy functional

    E(theta) = -0.5 * (ff^T x(theta))**2 / (x(theta)^T Kff x(theta)),

which has the same minimizer as Kff u = ff over the positive normalized ansatz
family.  Two analytic estimators are provided for x^T Kff x:

1. pauli_hamiltonian
   Evaluates the Pauli-encoded Kff as a PennyLane Hamiltonian expectation value.
   This is the closest shot-free analogue of a circuit-level VQLS measurement.

2. probability_laplacian
   Evaluates the equivalent spring energy from exact computational-basis
   probabilities.  This is specialized to the positive-amplitude 1D mechanics
   case and is useful as a low-overhead analytic reference.

Example:
    python pennylane_200atom_59fe_vqls_shot_free.py \
        --device lightning.gpu \
        --energy-estimator pauli_hamiltonian \
        --optimizer spsa \
        --maxiter 100

For a faster analytic mechanics-case run, use:
    python pennylane_200atom_59fe_vqls_shot_free.py \
        --energy-estimator probability_laplacian --maxiter 100
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

import numpy as np

try:
    import pennylane as qml
except ImportError as exc:  # pragma: no cover - helpful runtime message on user machine
    raise SystemExit(
        "This script requires PennyLane. The target environment is "
        "PennyLane 0.45.0, lightning.gpu 0.45.0, Python 3.12.13."
    ) from exc


# -----------------------------------------------------------------------------
# Mechanics assembly: unchanged 1D static MD-FE coupling model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Spring:
    """A quadratic spring contribution c/2 * (u_j - u_i - offset)**2.

    The stiffness field is used only for a matrix-free post-assembly energy
    estimator.  The global K and f are still assembled exactly as in the
    original notebook.
    """

    i: int
    j: int
    stiffness: float
    label: str


@dataclass
class MechanicalSystem:
    md_atoms: int
    fe_elems: int
    node_count: int
    atom_free_count: int
    K: np.ndarray
    f: np.ndarray
    Kff: np.ndarray
    ff: np.ndarray
    free: np.ndarray
    fixed: np.ndarray
    fixed_values: dict[int, float]
    u_reference: np.ndarray
    u_free_reference: np.ndarray
    springs: list[Spring]
    h_a: float
    h0: float
    l0: float
    k: float
    d0: float


def assemble_md_bond(
    K: np.ndarray,
    f: np.ndarray,
    i: int,
    j: int,
    k: float,
    d0: float,
    w: float,
    springs: list[Spring] | None = None,
) -> None:
    """
    MD bond between nodes (i,j), potential 0.5*k*(u_j - u_i - d0)^2.
    Weight w allows handshake averaging.
    """
    Ke = w * k * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    fe = w * np.array([k * d0, -k * d0], dtype=float)
    dofs = [i, j]
    for a in range(2):
        f[dofs[a]] += fe[a]
        for b in range(2):
            K[dofs[a], dofs[b]] += Ke[a, b]
    if springs is not None:
        springs.append(Spring(i=i, j=j, stiffness=float(w * k), label="MD"))


def assemble_fe_element(
    K: np.ndarray,
    f: np.ndarray,
    i: int,
    j: int,
    k: float,
    h_a: float,
    l_ref: float,
    w: float,
    springs: list[Spring] | None = None,
) -> None:
    """
    1D FE Cauchy-Born element between nodes (i,j).

    w_c = 0.5 * k * h_a * (F-1)^2,  F = 1 + (u_j - u_i)/l_ref
    -> linear contribution: c = k*h_a/l_ref.
    """
    c = w * (k * h_a / l_ref)
    Ke = c * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    dofs = [i, j]
    for a in range(2):
        for b in range(2):
            K[dofs[a], dofs[b]] += Ke[a, b]
    if springs is not None:
        springs.append(Spring(i=i, j=j, stiffness=float(c), label="FE"))
    # No constant term for FE in this linearized form, so f is unchanged.


def apply_dirichlet(
    K: np.ndarray,
    f: np.ndarray,
    bc: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = K.shape[0]
    all_idx = np.arange(n)
    fixed = np.array(sorted(bc.keys()), dtype=int)
    free = np.array([i for i in all_idx if i not in fixed], dtype=int)

    u_c = np.zeros(n, dtype=float)
    for i, val in bc.items():
        u_c[i] = val

    Kff = K[np.ix_(free, free)]
    Kfc = K[np.ix_(free, fixed)]
    ff = f[free] - Kfc @ u_c[fixed]
    return Kff, ff, free, fixed, u_c


def build_md_fe_system(
    md_atoms: int = 200,
    fe_elems: int = 59,
    k: float = 10.0,
    h0: float = 1.0,
    l0: float = 5.0,
    h_a: float | None = None,
    d0: float = -0.0,
    u_right: float = 1.5,
) -> MechanicalSystem:
    """Assemble the same static 1D MD-FE coupling system as the notebook."""
    if md_atoms < 2 or fe_elems < 1:
        raise ValueError("md_atoms must be >= 2 and fe_elems must be >= 1.")
    if h_a is None:
        h_a = h0

    # Total nodes: MD gives md_atoms nodes, FE adds fe_elems-1 new nodes after
    # the overlapped first FE element.
    nnode = md_atoms + (fe_elems - 1)
    K = np.zeros((nnode, nnode), dtype=float)
    f = np.zeros(nnode, dtype=float)
    springs: list[Spring] = []

    # MD bonds: (0,1)..(md_atoms-2, md_atoms-1).  Last MD bond is half-weighted.
    for i in range(md_atoms - 1):
        w_md = 0.5 if i == md_atoms - 2 else 1.0
        assemble_md_bond(K, f, i, i + 1, k, d0, w=w_md, springs=springs)

    # First FE element overlaps the last MD bond and is half-weighted.
    i_ovl = md_atoms - 2
    j_ovl = md_atoms - 1
    assemble_fe_element(K, f, i_ovl, j_ovl, k, h_a, l_ref=h0, w=0.5, springs=springs)

    # Remaining FE elements are full-weighted and have reference length l0.
    left = md_atoms - 1
    for _ in range(fe_elems - 1):
        right = left + 1
        assemble_fe_element(K, f, left, right, k, h_a, l_ref=l0, w=1.0, springs=springs)
        left = right

    fixed_values = {0: 0.0, nnode - 1: float(u_right)}
    Kff, ff, free, fixed, _u_c = apply_dirichlet(K, f, fixed_values)

    u_free_reference = np.linalg.solve(Kff, ff)
    u_reference = np.zeros(nnode, dtype=float)
    for idx, val in fixed_values.items():
        u_reference[idx] = val
    u_reference[free] = u_free_reference

    return MechanicalSystem(
        md_atoms=md_atoms,
        fe_elems=fe_elems,
        node_count=nnode,
        atom_free_count=int(np.count_nonzero(free < md_atoms)),
        K=K,
        f=f,
        Kff=Kff,
        ff=ff,
        free=free,
        fixed=fixed,
        fixed_values=fixed_values,
        u_reference=u_reference,
        u_free_reference=u_free_reference,
        springs=springs,
        h_a=float(h_a),
        h0=float(h0),
        l0=float(l0),
        k=float(k),
        d0=float(d0),
    )


# -----------------------------------------------------------------------------
# Pauli encoding Kff = sum c_i P_i
# -----------------------------------------------------------------------------


def pauli_action(label: str, qb: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the basis permutation and phases for a Pauli label.

    The leftmost Pauli character acts on the most significant qubit/wire.  This
    is the ordering used by PennyLane probabilities with wires=[0,1,...,qb-1].
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
        else:
            raise ValueError(f"Unsupported Pauli character {pauli!r} in {label!r}.")
    return targets, phases


def decompose_matrix_to_pauli_terms(
    A: np.ndarray,
    qb: int,
    coefficient_tol: float = 1e-10,
) -> tuple[list[str], np.ndarray]:
    """Encode A as a weighted Pauli sum over qb qubits."""
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
    """Convert a string such as 'IXYZ' into a PennyLane observable."""
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
    summary = []
    for label, coeff in list(zip(labels, coeffs, strict=False))[:max_terms]:
        summary.append((label, f"{float(coeff):.8g}"))
    return summary


# -----------------------------------------------------------------------------
# Positive controlled-Ry amplitude-tree ansatz
# -----------------------------------------------------------------------------


def positive_state_to_tree_angles(vector: np.ndarray, qb: int, eps: float = 1e-15) -> np.ndarray:
    """Convert a nonnegative real initial vector into breadth-first Ry tree angles."""
    vector = np.asarray(vector, dtype=float)
    vector = np.maximum(vector, 0.0)
    norm = np.linalg.norm(vector)
    if norm <= eps:
        raise ValueError("The amplitude-tree initial vector must be nonzero.")
    vector = vector / norm

    angles: list[float] = []
    blocks = [vector]
    for _level in range(qb):
        next_blocks = []
        for block in blocks:
            midpoint = len(block) // 2
            left_norm = np.linalg.norm(block[:midpoint])
            right_norm = np.linalg.norm(block[midpoint:])
            angles.append(2.0 * np.arctan2(right_norm, max(left_norm, eps)))
            next_blocks.extend([block[:midpoint], block[midpoint:]])
        blocks = next_blocks
    return np.asarray(angles, dtype=float)


def make_initial_vector(
    dim: int,
    mode: str,
    rng: np.random.Generator,
    system: MechanicalSystem | None = None,
) -> np.ndarray:
    """Initial nonnegative vector for the amplitude-tree ansatz."""
    mode = mode.lower()
    if mode == "random_positive":
        return np.abs(rng.standard_normal(dim)) + 1e-3
    if mode == "uniform":
        return np.ones(dim, dtype=float)
    if mode == "index_ramp":
        return np.linspace(1.0, float(dim), dim, dtype=float)
    if mode in {"reference", "reference_solution"}:
        if system is None:
            raise ValueError("reference_solution initial guess requires the assembled system.")
        warnings.warn(
            "reference_solution uses the classical solution as the initial ansatz vector. "
            "Use it only as a measurement-pipeline smoke test, not as a VQLS demonstration.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.maximum(system.u_free_reference, 0.0) + 1e-15
    raise ValueError(
        "Unsupported initial_state_guess. Choose one of: "
        "random_positive, uniform, index_ramp, reference_solution."
    )


def project_tree_angles(theta: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Keep controlled-Ry angles in [0, pi] so the tree amplitudes are nonnegative."""
    return np.clip(np.asarray(theta, dtype=float), eps, math.pi - eps)


def apply_controlled_ry_tree(theta: np.ndarray, qb: int) -> None:
    """Apply a positive-amplitude controlled-Ry tree in PennyLane.

    Wire 0 is the most significant bit in the computational-basis index.  The
    gate order matches positive_state_to_tree_angles.
    """
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



# -----------------------------------------------------------------------------
# Shot-free analytic quantum estimators
# -----------------------------------------------------------------------------


def make_device(
    device_name: str,
    wires: int,
    seed: int | None = None,
    allow_fallback: bool = True,
):
    """Create a shot-free PennyLane device, with optional graceful fallback."""

    def _try_create(name: str):
        attempts: list[dict[str, object]] = []
        if seed is not None:
            attempts.append({"wires": wires, "seed": seed})
        attempts.append({"wires": wires})
        # Fallbacks for plugin versions that require an explicit shots keyword.
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


def amplitudes_from_probabilities(probs: np.ndarray) -> np.ndarray:
    """Positive real amplitudes inferred from computational-basis probabilities."""
    probs = normalize_probabilities(probs)
    amps = np.sqrt(probs)
    norm = np.linalg.norm(amps)
    if norm <= 1e-15:
        raise ValueError("Cannot infer amplitudes from a zero probability vector.")
    return amps / norm


def reduced_stiffness_from_springs(
    amplitudes: np.ndarray,
    springs: Sequence[Spring],
    free: np.ndarray,
    node_count: int,
) -> float:
    """Compute x^T Kff x from spring energies using only reduced amplitudes.

    This is matrix-free and equivalent to the assembled reduced stiffness form.
    If a spring touches a fixed DOF, only the free amplitude contributes to Kff.
    Fixed values contribute to the load vector ff, not to the quadratic Kff form.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    free_lookup = -np.ones(node_count, dtype=int)
    free_lookup[free] = np.arange(len(free), dtype=int)

    value = 0.0
    for spring in springs:
        ai = free_lookup[spring.i]
        aj = free_lookup[spring.j]
        c = spring.stiffness
        if ai >= 0 and aj >= 0:
            diff = amplitudes[aj] - amplitudes[ai]
            value += c * diff * diff
        elif ai >= 0:
            value += c * amplitudes[ai] * amplitudes[ai]
        elif aj >= 0:
            value += c * amplitudes[aj] * amplitudes[aj]
    return float(value)


class PennyLaneShotFreeVQLS:
    """Analytic shot-free quantum estimator for the SPD VQLS energy objective."""

    def __init__(
        self,
        qb: int,
        load_vector: np.ndarray,
        labels: Sequence[str],
        coeffs: np.ndarray,
        system: MechanicalSystem,
        device_name: str = "lightning.gpu",
        seed: int | None = 7,
        energy_estimator: str = "pauli_hamiltonian",
        allow_device_fallback: bool = True,
    ) -> None:
        self.qb = int(qb)
        self.dim = 2**self.qb
        self.load_vector = np.asarray(load_vector, dtype=float)
        self.labels = list(labels)
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.system = system
        self.device_name = device_name
        self.seed = seed
        self.energy_estimator = energy_estimator.lower()

        if self.load_vector.shape != (self.dim,):
            raise ValueError("load_vector has the wrong dimension for qb.")
        if self.energy_estimator not in {"pauli_hamiltonian", "probability_laplacian"}:
            raise ValueError("energy_estimator must be 'pauli_hamiltonian' or 'probability_laplacian'.")

        self.dev_probs = make_device(
            device_name, self.qb, seed, allow_fallback=allow_device_fallback
        )

        def probs_circuit(theta):
            apply_controlled_ry_tree(theta, self.qb)
            return qml.probs(wires=list(range(self.qb)))

        try:
            self.probs_qnode = qml.QNode(probs_circuit, self.dev_probs, diff_method=None)
        except TypeError:  # Older or plugin-specific signatures
            self.probs_qnode = qml.QNode(probs_circuit, self.dev_probs)

        self.hamiltonian = None
        self.dev_hamiltonian = None
        self.stiffness_qnode = None
        if self.energy_estimator == "pauli_hamiltonian":
            self.hamiltonian = build_pennylane_hamiltonian(self.labels, self.coeffs)
            self.dev_hamiltonian = make_device(
                device_name,
                self.qb,
                None if seed is None else seed + 1009,
                allow_fallback=allow_device_fallback,
            )

            def stiffness_circuit(theta):
                apply_controlled_ry_tree(theta, self.qb)
                return qml.expval(self.hamiltonian)

            try:
                self.stiffness_qnode = qml.QNode(
                    stiffness_circuit, self.dev_hamiltonian, diff_method=None
                )
            except TypeError:
                self.stiffness_qnode = qml.QNode(stiffness_circuit, self.dev_hamiltonian)

    def probabilities(self, theta: np.ndarray) -> np.ndarray:
        theta = project_tree_angles(theta)
        return normalize_probabilities(np.asarray(self.probs_qnode(theta), dtype=float))

    def amplitudes(self, theta: np.ndarray) -> np.ndarray:
        return amplitudes_from_probabilities(self.probabilities(theta))

    def pauli_stiffness(self, theta: np.ndarray) -> float:
        if self.stiffness_qnode is None:
            raise RuntimeError("Pauli Hamiltonian stiffness QNode was not constructed.")
        theta = project_tree_angles(theta)
        return float(np.asarray(self.stiffness_qnode(theta), dtype=float))

    def energy_metrics(
        self,
        theta: np.ndarray,
        min_stiffness: float = 1e-14,
    ) -> dict[str, float | str | np.ndarray]:
        """Analytic estimate of the scaled SPD energy objective."""
        theta = project_tree_angles(theta)
        probs = self.probabilities(theta)
        amplitudes = amplitudes_from_probabilities(probs)
        load_overlap = float(self.load_vector @ amplitudes)

        if self.energy_estimator == "pauli_hamiltonian":
            stiffness = self.pauli_stiffness(theta)
        else:
            stiffness = reduced_stiffness_from_springs(
                amplitudes,
                self.system.springs,
                self.system.free,
                self.system.node_count,
            )

        if stiffness <= min_stiffness:
            # Positive penalty because the true SPD stiffness expectation should be positive.
            return {
                "energy": 1.0e6 + float((min_stiffness - stiffness) ** 2),
                "alpha": 0.0,
                "load_overlap": load_overlap,
                "stiffness_expectation": float(stiffness),
                "estimator": self.energy_estimator,
                "probabilities": probs,
                "amplitudes": amplitudes,
            }

        alpha = load_overlap / stiffness
        energy = -0.5 * load_overlap * alpha
        return {
            "energy": float(energy),
            "alpha": float(alpha),
            "load_overlap": load_overlap,
            "stiffness_expectation": float(stiffness),
            "estimator": self.energy_estimator,
            "probabilities": probs,
            "amplitudes": amplitudes,
        }

# -----------------------------------------------------------------------------
# Optimizers
# -----------------------------------------------------------------------------


@dataclass
class SPSAResult:
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
) -> SPSAResult:
    """Simple SPSA optimizer with projection to positive tree angles."""
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

    if maxiter <= 0:
        return SPSAResult(
            x=best_theta,
            fun=best_cost,
            nit=0,
            success=True,
            message="SPSA skipped because maxiter <= 0.",
            history=history,
        )

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

        # Opportunistic best tracking; the objective may still be stochastic for SPSA-style runs, so progress
        # points are remeasured below.
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

    # Re-evaluate the best point once at the end.
    best_cost = float(objective(best_theta))
    history.append({"iteration": maxiter, "cost": best_cost, "best_cost": best_cost})
    return SPSAResult(
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
    method: str = "Powell",
    maxiter: int = 100,
    progress_interval: int = 10,
    progress_callback: Callable[[int, np.ndarray, float, float], None] | None = None,
) -> SPSAResult:
    """Optional deterministic SciPy optimizer for shot-free objective checks.

    SPSA remains the dependency-light default.  This helper is useful when SciPy
    is available and you want a deterministic optimizer for the analytic QNode
    objective.  Gradients are not required; L-BFGS-B uses SciPy finite
    differences unless the user later adds an analytic gradient.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise SystemExit(
            "The selected optimizer requires SciPy. Install SciPy or use --optimizer spsa."
        ) from exc

    method = method.upper()
    theta0 = project_tree_angles(theta0)
    best_theta = theta0.copy()
    best_cost = float(objective(theta0))
    history: list[dict[str, float | int]] = [
        {"iteration": 0, "cost": best_cost, "best_cost": best_cost}
    ]
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

    bounds = [(1e-8, math.pi - 1e-8)] * theta0.size if method in {"POWELL", "L-BFGS-B"} else None
    options: dict[str, object] = {"maxiter": int(maxiter), "disp": False}
    if method == "POWELL":
        # Tolerances are intentionally moderate; the objective is ill-conditioned
        # for this mechanics system, and very tight Powell tolerances can spend a
        # long time improving energy without improving displacement visibly.
        options.update({"xtol": 1e-4, "ftol": 1e-8})
    if method == "L-BFGS-B":
        options.update({
            "maxfun": max(150000, 4 * theta0.size * int(maxiter)),
            "maxls": 50,
            "ftol": 1e-12,
            "gtol": 1e-8,
        })
    
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

    return SPSAResult(
        x=best_theta,
        fun=best_cost,
        nit=int(result.nit),
        success=bool(result.success),
        message=f"SciPy {method} completed: {result.message}",
        history=history,
    )


# -----------------------------------------------------------------------------
# Diagnostics and post-processing
# -----------------------------------------------------------------------------


def exact_alignment_cost_from_measured_amplitudes(
    amplitudes: np.ndarray,
    Kff: np.ndarray,
    ff: np.ndarray,
) -> float:
    """Classical diagnostic after analytic circuit readout; not used by the optimizer."""
    b_norm = np.linalg.norm(ff)
    if b_norm <= 1e-15:
        return float("nan")
    b_state = ff / b_norm
    A_state = Kff @ amplitudes
    denominator = float(A_state @ A_state)
    if denominator <= 1e-15:
        return 1.0
    numerator = float(b_state @ A_state)
    return 1.0 - numerator / math.sqrt(denominator)


def make_full_displacement(
    free_values: np.ndarray,
    system: MechanicalSystem,
) -> np.ndarray:
    u = np.zeros(system.node_count, dtype=float)
    for idx, val in system.fixed_values.items():
        u[idx] = val
    u[system.free] = free_values
    return u


def postprocess_forces(system: MechanicalSystem, u_full: np.ndarray) -> dict[str, object]:
    k = system.k
    h_a = system.h_a
    h0 = system.h0
    l0 = system.l0
    d0 = system.d0
    md_atoms = system.md_atoms
    fe_elems = system.fe_elems

    r_full = system.K @ u_full - system.f
    reactions = {int(i): float(r_full[i]) for i in system.fixed}

    def md_force(i: int, j: int) -> float:
        return float(k * ((u_full[j] - u_full[i]) - d0))

    def fe_P(i: int, j: int, l_ref: float) -> float:
        return float(k * h_a * ((1.0 + (u_full[j] - u_full[i]) / l_ref) - 1.0))

    last_md_i, last_md_j = md_atoms - 2, md_atoms - 1
    fe_pairs = [(md_atoms - 2, md_atoms - 1, h0)]
    left = md_atoms - 1
    for _ in range(fe_elems - 1):
        fe_pairs.append((left, left + 1, l0))
        left += 1

    return {
        "reactions": reactions,
        "last_md_bond_force_full": md_force(last_md_i, last_md_j),
        "overlap_fe_P": fe_P(last_md_i, last_md_j, h0),
        "fe_P_list": [fe_P(i, j, L) for (i, j, L) in fe_pairs],
    }


def save_scatter_plot(
    u_free_reference: np.ndarray,
    x_est: np.ndarray,
    atom_free_count: int,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 14,
            "lines.linewidth": 2,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "mathtext.fontset": "cm",
        }
    )

    plot_limit = 1.05 * max(
        float(np.max(np.abs(u_free_reference))),
        float(np.max(np.abs(x_est))),
        1e-12,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(u_free_reference[:atom_free_count], x_est[:atom_free_count], label="atoms")
    ax.scatter(u_free_reference[atom_free_count:], x_est[atom_free_count:], label="nodes")
    ax.plot([0.0, plot_limit], [0.0, plot_limit], "--", label="reference")
    ax.set_xlabel("CC displacement (nm)")
    ax.set_ylabel("QC displacement (nm)")
    ax.set_xlim(0.0, plot_limit)
    ax.set_ylim(0.0, plot_limit)
    ax.legend(frameon=False, ncol=1, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)



# -----------------------------------------------------------------------------
# Main solve routine
# -----------------------------------------------------------------------------


def solve_quasistatic_md_fe_chain_pennylane(
    md_atoms: int = 200,
    fe_elems: int = 59,
    k: float = 10.0,
    h0: float = 1.0,
    l0: float = 5.0,
    h_a: float | None = None,
    d0: float = -0.0,
    u_right: float = 1.5,
    device: str = "lightning.gpu",
    energy_estimator: str = "pauli_hamiltonian",
    optimizer: str = "spsa",
    optimizer_maxiter: int = 100,
    spsa_a: float = 0.03,
    spsa_c: float = 0.08,
    rng_seed: int = 7,
    initial_state_guess: str = "index_ramp",
    coefficient_tol: float = 1e-10,
    progress_interval: int = 10,
    print_vectors: bool = False,
    allow_device_fallback: bool = True,
) -> dict[str, object]:
    system = build_md_fe_system(
        md_atoms=md_atoms,
        fe_elems=fe_elems,
        k=k,
        h0=h0,
        l0=l0,
        h_a=h_a,
        d0=d0,
        u_right=u_right,
    )

    dim = len(system.ff)
    qb = int(round(math.log2(dim)))
    if 2**qb != dim:
        raise ValueError(f"The reduced system has {dim} free DOFs, which is not a power of two.")

    print("PennyLane version:", qml.__version__)
    print("VQLS level: shot-free analytic PennyLane quantum-circuit simulation")
    print("Mechanics case: 1D static MD-FE coupling")
    print("Nodes:", system.node_count)
    print("Free DOFs:", dim)
    print("Qubits:", qb)
    print("Device requested:", device)
    print("Energy estimator:", energy_estimator)
    print("Optimizer:", optimizer)
    print("Shots: analytic mode; no finite shot count is set")
    print("Nonzero load entries:", int(np.count_nonzero(np.abs(system.ff) > 1e-14)))

    labels, coeffs = decompose_matrix_to_pauli_terms(system.Kff, qb, coefficient_tol=coefficient_tol)
    print(f"Pauli encoding uses {len(labels)} nonzero Pauli-string terms.")
    print("First encoded terms:")
    for label, coeff_text in summarize_pauli_terms(labels, coeffs, max_terms=8):
        print(f"  {label}: {coeff_text}")

    rng = np.random.default_rng(rng_seed)
    initial_vector = make_initial_vector(dim, initial_state_guess, rng, system=system)
    theta0 = project_tree_angles(positive_state_to_tree_angles(initial_vector, qb))

    estimator = PennyLaneShotFreeVQLS(
        qb=qb,
        load_vector=system.ff,
        labels=labels,
        coeffs=coeffs,
        system=system,
        device_name=device,
        seed=rng_seed,
        energy_estimator=energy_estimator,
        allow_device_fallback=allow_device_fallback,
    )

    def objective(theta: np.ndarray) -> float:
        return float(estimator.energy_metrics(theta)["energy"])

    print("Ansatz type: controlled-Ry positive amplitude tree")
    print("Number of ansatz parameters:", len(theta0))
    print("Initial state guess:", initial_state_guess)

    initial_metrics = estimator.energy_metrics(theta0)
    print(f"Initial shot-free energy = {float(initial_metrics['energy']):.8e}")
    print(f"Initial shot-free alpha = {float(initial_metrics['alpha']):.8e}")
    print(f"Initial shot-free stiffness expectation = {float(initial_metrics['stiffness_expectation']):.8e}")
    print(f"Initial shot-free load overlap = {float(initial_metrics['load_overlap']):.8e}")

    exact_optimal_energy = float(-0.5 * system.ff @ system.u_free_reference)
    print(f"Classical reference minimum energy = {exact_optimal_energy:.8e}")

    progress_metrics: list[dict[str, float | int]] = []

    def progress_callback(iteration: int, theta: np.ndarray, current_cost: float, best_cost: float) -> None:
        metrics = estimator.energy_metrics(theta)
        record = {
            "iteration": int(iteration),
            "energy": float(metrics["energy"]),
            "alpha": float(metrics["alpha"]),
            "stiffness_expectation": float(metrics["stiffness_expectation"]),
            "load_overlap": float(metrics["load_overlap"]),
            "best_energy": float(best_cost),
        }
        progress_metrics.append(record)
        if iteration == 0 or iteration == 1 or (progress_interval > 0 and iteration % progress_interval == 0):
            print(
                f"Iteration {iteration}: "
                f"energy = {record['energy']:.8e}, "
                f"best = {record['best_energy']:.8e}, "
                f"alpha = {record['alpha']:.8e}, "
                f"stiffness = {record['stiffness_expectation']:.8e}, "
                f"load = {record['load_overlap']:.8e}"
            )

    optimizer_key = optimizer.lower().replace("_", "-")
    if optimizer_key == "spsa":
        opt = spsa_minimize(
            objective=objective,
            theta0=theta0,
            rng=rng,
            maxiter=optimizer_maxiter,
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
            maxiter=optimizer_maxiter,
            progress_interval=progress_interval,
            progress_callback=progress_callback,
        )
    elif optimizer_key == "scipy-lbfgsb":
        opt = scipy_minimize(
            objective=objective,
            theta0=theta0,
            method="L-BFGS-B",
            maxiter=optimizer_maxiter,
            progress_interval=progress_interval,
            progress_callback=progress_callback,
        )
    else:
        raise ValueError("optimizer must be 'spsa', 'scipy-powell', or 'scipy-lbfgsb'.")

    # Final analytic circuit evaluation for the solution vector.  Unlike the
    # finite-shot version, this is not a tomography-like sampled readout: qml.probs
    # returns exact simulator probabilities because the device is analytic.
    final_metrics = estimator.energy_metrics(opt.x)
    final_probs = np.asarray(final_metrics["probabilities"], dtype=float)
    final_amplitudes = np.asarray(final_metrics["amplitudes"], dtype=float)
    alpha_quantum = float(final_metrics["alpha"])
    x_est = alpha_quantum * final_amplitudes
    u_est = make_full_displacement(x_est, system)

    # Classical dense operations below are diagnostics only.  They are not used
    # in the optimizer or analytic quantum objective.
    relative_solution_error = float(
        np.linalg.norm(x_est - system.u_free_reference) / np.linalg.norm(system.u_free_reference)
    )
    relative_residual = float(np.linalg.norm(system.Kff @ x_est - system.ff) / np.linalg.norm(system.ff))
    final_alignment_cost = exact_alignment_cost_from_measured_amplitudes(
        final_amplitudes, system.Kff, system.ff
    )

    # A diagnostic scale using exact Kff with the circuit amplitudes separates
    # ansatz shape error from any scalar-estimator mismatch.  For the shot-free
    # Pauli Hamiltonian estimator it should agree with alpha_quantum up to small
    # numerical/Pauli-decomposition tolerance.
    exact_stiffness_measured_shape = float(final_amplitudes @ (system.Kff @ final_amplitudes))
    exact_load_measured_shape = float(system.ff @ final_amplitudes)
    alpha_postprocessed = exact_load_measured_shape / exact_stiffness_measured_shape
    x_est_postprocessed_scale = alpha_postprocessed * final_amplitudes
    relative_solution_error_postprocessed_scale = float(
        np.linalg.norm(x_est_postprocessed_scale - system.u_free_reference)
        / np.linalg.norm(system.u_free_reference)
    )
    relative_residual_postprocessed_scale = float(
        np.linalg.norm(system.Kff @ x_est_postprocessed_scale - system.ff) / np.linalg.norm(system.ff)
    )
    theta_change_norm = float(np.linalg.norm(np.asarray(opt.x) - theta0))

    forces_quantum = postprocess_forces(system, u_est)

    print("\nFinal shot-free VQLS results")
    print(f"Final shot-free training energy: {float(final_metrics['energy']):.8e}")
    print(f"Classical reference minimum energy: {exact_optimal_energy:.8e}")
    print(f"Final shot-free alpha: {alpha_quantum:.8e}")
    print(f"Final shot-free stiffness expectation: {float(final_metrics['stiffness_expectation']):.8e}")
    print(f"Final shot-free load overlap: {float(final_metrics['load_overlap']):.8e}")
    print(f"Theta movement norm ||theta_opt - theta_initial||: {theta_change_norm:.8e}")
    print(f"Posthoc Level-2 alignment cost from analytic amplitudes: {final_alignment_cost:.8e}")
    print(f"Relative solution error, analytic quantum scale: {relative_solution_error:.8e}")
    print(f"Relative residual, analytic quantum scale: {relative_residual:.8e}")
    print(
        "Relative solution error, exact postprocessed scale: "
        f"{relative_solution_error_postprocessed_scale:.8e}"
    )
    print(
        "Relative residual, exact postprocessed scale: "
        f"{relative_residual_postprocessed_scale:.8e}"
    )
    print("Optimizer success:", opt.success, "|", opt.message)

    if print_vectors:
        print("VQLS analytic free displacement:", x_est)
        print("Reference free displacement:", system.u_free_reference)
    else:
        print("VQLS analytic displacement preview:", np.r_[x_est[:5], x_est[-5:]])
        print("Reference displacement preview:", np.r_[system.u_free_reference[:5], system.u_free_reference[-5:]])

    return {
        "vqls_level": "shot-free analytic PennyLane VQLS simulation",
        "pennylane_version": qml.__version__,
        "device_requested": device,
        "shot_mode": "analytic_shots_none",
        "energy_estimator": energy_estimator,
        "optimizer": optimizer,
        "md_atoms": md_atoms,
        "fe_elems": fe_elems,
        "node_count": system.node_count,
        "free_dofs": system.free,
        "fixed_dofs": system.fixed,
        "atom_free_count": system.atom_free_count,
        "K": system.K,
        "f": system.f,
        "AA": system.Kff,
        "bb": system.ff,
        "u": system.u_reference,
        "u_free": system.u_free_reference,
        "x_est": x_est,
        "u_est": u_est,
        "x_est_postprocessed_scale": x_est_postprocessed_scale,
        "pauli_term_count": len(labels),
        "pauli_terms": list(zip(labels, coeffs.tolist(), strict=False)),
        "pauli_summary": summarize_pauli_terms(labels, coeffs),
        "theta_initial": theta0,
        "theta_opt": opt.x,
        "theta_change_norm": theta_change_norm,
        "final_probabilities": final_probs,
        "final_amplitudes": final_amplitudes,
        "optimization_result": opt,
        "optimization_history": opt.history,
        "progress_metrics": progress_metrics,
        "initial_training_cost": float(initial_metrics["energy"]),
        "initial_alpha": float(initial_metrics["alpha"]),
        "initial_stiffness_expectation": float(initial_metrics["stiffness_expectation"]),
        "initial_load_overlap": float(initial_metrics["load_overlap"]),
        "final_training_cost": float(final_metrics["energy"]),
        "reference_minimum_energy": exact_optimal_energy,
        "final_alpha": alpha_quantum,
        "final_stiffness_expectation": float(final_metrics["stiffness_expectation"]),
        "final_load_overlap": float(final_metrics["load_overlap"]),
        "final_level2_alignment_cost_posthoc": final_alignment_cost,
        "relative_solution_error": relative_solution_error,
        "relative_residual": relative_residual,
        "relative_solution_error_postprocessed_scale": relative_solution_error_postprocessed_scale,
        "relative_residual_postprocessed_scale": relative_residual_postprocessed_scale,
        "forces_quantum": forces_quantum,
        "initial_state_guess": initial_state_guess,
        "ansatz_type": "controlled_ry_amplitude_tree",
    }


def save_results_npz(out: dict[str, object], output_path: Path) -> None:
    """Save numerical arrays and compact metadata."""
    opt: SPSAResult = out["optimization_result"]  # type: ignore[assignment]
    metadata = {
        "vqls_level": out["vqls_level"],
        "pennylane_version": out["pennylane_version"],
        "device_requested": out["device_requested"],
        "shot_mode": out["shot_mode"],
        "energy_estimator": out["energy_estimator"],
        "optimizer": out["optimizer"],
        "md_atoms": out["md_atoms"],
        "fe_elems": out["fe_elems"],
        "node_count": out["node_count"],
        "pauli_term_count": out["pauli_term_count"],
        "initial_training_cost": out["initial_training_cost"],
        "initial_alpha": out["initial_alpha"],
        "initial_stiffness_expectation": out["initial_stiffness_expectation"],
        "initial_load_overlap": out["initial_load_overlap"],
        "final_training_cost": out["final_training_cost"],
        "reference_minimum_energy": out["reference_minimum_energy"],
        "final_alpha": out["final_alpha"],
        "final_stiffness_expectation": out["final_stiffness_expectation"],
        "final_load_overlap": out["final_load_overlap"],
        "theta_change_norm": out["theta_change_norm"],
        "final_level2_alignment_cost_posthoc": out["final_level2_alignment_cost_posthoc"],
        "relative_solution_error": out["relative_solution_error"],
        "relative_residual": out["relative_residual"],
        "relative_solution_error_postprocessed_scale": out[
            "relative_solution_error_postprocessed_scale"
        ],
        "relative_residual_postprocessed_scale": out["relative_residual_postprocessed_scale"],
        "optimizer_success": opt.success,
        "optimizer_message": opt.message,
        "optimizer_iterations": opt.nit,
        "initial_state_guess": out["initial_state_guess"],
        "ansatz_type": out["ansatz_type"],
    }

    np.savez_compressed(
        output_path,
        K=np.asarray(out["K"]),
        f=np.asarray(out["f"]),
        AA=np.asarray(out["AA"]),
        bb=np.asarray(out["bb"]),
        u=np.asarray(out["u"]),
        u_free=np.asarray(out["u_free"]),
        x_est=np.asarray(out["x_est"]),
        u_est=np.asarray(out["u_est"]),
        x_est_postprocessed_scale=np.asarray(out["x_est_postprocessed_scale"]),
        final_probabilities=np.asarray(out["final_probabilities"]),
        final_amplitudes=np.asarray(out["final_amplitudes"]),
        theta_initial=np.asarray(out["theta_initial"]),
        theta_opt=np.asarray(out["theta_opt"]),
        free_dofs=np.asarray(out["free_dofs"]),
        fixed_dofs=np.asarray(out["fixed_dofs"]),
        metadata=json.dumps(metadata, indent=2),
        optimization_history=json.dumps(opt.history, indent=2),
        progress_metrics=json.dumps(out["progress_metrics"], indent=2),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shot-free analytic PennyLane VQLS for the 200-atom/59-FE 1D MD-FE mechanics case."
    )
    parser.add_argument("--md-atoms", type=int, default=200)
    parser.add_argument("--fe-elems", type=int, default=59)
    parser.add_argument("--k", type=float, default=10.0)
    parser.add_argument("--h0", type=float, default=1.0)
    parser.add_argument("--l0", type=float, default=5.0)
    parser.add_argument("--h-a", type=float, default=None)
    parser.add_argument("--d0", type=float, default=-0.0)
    parser.add_argument("--u-right", type=float, default=1.5)

    parser.add_argument("--device", type=str, default="lightning.gpu")
    parser.add_argument(
        "--energy-estimator",
        type=str,
        default="pauli_hamiltonian",
        choices=["pauli_hamiltonian", "probability_laplacian"],
    )
    parser.add_argument("--coefficient-tol", type=float, default=1e-10)
    parser.add_argument("--no-device-fallback", action="store_true")

    # Backward-compatible no-op arguments from the finite-shot script.  They are
    # accepted so an old command line does not fail, but they are ignored.
    parser.add_argument("--shots-hamiltonian", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shots-probs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--solution-shots", type=int, default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "--optimizer",
        type=str,
        default="spsa",
        choices=["spsa", "scipy-powell", "scipy-lbfgsb"],
    )
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--spsa-a", type=float, default=0.03)
    parser.add_argument("--spsa-c", type=float, default=0.08)
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument(
        "--initial-state-guess",
        type=str,
        default="index_ramp",
        choices=["random_positive", "uniform", "index_ramp", "reference_solution"],
    )
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--print-vectors", action="store_true")

    parser.add_argument("--output-prefix", type=str, default="shot_free")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(
        value is not None
        for value in (args.shots_hamiltonian, args.shots_probs, args.solution_shots)
    ):
        warnings.warn(
            "Shot arguments were supplied but are ignored by this shot-free script. "
            "The PennyLane devices are created without a finite shot count.",
            RuntimeWarning,
            stacklevel=2,
        )

    out = solve_quasistatic_md_fe_chain_pennylane(
        md_atoms=args.md_atoms,
        fe_elems=args.fe_elems,
        k=args.k,
        h0=args.h0,
        l0=args.l0,
        h_a=args.h_a,
        d0=args.d0,
        u_right=args.u_right,
        device=args.device,
        energy_estimator=args.energy_estimator,
        optimizer=args.optimizer,
        optimizer_maxiter=args.maxiter,
        spsa_a=args.spsa_a,
        spsa_c=args.spsa_c,
        rng_seed=args.rng_seed,
        initial_state_guess=args.initial_state_guess,
        coefficient_tol=args.coefficient_tol,
        progress_interval=args.progress_interval,
        print_vectors=args.print_vectors,
        allow_device_fallback=not args.no_device_fallback,
    )

    prefix = Path(args.output_prefix)
    results_path = prefix.with_suffix(".npz")
    save_results_npz(out, results_path)
    print(f"Saved numerical results to {results_path}")

    if not args.no_plot:
        plot_path = prefix.with_name(prefix.name + "_scatter.png")
        save_scatter_plot(
            np.asarray(out["u_free"]),
            np.asarray(out["x_est"]),
            int(out["atom_free_count"]),
            plot_path,
        )
        print(f"Saved scatter plot to {plot_path}")


if __name__ == "__main__":
    main()
