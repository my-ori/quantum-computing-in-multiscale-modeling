#!/usr/bin/env python3
"""
Restarted BCOW + VQLS PennyLane simulation for the 1D dynamic MD-FE ODE problem.

This driver keeps the mechanics construction and the single-window BCOW+VQLS
machinery from 1Ddynamic_BCOW_VQLS_pennylane.py, but changes the time strategy:

    y(0) -- small BCOW+VQLS window --> y(dt)
         -- small BCOW+VQLS window --> y(2dt)
         ...
         -- small BCOW+VQLS window --> y(T)

The purpose is to avoid one giant BCOW history system for long final times such
as T=10.  Each window solves a small linear system

    C_{m,k,p}(A h) |history_s> = |rhs_s>,       rhs_s = y_s in the first block,

extracts the final BCOW block, and feeds that vector into the next window.

Important interpretation:
    This is still a circuit-level, shot-free VQLS simulation using dense Pauli
    Hamiltonian residuals.  Restarting keeps each VQLS instance small enough for
    near-term validation, but errors accumulate across windows.  Use
    --compare-direct and the per-window/global diagnostics to decide whether the
    chosen window size and VQLS tolerance are acceptable.

Example for the small case that previously worked at t=0.1, rolled to t=10:

    python 1Ddynamic_BCOW_VQLS_restarted_pennylane.py \
        --n1 5 --n2 2 --pulse-atoms 3 \
        --t 10 --restart-window 0.1 \
        --segments 1 --taylor-order 4 --padding-steps 1 \
        --target direct-c --device lightning.gpu \
        --energy-estimator pauli_hamiltonian \
        --optimizer scipy-lbfgsb --maxiter 300 \
        --ansatz real_ry_identity --layers 12 \
        --initial-state-guess rhs --init-scale 0.0 \
        --theta-warm-start previous --theta-jitter 0.01 \
        --compare-direct --exact-every 10 \
        --output-prefix bcow_vqls_restarted_t10
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    import scipy.sparse as sp
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires SciPy. Install with: pip install scipy") from exc


# -----------------------------------------------------------------------------
# Dynamic imports of the existing project files
# -----------------------------------------------------------------------------


def load_module_from_path(path: str | Path, module_name: str):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find module at {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# -----------------------------------------------------------------------------
# Restart-friendly ansatz layer
# -----------------------------------------------------------------------------


def ansatz_key(ansatz: str) -> str:
    return ansatz.lower().replace("-", "_")


def parameter_count(ansatz: str, layers: int, n_qubits: int, vqls_module) -> int:
    """Parameter count for the original and identity-preserving ansaetze."""
    key = ansatz_key(ansatz)
    if key in {"real_ry", "ry", "hardware_efficient", "he", "rot"}:
        return int(vqls_module.parameter_count(ansatz, layers, n_qubits))
    if key in {"real_ry_identity", "ry_identity", "identity_ry"}:
        # Per layer: n single-qubit RY rotations + n ring CRY entanglers.
        return int(layers) * int(n_qubits) * 2
    if key in {"hardware_efficient_identity", "he_identity", "rot_identity"}:
        # Per layer: Rot(phi,theta,omega) on every qubit + n ring CRY entanglers.
        return int(layers) * int(n_qubits) * 4
    raise ValueError(
        "ansatz must be one of real_ry, hardware_efficient, real_ry_identity, "
        "or hardware_efficient_identity."
    )


def apply_restart_ansatz(theta: np.ndarray, ansatz: str, layers: int, n_qubits: int, qml, vqls_module) -> None:
    """Apply a trainable ansatz.

    The two *_identity variants are useful for restarted VQLS because zero
    parameters leave the prepared RHS state unchanged.  This makes
    --initial-state-guess rhs --init-scale 0.0 a true warm start.
    """
    key = ansatz_key(ansatz)
    if key in {"real_ry", "ry", "hardware_efficient", "he", "rot"}:
        vqls_module.apply_ansatz(theta, ansatz, layers, n_qubits)
        return

    theta = np.asarray(theta, dtype=float)
    wires = list(range(n_qubits))

    if key in {"real_ry_identity", "ry_identity", "identity_ry"}:
        params = theta.reshape((layers, 2, n_qubits))
        for layer in range(layers):
            for wire in wires:
                qml.RY(params[layer, 0, wire], wires=wire)
            if n_qubits >= 2:
                for wire in range(n_qubits - 1):
                    qml.CRY(params[layer, 1, wire], wires=[wire, wire + 1])
                if n_qubits > 2:
                    qml.CRY(params[layer, 1, n_qubits - 1], wires=[n_qubits - 1, 0])
        return

    if key in {"hardware_efficient_identity", "he_identity", "rot_identity"}:
        params = theta.reshape((layers, n_qubits, 4))
        for layer in range(layers):
            for wire in wires:
                qml.Rot(params[layer, wire, 0], params[layer, wire, 1], params[layer, wire, 2], wires=wire)
            if n_qubits >= 2:
                for wire in range(n_qubits - 1):
                    qml.CRY(params[layer, wire, 3], wires=[wire, wire + 1])
                if n_qubits > 2:
                    qml.CRY(params[layer, n_qubits - 1, 3], wires=[n_qubits - 1, 0])
        return

    raise ValueError(f"Unsupported ansatz {ansatz!r}.")


class RestartedPennyLaneVQLS:
    """Shot-free VQLS estimator using the restart-friendly ansatz layer."""

    def __init__(
        self,
        *,
        n_qubits: int,
        initial_state: np.ndarray,
        ansatz: str,
        layers: int,
        H_denominator,
        H_numerator,
        device_name: str,
        seed: int | None,
        allow_device_fallback: bool,
        qml,
        vqls_module,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.dim = 2**self.n_qubits
        self.initial_state = vqls_module.normalized_state(initial_state)
        if self.initial_state.shape != (self.dim,):
            raise ValueError("Initial state has wrong dimension.")
        self.ansatz = ansatz
        self.layers = int(layers)
        self.H_denominator = H_denominator
        self.H_numerator = H_numerator
        self.device_name = device_name
        self.seed = seed
        self.qml = qml
        self.vqls_module = vqls_module

        self.dev = vqls_module.make_device(device_name, self.n_qubits, seed, allow_fallback=allow_device_fallback)
        wires = list(range(self.n_qubits))

        def residual_circuit(theta):
            qml.StatePrep(self.initial_state, wires=wires)
            apply_restart_ansatz(theta, self.ansatz, self.layers, self.n_qubits, qml, vqls_module)
            return [qml.expval(self.H_denominator), qml.expval(self.H_numerator)]

        try:
            self.residual_qnode = qml.QNode(residual_circuit, self.dev, diff_method=None)
        except TypeError:
            self.residual_qnode = qml.QNode(residual_circuit, self.dev)

        self.state_dev = vqls_module.make_device(
            device_name,
            self.n_qubits,
            None if seed is None else seed + 7919,
            allow_fallback=allow_device_fallback,
        )

        def state_circuit(theta):
            qml.StatePrep(self.initial_state, wires=wires)
            apply_restart_ansatz(theta, self.ansatz, self.layers, self.n_qubits, qml, vqls_module)
            return qml.state()

        try:
            self.state_qnode = qml.QNode(state_circuit, self.state_dev, diff_method=None)
        except TypeError:
            self.state_qnode = qml.QNode(state_circuit, self.state_dev)

    def expectations(self, theta: np.ndarray) -> tuple[float, float]:
        vals = self.residual_qnode(np.asarray(theta, dtype=float))
        denominator = float(np.real(np.asarray(vals[0])))
        numerator = float(np.real(np.asarray(vals[1])))
        if denominator < 0 and denominator > -1e-10:
            denominator = 0.0
        if numerator < 0 and numerator > -1e-10:
            numerator = 0.0
        return denominator, numerator

    def cost_metrics(self, theta: np.ndarray, min_denominator: float = 1e-14) -> dict[str, float]:
        denominator, numerator = self.expectations(theta)
        if denominator <= min_denominator:
            ratio = 0.0
            cost = 1.0 + (min_denominator - denominator) ** 2
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
        return self.vqls_module.normalized_state(np.asarray(self.state_qnode(np.asarray(theta, dtype=float)), dtype=complex))


# -----------------------------------------------------------------------------
# Pauli Hamiltonian cache helpers
# -----------------------------------------------------------------------------


@dataclass
class CachedOperators:
    layout: object
    C: sp.csr_matrix
    H_den_qml: object
    denominator_metadata: dict[str, object]
    target_dim: int
    n_qubits: int


@dataclass
class WindowResult:
    window_index: int
    attempt_index: int
    t_start: float
    t_end: float
    dt: float
    segments: int
    taylor_order: int
    padding_steps: int
    vqls_qubits: int
    ansatz_parameter_count: int
    initial_cost: float
    initial_sqrt_cost: float
    final_cost: float
    final_sqrt_cost: float
    relative_residual: float
    alpha_real: float
    alpha_imag: float
    history_norm: float
    final_subspace_probability: float
    final_copy_relative_mismatch: float
    relative_history_error_vs_recurrence: float | None
    relative_final_error_vs_recurrence: float | None
    relative_local_error_vs_expm: float | None
    optimizer_success: bool
    optimizer_message: str
    optimizer_iterations: int
    optimizer_wall_time_seconds: float
    hamiltonian_build_seconds: float
    pauli_terms_denominator: int
    pauli_terms_numerator: int
    theta_opt: np.ndarray
    y_end_padded: np.ndarray
    history_vec: np.ndarray
    progress_records: list[dict[str, float | int]]
    optimizer_history: list[dict[str, float | int]]

    def summary_dict(self) -> dict[str, object]:
        d = asdict(self)
        # Large arrays are stored in NPZ, not in JSON/CSV summaries.
        d.pop("theta_opt", None)
        d.pop("y_end_padded", None)
        d.pop("history_vec", None)
        d.pop("progress_records", None)
        d.pop("optimizer_history", None)
        return d


def check_power_of_two_dim(dim: int, max_qubits: int) -> int:
    n_qubits = int(math.log2(dim)) if dim > 0 else -1
    if dim <= 0 or 2**n_qubits != dim:
        raise ValueError(f"Target dimension {dim} is not a power of two.")
    if n_qubits > max_qubits:
        raise ValueError(
            f"This restarted Pauli-expanded VQLS window uses {n_qubits} qubits, "
            f"but --max-vqls-qubits is {max_qubits}.  Decrease --restart-window, "
            "decrease per-window --segments/--padding-steps, or increase the guardrail for small tests."
        )
    return n_qubits


def build_denominator_hamiltonian(A_target: sp.spmatrix, coefficient_tol: float, max_qubits: int, vqls_module):
    A = sp.csr_matrix(A_target, dtype=complex)
    dim = A.shape[0]
    n_qubits = check_power_of_two_dim(dim, max_qubits)
    t0 = time.perf_counter()
    H_den_sparse = (A.conjugate().transpose() @ A).tocsr()
    H_den_dense = H_den_sparse.toarray()
    labels_den, coeffs_den = vqls_module.decompose_hermitian_to_pauli_terms(H_den_dense, n_qubits, coefficient_tol)
    H_den_qml = vqls_module.build_pennylane_hamiltonian(labels_den, coeffs_den)
    metadata = {
        "n_qubits": n_qubits,
        "dim": dim,
        "pauli_terms_denominator": len(labels_den),
        "coefficient_tol": coefficient_tol,
        "denominator_build_seconds": time.perf_counter() - t0,
        "denominator_terms_preview": vqls_module.summarize_terms(labels_den, coeffs_den, max_terms=10),
    }
    return H_den_qml, metadata


def build_numerator_hamiltonian(A_target: sp.spmatrix, rhs_target: np.ndarray, coefficient_tol: float, max_qubits: int, vqls_module):
    A = sp.csr_matrix(A_target, dtype=complex)
    dim = A.shape[0]
    n_qubits = check_power_of_two_dim(dim, max_qubits)
    t0 = time.perf_counter()
    b = vqls_module.normalized_state(rhs_target)
    v = A.conjugate().transpose() @ b
    H_num_dense = np.outer(v, v.conjugate())
    labels_num, coeffs_num = vqls_module.decompose_hermitian_to_pauli_terms(H_num_dense, n_qubits, coefficient_tol)
    H_num_qml = vqls_module.build_pennylane_hamiltonian(labels_num, coeffs_num)
    metadata = {
        "pauli_terms_numerator": len(labels_num),
        "numerator_build_seconds": time.perf_counter() - t0,
        "numerator_terms_preview": vqls_module.summarize_terms(labels_num, coeffs_num, max_terms=10),
    }
    return H_num_qml, metadata


def operator_cache_key(args: argparse.Namespace, dt: float, segments: int, padding_steps: int) -> tuple[object, ...]:
    return (
        args.target,
        round(float(dt), 15),
        int(segments),
        int(args.taylor_order),
        int(padding_steps),
        bool(args.no_pad_clock),
    )


def get_cached_operators(
    *,
    cache: dict[tuple[object, ...], CachedOperators],
    args: argparse.Namespace,
    bcow_module,
    vqls_module,
    system,
    dt: float,
    segments: int,
    padding_steps: int,
) -> CachedOperators:
    key = operator_cache_key(args, dt, segments, padding_steps)
    if key in cache:
        return cache[key]

    layout = bcow_module.build_bcow_layout(
        system_dim=system.state_dim_padded,
        t_final=dt,
        segments=segments,
        taylor_order=args.taylor_order,
        padding_steps=padding_steps,
        pad_clock_to_power2=not args.no_pad_clock,
    )
    C = bcow_module.build_bcow_sparse_matrix(system.A_sparse_padded, layout)
    dummy_rhs = bcow_module.build_bcow_rhs(np.ones(system.state_dim_padded, dtype=complex), layout)
    target_info = vqls_module.build_vqls_target(C, dummy_rhs, args.target, bcow_module)
    H_den_qml, denom_meta = build_denominator_hamiltonian(
        target_info.A_target,
        coefficient_tol=args.coefficient_tol,
        max_qubits=args.max_vqls_qubits,
        vqls_module=vqls_module,
    )
    cached = CachedOperators(
        layout=layout,
        C=C,
        H_den_qml=H_den_qml,
        denominator_metadata=denom_meta,
        target_dim=target_info.dim,
        n_qubits=target_info.n_qubits,
    )
    cache[key] = cached
    return cached


# -----------------------------------------------------------------------------
# One restarted VQLS window
# -----------------------------------------------------------------------------


def should_compute_exact_for_window(args: argparse.Namespace, window_index: int, n_windows: int) -> bool:
    if args.skip_exact:
        return False
    if window_index == n_windows:
        return True
    if args.exact_every <= 0:
        return False
    return window_index % args.exact_every == 0


def initial_theta_for_window(
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
    n_params: int,
    theta_previous: np.ndarray | None,
    retry_index: int,
) -> np.ndarray:
    warm_key = args.theta_warm_start.lower().replace("-", "_")
    if retry_index == 0 and warm_key == "previous" and theta_previous is not None and theta_previous.shape == (n_params,):
        theta = theta_previous.copy()
        if args.theta_jitter > 0:
            theta += args.theta_jitter * rng.standard_normal(n_params)
        return theta
    return args.init_scale * rng.standard_normal(n_params)


def solve_window_once(
    *,
    args: argparse.Namespace,
    bcow_module,
    vqls_module,
    qml,
    system,
    cached: CachedOperators,
    y_start_padded: np.ndarray,
    t_start: float,
    t_end: float,
    window_index: int,
    n_windows: int,
    retry_index: int,
    rng: np.random.Generator,
    theta_previous: np.ndarray | None,
) -> WindowResult:
    layout = cached.layout
    rhs = bcow_module.build_bcow_rhs(y_start_padded, layout)
    target_info = vqls_module.build_vqls_target(cached.C, rhs, args.target, bcow_module)

    recurrence_history: np.ndarray | None = None
    initial_guess_key = args.initial_state_guess.lower().replace("-", "_")
    if args.compare_direct or initial_guess_key in {"recurrence", "reference", "reference_solution"} or args.carry_mode == "recurrence":
        recurrence_history = bcow_module.solve_bcow_by_recurrence(system.A_sparse_padded, y_start_padded, layout)

    t_ham_start = time.perf_counter()
    H_num_qml, num_meta = build_numerator_hamiltonian(
        target_info.A_target,
        target_info.rhs_target,
        coefficient_tol=args.coefficient_tol,
        max_qubits=args.max_vqls_qubits,
        vqls_module=vqls_module,
    )
    hamiltonian_build_seconds = float(cached.denominator_metadata.get("denominator_build_seconds", 0.0)) + float(
        num_meta.get("numerator_build_seconds", 0.0)
    )
    # Wall time spent in this call only; denominator may be cached from an earlier window.
    numerator_build_seconds_this_call = time.perf_counter() - t_ham_start

    init_state = vqls_module.make_initial_state(
        target_info.dim,
        args.initial_state_guess,
        rng,
        target_info.rhs_target,
        recurrence_history=recurrence_history,
        target=target_info.target,
    )

    n_params = parameter_count(args.ansatz, args.layers, target_info.n_qubits, vqls_module)
    theta0 = initial_theta_for_window(
        args=args,
        rng=rng,
        n_params=n_params,
        theta_previous=theta_previous,
        retry_index=retry_index,
    )

    estimator = RestartedPennyLaneVQLS(
        n_qubits=target_info.n_qubits,
        initial_state=init_state,
        ansatz=args.ansatz,
        layers=args.layers,
        H_denominator=cached.H_den_qml,
        H_numerator=H_num_qml,
        device_name=args.device,
        seed=args.rng_seed + 10000 * window_index + 101 * retry_index,
        allow_device_fallback=not args.no_device_fallback,
        qml=qml,
        vqls_module=vqls_module,
    )

    def objective(theta: np.ndarray) -> float:
        return float(estimator.cost_metrics(theta)["cost"])

    initial_metrics = estimator.cost_metrics(theta0)
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
        if args.verbose_windows and (
            iteration == 0
            or iteration == 1
            or (args.progress_interval > 0 and iteration % args.progress_interval == 0)
        ):
            print(
                f"  window {window_index:04d} attempt {retry_index}: iter={iteration}, "
                f"cost={record['cost']:.8e}, sqrt={record['sqrt_cost']:.8e}, "
                f"best={record['best_cost']:.8e}"
            )

    opt_start = time.perf_counter()
    opt_key = args.optimizer.lower().replace("_", "-")
    if opt_key == "spsa":
        opt = vqls_module.spsa_minimize(
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
        opt = vqls_module.scipy_minimize(
            objective=objective,
            theta0=theta0,
            method="Powell",
            maxiter=args.maxiter,
            progress_interval=args.progress_interval,
            progress_callback=progress_callback,
        )
    elif opt_key in {"scipy-lbfgsb", "lbfgsb", "l-bfgs-b"}:
        opt = vqls_module.scipy_minimize(
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
    alpha = vqls_module.optimal_scale(target_info.A_target, psi_opt, target_info.rhs_target)
    solution_target = alpha * psi_opt
    history_vec = vqls_module.extract_history_from_target_solution(solution_target, target_info)
    history = history_vec.reshape((layout.padded_clock_dim, layout.system_dim))
    final_approx_padded, history_norm, final_subspace_probability, final_copy_relative_mismatch = bcow_module.extract_final_vector(
        history,
        layout,
        mode=args.extract_final,
    )

    residual_vec = target_info.A_target @ solution_target - target_info.rhs_target
    relative_residual = float(np.linalg.norm(residual_vec) / max(np.linalg.norm(target_info.rhs_target), 1e-15))

    relative_history_error_vs_recurrence: float | None = None
    relative_final_error_vs_recurrence: float | None = None
    if recurrence_history is not None:
        recurrence_vec = recurrence_history.reshape(-1)
        relative_history_error_vs_recurrence = float(
            np.linalg.norm(history_vec - recurrence_vec) / max(np.linalg.norm(recurrence_vec), 1e-15)
        )
        recurrence_final_padded, *_ = bcow_module.extract_final_vector(recurrence_history, layout, mode=args.extract_final)
        relative_final_error_vs_recurrence = float(
            np.linalg.norm(final_approx_padded - recurrence_final_padded)
            / max(np.linalg.norm(recurrence_final_padded), 1e-15)
        )

    relative_local_error_vs_expm: float | None = None
    if should_compute_exact_for_window(args, window_index, n_windows):
        local_exact_padded = bcow_module.expm_times_vec(system.A_sparse_padded, t_end - t_start, y_start_padded)
        relative_local_error_vs_expm = float(
            np.linalg.norm(final_approx_padded - local_exact_padded) / max(np.linalg.norm(local_exact_padded), 1e-15)
        )

    # For diagnostic modes only; normal research path is carry_mode='vqls'.
    carry_mode = args.carry_mode.lower().replace("-", "_")
    if carry_mode == "vqls":
        y_end_padded = final_approx_padded
    elif carry_mode == "recurrence":
        if recurrence_history is None:
            recurrence_history = bcow_module.solve_bcow_by_recurrence(system.A_sparse_padded, y_start_padded, layout)
        y_end_padded, *_ = bcow_module.extract_final_vector(recurrence_history, layout, mode=args.extract_final)
    elif carry_mode == "expm":
        y_end_padded = bcow_module.expm_times_vec(system.A_sparse_padded, t_end - t_start, y_start_padded)
    else:
        raise ValueError("carry_mode must be 'vqls', 'recurrence', or 'expm'.")

    return WindowResult(
        window_index=window_index,
        attempt_index=retry_index,
        t_start=float(t_start),
        t_end=float(t_end),
        dt=float(t_end - t_start),
        segments=int(layout.segments),
        taylor_order=int(layout.taylor_order),
        padding_steps=int(layout.padding_steps),
        vqls_qubits=int(target_info.n_qubits),
        ansatz_parameter_count=int(n_params),
        initial_cost=float(initial_metrics["cost"]),
        initial_sqrt_cost=float(initial_metrics["sqrt_cost"]),
        final_cost=float(final_metrics["cost"]),
        final_sqrt_cost=float(final_metrics["sqrt_cost"]),
        relative_residual=relative_residual,
        alpha_real=float(np.real(alpha)),
        alpha_imag=float(np.imag(alpha)),
        history_norm=float(history_norm),
        final_subspace_probability=float(final_subspace_probability),
        final_copy_relative_mismatch=float(final_copy_relative_mismatch),
        relative_history_error_vs_recurrence=relative_history_error_vs_recurrence,
        relative_final_error_vs_recurrence=relative_final_error_vs_recurrence,
        relative_local_error_vs_expm=relative_local_error_vs_expm,
        optimizer_success=bool(opt.success),
        optimizer_message=str(opt.message),
        optimizer_iterations=int(opt.nit),
        optimizer_wall_time_seconds=float(opt_seconds),
        hamiltonian_build_seconds=float(hamiltonian_build_seconds),
        pauli_terms_denominator=int(cached.denominator_metadata["pauli_terms_denominator"]),
        pauli_terms_numerator=int(num_meta["pauli_terms_numerator"]),
        theta_opt=np.asarray(opt.x, dtype=float),
        y_end_padded=np.asarray(y_end_padded, dtype=complex),
        history_vec=np.asarray(history_vec, dtype=complex),
        progress_records=progress_records,
        optimizer_history=opt.history,
    )


def solve_window_with_retries(
    *,
    args: argparse.Namespace,
    bcow_module,
    vqls_module,
    qml,
    system,
    cached: CachedOperators,
    y_start_padded: np.ndarray,
    t_start: float,
    t_end: float,
    window_index: int,
    n_windows: int,
    rng: np.random.Generator,
    theta_previous: np.ndarray | None,
) -> WindowResult:
    best_result: WindowResult | None = None
    for retry_index in range(args.window_retries + 1):
        result = solve_window_once(
            args=args,
            bcow_module=bcow_module,
            vqls_module=vqls_module,
            qml=qml,
            system=system,
            cached=cached,
            y_start_padded=y_start_padded,
            t_start=t_start,
            t_end=t_end,
            window_index=window_index,
            n_windows=n_windows,
            retry_index=retry_index,
            rng=rng,
            theta_previous=theta_previous,
        )
        if best_result is None or result.relative_residual < best_result.relative_residual:
            best_result = result
        if result.relative_residual <= args.residual_target:
            break
        if retry_index < args.window_retries:
            print(
                f"  window {window_index:04d}: residual {result.relative_residual:.6e} > "
                f"target {args.residual_target:.6e}; retrying with a fresh theta."
            )

    assert best_result is not None
    if best_result.relative_residual > args.residual_target:
        message = (
            f"window {window_index} best residual {best_result.relative_residual:.6e} "
            f"exceeds --residual-target {args.residual_target:.6e}"
        )
        if args.stop_on_window_failure:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return best_result


# -----------------------------------------------------------------------------
# Restarted orchestration and outputs
# -----------------------------------------------------------------------------


def build_time_grid(t_final: float, restart_window: float, restart_steps: int) -> np.ndarray:
    if abs(float(t_final)) <= 0.0:
        return np.array([0.0], dtype=float)
    if restart_steps > 0:
        n_windows = int(restart_steps)
    else:
        if restart_window <= 0:
            raise ValueError("--restart-window must be positive when --restart-steps is not set.")
        n_windows = max(1, int(math.ceil(abs(float(t_final)) / abs(float(restart_window)))))
    return np.linspace(0.0, float(t_final), n_windows + 1)


def resolve_segments_for_window(args: argparse.Namespace, bcow_module, A: sp.spmatrix, dt: float) -> tuple[int, float]:
    if args.segments == 0:
        return bcow_module.choose_segments_from_norm(
            A,
            dt,
            segment_norm_bound=args.segment_norm_bound,
            exact_norm_dim_limit=args.norm_exact_dim_limit,
        )
    norm_A = bcow_module.estimate_operator_norm(A, exact_dim_limit=args.norm_exact_dim_limit)
    return int(args.segments), norm_A


def write_summary_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    preferred = [
        "window_index",
        "attempt_index",
        "t_start",
        "t_end",
        "dt",
        "segments",
        "taylor_order",
        "padding_steps",
        "vqls_qubits",
        "ansatz_parameter_count",
        "initial_sqrt_cost",
        "final_sqrt_cost",
        "relative_residual",
        "final_subspace_probability",
        "final_copy_relative_mismatch",
        "relative_history_error_vs_recurrence",
        "relative_final_error_vs_recurrence",
        "relative_local_error_vs_expm",
        "global_relative_error_vs_expm",
        "global_state_distance_vs_expm",
        "global_overlap_abs_vs_expm",
        "optimizer_success",
        "optimizer_iterations",
        "optimizer_wall_time_seconds",
        "pauli_terms_denominator",
        "pauli_terms_numerator",
    ]
    fieldnames = [name for name in preferred if any(name in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_restarted_outputs(
    *,
    args: argparse.Namespace,
    system,
    metadata: dict[str, object],
    window_results: Sequence[WindowResult],
    restart_states: np.ndarray,
    final_exact_original: np.ndarray | None,
) -> tuple[Path, Path, Path, Path | None]:
    prefix = Path(args.output_prefix)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    summary_csv_path = prefix.with_name(prefix.name + "_window_summary.csv")
    displacement_csv_path: Path | None = None

    final_padded = restart_states[-1]
    final_original = final_padded[: system.state_dim_original]
    n_mech = system.n_mech
    u_approx = np.real_if_close(final_original[:n_mech], tol=1e8)
    v_approx = np.real_if_close(final_original[n_mech:], tol=1e8)

    payload: dict[str, object] = {
        "metadata": json.dumps(metadata, indent=2),
        "restart_times": np.asarray(metadata["restart_times"], dtype=float),
        "restart_states_padded": np.asarray(restart_states, dtype=complex),
        "x_approx_padded": np.asarray(final_padded, dtype=complex),
        "x_approx": np.asarray(final_original, dtype=complex),
        "u_approx": np.asarray(u_approx),
        "v_approx": np.asarray(v_approx),
        "A_original": np.asarray(system.A_original),
        "y0_original": np.asarray(system.y0_original),
        "X_all": np.asarray(system.X_all),
    }

    if window_results:
        payload["theta_last"] = np.asarray(window_results[-1].theta_opt, dtype=float)
        payload["window_final_residuals"] = np.asarray([w.relative_residual for w in window_results], dtype=float)
        payload["window_final_sqrt_costs"] = np.asarray([w.final_sqrt_cost for w in window_results], dtype=float)
        payload["window_final_subspace_probabilities"] = np.asarray(
            [w.final_subspace_probability for w in window_results], dtype=float
        )
        payload["window_final_copy_mismatches"] = np.asarray(
            [w.final_copy_relative_mismatch for w in window_results], dtype=float
        )
        if args.save_window_histories:
            for w in window_results:
                payload[f"history_window_{w.window_index:04d}"] = np.asarray(w.history_vec, dtype=complex)

    if final_exact_original is not None:
        payload["x_exact"] = np.asarray(final_exact_original, dtype=complex)
        payload["u_exact"] = np.real_if_close(final_exact_original[:n_mech], tol=1e8)
        payload["v_exact"] = np.real_if_close(final_exact_original[n_mech:], tol=1e8)
        table = np.column_stack([system.X_all, payload["u_exact"], u_approx])
        displacement_csv_path = prefix.with_name(prefix.name + "_displacement.csv")
        np.savetxt(displacement_csv_path, table, delimiter=",", header="X_all,u_exact,u_restarted_vqls", comments="")

    payload["window_summaries_json"] = json.dumps(metadata["window_summaries"], indent=2)
    np.savez_compressed(npz_path, **payload)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    write_summary_csv(summary_csv_path, metadata["window_summaries"])
    return npz_path, json_path, summary_csv_path, displacement_csv_path


def solve_restarted_bcow_vqls(args: argparse.Namespace) -> dict[str, object]:
    vqls_module = load_module_from_path(args.vqls_module_path, "bcow_vqls_single_window_module")
    bcow_module = load_module_from_path(args.bcow_module_path, "bcow_dynamic_module_restarted")
    qml = vqls_module.qml

    if args.energy_estimator != "pauli_hamiltonian":
        raise ValueError("This restarted driver currently supports --energy-estimator pauli_hamiltonian only.")
    if args.taylor_order < 5:
        warnings.warn(
            "BCOW's condition/error analysis is usually stated for k >= 5. "
            "Small k is useful for VQLS smoke tests but not recommended for final physics runs.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.u_right is not None:
        warnings.warn(
            "--u-right is accepted only for CLI compatibility with the static VQLS script; ignored here.",
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

    times = build_time_grid(args.t, args.restart_window, args.restart_steps)
    n_windows = len(times) - 1
    if n_windows < 1:
        raise ValueError("The requested final time creates zero restart windows.")

    norm_A = bcow_module.estimate_operator_norm(system.A_sparse_padded, exact_dim_limit=args.norm_exact_dim_limit)
    print("PennyLane version:", qml.__version__)
    print("Algorithm: restarted BCOW + PennyLane VQLS")
    print("Mechanics case: unchanged 1D dynamic MD-FE coupling")
    print("Mechanical DOFs:", system.n_mech)
    print("Original state dimension:", system.state_dim_original)
    print("Padded ODE system dimension:", system.state_dim_padded)
    print("Total final time T:", f"{args.t:.8e}")
    print("Restart windows:", n_windows)
    print("Nominal local dt:", f"{times[1] - times[0]:.8e}")
    print("Estimated ||A||:", f"{norm_A:.8e}")
    print("Per-window BCOW segments mode:", "automatic" if args.segments == 0 else args.segments)
    print("BCOW Taylor order k:", args.taylor_order)
    print("BCOW padding steps p:", args.padding_steps)
    print("VQLS target:", args.target)
    print("PennyLane device requested:", args.device)
    print("Energy estimator:", args.energy_estimator)
    print("Optimizer:", args.optimizer)
    print("Ansatz:", args.ansatz)
    print("Ansatz layers:", args.layers)
    print("Initial state guess per window:", args.initial_state_guess)
    print("Theta warm start:", args.theta_warm_start)
    print("Carry mode:", args.carry_mode)
    print("Shots: analytic mode; no finite shot count is set")

    y_current = np.asarray(system.y0_padded, dtype=complex).copy()
    restart_states = [y_current.copy()]
    theta_previous: np.ndarray | None = None
    window_results: list[WindowResult] = []
    window_summaries: list[dict[str, object]] = []
    operator_cache: dict[tuple[object, ...], CachedOperators] = {}

    for idx in range(1, n_windows + 1):
        t_start = float(times[idx - 1])
        t_end = float(times[idx])
        dt = t_end - t_start
        segments, norm_for_segments = resolve_segments_for_window(args, bcow_module, system.A_sparse_padded, dt)
        if segments < 1:
            raise ValueError("Per-window segments must be >= 1, or 0 for automatic selection.")
        padding_steps = int(args.padding_steps)
        if padding_steps < 0:
            padding_steps = segments

        cached = get_cached_operators(
            cache=operator_cache,
            args=args,
            bcow_module=bcow_module,
            vqls_module=vqls_module,
            system=system,
            dt=dt,
            segments=segments,
            padding_steps=padding_steps,
        )

        print(
            f"\nWindow {idx}/{n_windows}: t=[{t_start:.8e}, {t_end:.8e}], "
            f"dt={dt:.8e}, m={segments}, k={args.taylor_order}, p={padding_steps}, "
            f"||A dt/m||≈{norm_for_segments * abs(dt / segments):.8e}, "
            f"clock_blocks={cached.layout.padded_clock_dim}, qubits={cached.n_qubits}"
        )
        print(
            f"  C shape={cached.C.shape}, nnz={cached.C.nnz}, "
            f"denominator Pauli terms={cached.denominator_metadata['pauli_terms_denominator']}"
        )

        result = solve_window_with_retries(
            args=args,
            bcow_module=bcow_module,
            vqls_module=vqls_module,
            qml=qml,
            system=system,
            cached=cached,
            y_start_padded=y_current,
            t_start=t_start,
            t_end=t_end,
            window_index=idx,
            n_windows=n_windows,
            rng=rng,
            theta_previous=theta_previous,
        )
        y_current = result.y_end_padded.copy()
        restart_states.append(y_current.copy())
        theta_previous = result.theta_opt.copy()
        window_results.append(result)

        summary = result.summary_dict()
        if should_compute_exact_for_window(args, idx, n_windows):
            x_exact_global = bcow_module.expm_times_vec(system.A_sparse_original, t_end, system.y0_original)
            approx_original = y_current[: system.state_dim_original]
            global_rel_error = float(
                np.linalg.norm(approx_original - x_exact_global) / max(np.linalg.norm(x_exact_global), 1e-15)
            )
            global_state_distance, global_overlap_abs = vqls_module.normalized_state_metrics(approx_original, x_exact_global)
            summary["global_relative_error_vs_expm"] = global_rel_error
            summary["global_state_distance_vs_expm"] = global_state_distance
            summary["global_overlap_abs_vs_expm"] = global_overlap_abs
        else:
            summary["global_relative_error_vs_expm"] = None
            summary["global_state_distance_vs_expm"] = None
            summary["global_overlap_abs_vs_expm"] = None
        window_summaries.append(summary)

        print(
            f"  result: sqrt(cost)={result.final_sqrt_cost:.8e}, "
            f"residual={result.relative_residual:.8e}, "
            f"final_prob={result.final_subspace_probability:.8e}, "
            f"copy_mismatch={result.final_copy_relative_mismatch:.8e}"
        )
        if result.relative_final_error_vs_recurrence is not None:
            print("  final error vs local BCOW recurrence:", f"{result.relative_final_error_vs_recurrence:.8e}")
        if summary.get("global_relative_error_vs_expm") is not None:
            print("  cumulative final error vs expm_multiply:", f"{summary['global_relative_error_vs_expm']:.8e}")

    restart_states_array = np.vstack([state.reshape(1, -1) for state in restart_states])

    final_exact_original: np.ndarray | None = None
    final_relative_error_vs_expm: float | None = None
    final_state_distance_vs_expm: float | None = None
    final_overlap_abs_vs_expm: float | None = None
    if not args.skip_exact:
        final_exact_original = bcow_module.expm_times_vec(system.A_sparse_original, args.t, system.y0_original)
        final_original = y_current[: system.state_dim_original]
        final_relative_error_vs_expm = float(
            np.linalg.norm(final_original - final_exact_original) / max(np.linalg.norm(final_exact_original), 1e-15)
        )
        final_state_distance_vs_expm, final_overlap_abs_vs_expm = vqls_module.normalized_state_metrics(
            final_original,
            final_exact_original,
        )

    diagnostics = {
        "n_windows": n_windows,
        "max_window_residual": float(max(w.relative_residual for w in window_results)),
        "mean_window_residual": float(np.mean([w.relative_residual for w in window_results])),
        "max_window_sqrt_cost": float(max(w.final_sqrt_cost for w in window_results)),
        "mean_window_sqrt_cost": float(np.mean([w.final_sqrt_cost for w in window_results])),
        "min_final_subspace_probability": float(min(w.final_subspace_probability for w in window_results)),
        "max_final_copy_relative_mismatch": float(max(w.final_copy_relative_mismatch for w in window_results)),
        "final_relative_error_vs_expm": final_relative_error_vs_expm,
        "final_state_distance_vs_expm": final_state_distance_vs_expm,
        "final_overlap_abs_vs_expm": final_overlap_abs_vs_expm,
        "wall_time_seconds": float(time.perf_counter() - total_start),
    }

    metadata: dict[str, object] = {
        "algorithm": "Restarted BCOW + shot-free PennyLane VQLS residual simulation",
        "mechanics_case": "1D dynamic MD-FE coupling, y_dot = A y, y=[u;v]",
        "mechanics_builder_changed": False,
        "bcow_module_path": str(Path(args.bcow_module_path).resolve()),
        "vqls_module_path": str(Path(args.vqls_module_path).resolve()),
        "pennylane_version": qml.__version__,
        "device_requested": args.device,
        "shot_mode": "analytic_shots_none",
        "vqls_target": args.target,
        "energy_estimator": args.energy_estimator,
        "optimizer": args.optimizer,
        "ansatz": args.ansatz,
        "ansatz_layers": args.layers,
        "initial_state_guess": args.initial_state_guess,
        "theta_warm_start": args.theta_warm_start,
        "theta_jitter": args.theta_jitter,
        "carry_mode": args.carry_mode,
        "residual_target": args.residual_target,
        "window_retries": args.window_retries,
        "n1": args.n1,
        "n2": args.n2,
        "mechanical_dofs": system.n_mech,
        "state_dim_original": system.state_dim_original,
        "state_dim_padded": system.state_dim_padded,
        "sparsity_A_original_nnz": int(system.A_sparse_original.nnz),
        "sparsity_A_padded_nnz": int(system.A_sparse_padded.nnz),
        "estimated_A_norm": float(norm_A),
        "t_final": float(args.t),
        "restart_window_requested": float(args.restart_window),
        "restart_steps_requested": int(args.restart_steps),
        "restart_times": [float(x) for x in times],
        "segments_argument": int(args.segments),
        "segment_norm_bound": float(args.segment_norm_bound),
        "taylor_order": int(args.taylor_order),
        "padding_steps_argument": int(args.padding_steps),
        "extract_final": args.extract_final,
        "max_vqls_qubits": int(args.max_vqls_qubits),
        "coefficient_tol": float(args.coefficient_tol),
        "diagnostics": diagnostics,
        "window_summaries": window_summaries,
    }

    npz_path, json_path, summary_csv_path, displacement_csv_path = save_restarted_outputs(
        args=args,
        system=system,
        metadata=metadata,
        window_results=window_results,
        restart_states=restart_states_array,
        final_exact_original=final_exact_original,
    )

    print("\nFinal restarted BCOW+VQLS results")
    print("Total windows:", n_windows)
    print("Max window residual:", f"{diagnostics['max_window_residual']:.8e}")
    print("Mean window residual:", f"{diagnostics['mean_window_residual']:.8e}")
    print("Max window sqrt(cost):", f"{diagnostics['max_window_sqrt_cost']:.8e}")
    print("Min final clock-subspace probability:", f"{diagnostics['min_final_subspace_probability']:.8e}")
    print("Max final copy relative mismatch:", f"{diagnostics['max_final_copy_relative_mismatch']:.8e}")
    if final_relative_error_vs_expm is not None:
        print("Final cumulative error vs expm_multiply:", f"{final_relative_error_vs_expm:.8e}")
        print("Final normalized state distance vs expm_multiply:", f"{final_state_distance_vs_expm:.8e}")
        print("Final |normalized overlap| vs expm_multiply:", f"{final_overlap_abs_vs_expm:.8e}")
    print("Saved numerical results to", npz_path)
    print("Saved metadata JSON to", json_path)
    print("Saved per-window summary CSV to", summary_csv_path)
    if displacement_csv_path is not None:
        print("Saved displacement CSV to", displacement_csv_path)

    return {
        "metadata": metadata,
        "npz_path": npz_path,
        "json_path": json_path,
        "summary_csv_path": summary_csv_path,
        "displacement_csv_path": displacement_csv_path,
        "diagnostics": diagnostics,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restarted/rolling-horizon BCOW + shot-free PennyLane VQLS for the 1D dynamic MD-FE ODE system."
    )

    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--bcow-module-path", type=str, default=str(script_dir / "1Ddynamic_BCOW_ODE_solver.py"))
    parser.add_argument("--vqls-module-path", type=str, default=str(script_dir / "1Ddynamic_BCOW_VQLS_pennylane.py"))

    # Dynamic mechanics arguments.  --md-atoms/--fe-elems are aliases for continuity.
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

    # Restart/time layout arguments.
    parser.add_argument("--t", type=float, default=10.0, help="Total final time T.")
    parser.add_argument(
        "--restart-window",
        type=float,
        default=0.1,
        help="Preferred local window length. The code uses ceil(|T|/restart_window) equal windows.",
    )
    parser.add_argument(
        "--restart-steps",
        type=int,
        default=0,
        help="Number of equal restart windows. If >0, overrides --restart-window.",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=1,
        help="Per-window BCOW segment count m. Use 0 for automatic selection per window.",
    )
    parser.add_argument("--segment-norm-bound", type=float, default=1.0)
    parser.add_argument("--norm-exact-dim-limit", type=int, default=0)
    parser.add_argument("--taylor-order", type=int, default=4)
    parser.add_argument("--padding-steps", type=int, default=1, help="Per-window final-time copy count p. Use -1 to set p=m.")
    parser.add_argument("--no-pad-clock", action="store_true")
    parser.add_argument("--extract-final", choices=["first", "last", "average"], default="average")

    # VQLS arguments.
    parser.add_argument("--target", choices=["direct-c", "hermitian-dilation"], default="direct-c")
    parser.add_argument("--device", type=str, default="lightning.gpu")
    parser.add_argument("--energy-estimator", choices=["pauli_hamiltonian"], default="pauli_hamiltonian")
    parser.add_argument("--coefficient-tol", type=float, default=1e-10)
    parser.add_argument("--max-vqls-qubits", type=int, default=10)
    parser.add_argument("--no-device-fallback", action="store_true")

    parser.add_argument(
        "--ansatz",
        choices=["hardware_efficient", "real_ry", "real_ry_identity", "hardware_efficient_identity"],
        default="real_ry_identity",
    )
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--init-scale", type=float, default=0.0)
    parser.add_argument(
        "--initial-state-guess",
        type=str,
        default="rhs",
        choices=["zero", "uniform", "index_ramp", "random_complex", "rhs", "recurrence", "reference_solution"],
    )
    parser.add_argument(
        "--theta-warm-start",
        choices=["none", "previous"],
        default="previous",
        help="Use the previous window's optimized theta as the next initial theta when dimensions match.",
    )
    parser.add_argument("--theta-jitter", type=float, default=0.0, help="Gaussian noise added to previous theta warm-start.")

    parser.add_argument("--optimizer", choices=["spsa", "scipy-powell", "scipy-lbfgsb"], default="scipy-lbfgsb")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--spsa-a", type=float, default=0.05)
    parser.add_argument("--spsa-c", type=float, default=0.08)
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--progress-interval", type=int, default=25)

    # Restart control/diagnostics.
    parser.add_argument("--residual-target", type=float, default=5e-2)
    parser.add_argument("--window-retries", type=int, default=0)
    parser.add_argument("--stop-on-window-failure", action="store_true")
    parser.add_argument(
        "--carry-mode",
        choices=["vqls", "recurrence", "expm"],
        default="vqls",
        help="Use vqls for the research route. recurrence/expm are diagnostics to isolate error accumulation.",
    )
    parser.add_argument("--compare-direct", action="store_true", help="Compute local BCOW recurrence for per-window diagnostics.")
    parser.add_argument("--skip-exact", action="store_true", help="Skip expm_multiply cumulative/global references.")
    parser.add_argument(
        "--exact-every",
        type=int,
        default=10,
        help="Compute cumulative expm_multiply diagnostics every N windows and at the final window. Use 0 for final only.",
    )
    parser.add_argument("--save-window-histories", action="store_true", help="Store every optimized BCOW history vector in the NPZ.")
    parser.add_argument("--verbose-windows", action="store_true", help="Print optimizer progress inside each window.")
    parser.add_argument("--output-prefix", type=str, default="bcow_vqls_restarted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solve_restarted_bcow_vqls(args)


if __name__ == "__main__":
    main()
