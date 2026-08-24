#!/usr/bin/env python3
"""
Boundary-reduced restarted BCOW + VQLS PennyLane simulation for the 1D dynamic
MD-FE ODE problem.

This script keeps the user's preferred route:

    ODE  ->  linear system  ->  VQLS solve

but uses a special structural property of the BCOW matrix before VQLS sees the
linear system.  In the homogeneous case y_dot = A y, the BCOW Taylor variables
inside each segment satisfy

    x_{i,j} = (h A)^j / j!  x_{i,0},       j = 1,...,k,

and the segment-boundary variable satisfies

    x_{i+1,0} = sum_{j=0}^k x_{i,j}
              = P_k(h A) x_{i,0},

where

    P_k(h A) = sum_{j=0}^k (h A)^j / j!.

Therefore the full BCOW history linear system can be compressed to a smaller
boundary-only linear system

    C_red |boundary_history> = |rhs>,

with

    C_red = I - sum_{i=0}^{m-1} |i+1><i| ⊗ P_k(h A),
    rhs   = |0> ⊗ y_start.

For one restarted window with m=1, the VQLS target is only two system blocks:

    [ I   0 ] [y_start] = [y_start]
    [-P   I ] [y_end  ]   [0      ]

This removes the k internal Taylor blocks and the BCOW final-copy padding blocks
from the VQLS unknown vector.  It is still a linear-system VQLS method; it just
uses the BCOW matrix structure instead of treating C as a fully generic matrix.

Keep this file in the same directory as:

    1Ddynamic_BCOW_ODE_solver.py
    1Ddynamic_BCOW_VQLS_pennylane.py
    1Ddynamic_BCOW_VQLS_restarted_pennylane.py

Example: first retest the previously hard t=0.5 case with five dt=0.1 windows
and a much smaller boundary-reduced VQLS system:

    python 1Ddynamic_BCOW_VQLS_boundary_reduced_pennylane.py \
        --no-pad --no-device-fallback \
        --target direct-c \
        --device lightning.gpu \
        --energy-estimator pauli_hamiltonian \
        --compare-direct \
        --n1 5 --n2 2 --pulse-atoms 3 \
        --t 0.5 --restart-window 0.1 \
        --segments 1 --taylor-order 4 \
        --coefficient-tol 1e-10 \
        --max-vqls-qubits 12 \
        --ansatz real_ry_identity --layers 8 \
        --init-scale 0.0 --initial-state-guess rhs \
        --theta-warm-start previous --theta-jitter 0.01 \
        --optimizer scipy-lbfgsb --maxiter 300 \
        --residual-target 1e-2 --window-retries 1 \
        --progress-interval 25 --exact-every 1 \
        --output-prefix boundary_reduced_t05_dt01

Then try t=10 with 100 short windows:

    python 1Ddynamic_BCOW_VQLS_boundary_reduced_pennylane.py \
        --no-pad --no-device-fallback \
        --target direct-c \
        --device lightning.gpu \
        --energy-estimator pauli_hamiltonian \
        --compare-direct \
        --n1 5 --n2 2 --pulse-atoms 3 \
        --t 10 --restart-window 0.1 \
        --segments 1 --taylor-order 4 \
        --coefficient-tol 1e-10 \
        --max-vqls-qubits 12 \
        --ansatz real_ry_identity --layers 8 \
        --init-scale 0.0 --initial-state-guess rhs \
        --theta-warm-start previous --theta-jitter 0.01 \
        --optimizer scipy-lbfgsb --maxiter 300 \
        --residual-target 1e-2 --window-retries 2 \
        --progress-interval 25 --exact-every 10 \
        --output-prefix boundary_reduced_t10_dt01
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
from typing import Sequence

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


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


# -----------------------------------------------------------------------------
# Boundary-reduced BCOW linear system
# -----------------------------------------------------------------------------


@dataclass
class BoundaryReducedLayout:
    segments: int
    taylor_order: int
    h_step: float
    boundary_clock_dim: int
    padded_boundary_clock_dim: int
    system_dim: int
    clock_qubits: int
    system_qubits: int
    total_vector_dim: int
    total_qubits_without_dilation: int
    total_qubits_with_hermitian_dilation: int


@dataclass
class ReducedOperators:
    layout: BoundaryReducedLayout
    P: sp.csr_matrix
    C: sp.csr_matrix
    H_den_qml: object
    denominator_metadata: dict[str, object]
    target_dim: int
    n_qubits: int


@dataclass
class ReducedWindowResult:
    window_index: int
    attempt_index: int
    t_start: float
    t_end: float
    dt: float
    segments: int
    taylor_order: int
    boundary_clock_dim: int
    padded_boundary_clock_dim: int
    vqls_qubits: int
    ansatz_parameter_count: int
    initial_cost: float
    initial_sqrt_cost: float
    final_cost: float
    final_sqrt_cost: float
    relative_residual: float
    alpha_real: float
    alpha_imag: float
    boundary_history_norm: float
    final_block_probability: float
    relative_boundary_error_vs_recurrence: float | None
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
    boundary_vec: np.ndarray
    progress_records: list[dict[str, float | int]]
    optimizer_history: list[dict[str, float | int]]

    def summary_dict(self) -> dict[str, object]:
        d = asdict(self)
        d.pop("theta_opt", None)
        d.pop("y_end_padded", None)
        d.pop("boundary_vec", None)
        d.pop("progress_records", None)
        d.pop("optimizer_history", None)
        return d


@dataclass
class GlobalMetric:
    window_index: int
    time: float
    global_relative_error_vs_expm: float
    global_state_distance_vs_expm: float
    global_overlap_abs_vs_expm: float



def build_reduced_layout(
    *,
    system_dim: int,
    t_final: float,
    segments: int,
    taylor_order: int,
    pad_clock_to_power2: bool = True,
) -> BoundaryReducedLayout:
    if segments < 1:
        raise ValueError("segments must be >= 1.")
    if taylor_order < 1:
        raise ValueError("taylor_order must be >= 1.")
    if system_dim < 1:
        raise ValueError("system_dim must be positive.")

    h_step = float(t_final) / int(segments)
    boundary_clock_dim = int(segments) + 1
    padded_boundary_clock_dim = next_power_of_two(boundary_clock_dim) if pad_clock_to_power2 else boundary_clock_dim
    if not pad_clock_to_power2 and padded_boundary_clock_dim & (padded_boundary_clock_dim - 1):
        warnings.warn(
            "The reduced boundary clock dimension is not a power of two. This is fine classically, "
            "but a quantum register normally requires padding.",
            RuntimeWarning,
            stacklevel=2,
        )
    clock_qubits = int(math.ceil(math.log2(padded_boundary_clock_dim))) if padded_boundary_clock_dim > 1 else 0
    system_qubits = int(math.ceil(math.log2(system_dim))) if system_dim > 1 else 0
    total_vector_dim = int(padded_boundary_clock_dim) * int(system_dim)
    return BoundaryReducedLayout(
        segments=int(segments),
        taylor_order=int(taylor_order),
        h_step=float(h_step),
        boundary_clock_dim=int(boundary_clock_dim),
        padded_boundary_clock_dim=int(padded_boundary_clock_dim),
        system_dim=int(system_dim),
        clock_qubits=int(clock_qubits),
        system_qubits=int(system_qubits),
        total_vector_dim=int(total_vector_dim),
        total_qubits_without_dilation=int(clock_qubits + system_qubits),
        total_qubits_with_hermitian_dilation=int(1 + clock_qubits + system_qubits),
    )



def build_taylor_propagator(A: sp.spmatrix, h: float, k: int, drop_tol: float = 0.0) -> sp.csr_matrix:
    """Build P_k(hA) = sum_{j=0}^k (hA)^j / j! as a sparse matrix.

    For the current small/medium PennyLane VQLS simulations this explicit sparse
    polynomial is acceptable.  It is still far smaller than the full BCOW history
    system because the k internal Taylor blocks are removed from the VQLS unknowns.
    """
    A = sp.csr_matrix(A, dtype=complex)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    n = A.shape[0]
    I = sp.eye(n, dtype=complex, format="csr")
    P = I.copy()
    term = I.copy()
    for j in range(1, int(k) + 1):
        term = (float(h) / float(j)) * (A @ term)
        if drop_tol > 0:
            term = term.tocsr()
            term.data[np.abs(term.data) < drop_tol] = 0.0
            term.eliminate_zeros()
        P = (P + term).tocsr()
    if drop_tol > 0:
        P.eliminate_zeros()
    return P.tocsr()



def build_boundary_reduced_matrix(P: sp.spmatrix, layout: BoundaryReducedLayout) -> sp.csr_matrix:
    """Build C_red = I - sum_i |i+1><i| tensor P_k(hA)."""
    P = sp.csr_matrix(P, dtype=complex)
    if P.shape != (layout.system_dim, layout.system_dim):
        raise ValueError("P shape does not match layout.system_dim.")
    cdim = layout.padded_boundary_clock_dim
    n = layout.system_dim
    eye_clock = sp.eye(cdim, dtype=complex, format="csr")
    eye_sys = sp.eye(n, dtype=complex, format="csr")

    rows = []
    cols = []
    data = []
    for i in range(layout.segments):
        rows.append(i + 1)
        cols.append(i)
        data.append(-1.0)
    shift = sp.coo_matrix((data, (rows, cols)), shape=(cdim, cdim), dtype=complex).tocsr()
    return (sp.kron(eye_clock, eye_sys, format="csr") + sp.kron(shift, P, format="csr")).tocsr()



def build_boundary_rhs(y0: np.ndarray, layout: BoundaryReducedLayout) -> np.ndarray:
    y0 = np.asarray(y0, dtype=complex)
    if y0.shape != (layout.system_dim,):
        raise ValueError("y0 has the wrong shape for the reduced layout.")
    rhs = np.zeros(layout.total_vector_dim, dtype=complex)
    rhs[: layout.system_dim] = y0
    return rhs



def solve_boundary_recurrence(P: sp.spmatrix, y0: np.ndarray, layout: BoundaryReducedLayout) -> np.ndarray:
    P = sp.csr_matrix(P, dtype=complex)
    y0 = np.asarray(y0, dtype=complex)
    boundary = np.zeros((layout.padded_boundary_clock_dim, layout.system_dim), dtype=complex)
    boundary[0] = y0
    for i in range(layout.segments):
        boundary[i + 1] = P @ boundary[i]
    return boundary



def extract_final_boundary_vector(boundary_vec: np.ndarray, layout: BoundaryReducedLayout) -> tuple[np.ndarray, float, float]:
    boundary = np.asarray(boundary_vec, dtype=complex).reshape((layout.padded_boundary_clock_dim, layout.system_dim))
    y_end = boundary[layout.segments].copy()
    hist_norm = float(np.linalg.norm(boundary.reshape(-1)))
    prob = 0.0 if hist_norm <= 1e-15 else float(np.linalg.norm(y_end) ** 2 / (hist_norm**2))
    return y_end, hist_norm, prob



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


# -----------------------------------------------------------------------------
# Operator caching and one-window VQLS solve
# -----------------------------------------------------------------------------


def operator_cache_key(args: argparse.Namespace, dt: float, segments: int) -> tuple[object, ...]:
    return (
        args.target,
        round(float(dt), 15),
        int(segments),
        int(args.taylor_order),
        bool(args.no_pad_clock),
        float(args.propagator_drop_tol),
    )



def get_reduced_operators(
    *,
    cache: dict[tuple[object, ...], ReducedOperators],
    args: argparse.Namespace,
    bcow_module,
    restarted_module,
    vqls_module,
    system,
    dt: float,
    segments: int,
) -> ReducedOperators:
    key = operator_cache_key(args, dt, segments)
    if key in cache:
        return cache[key]

    layout = build_reduced_layout(
        system_dim=system.state_dim_padded,
        t_final=dt,
        segments=segments,
        taylor_order=args.taylor_order,
        pad_clock_to_power2=not args.no_pad_clock,
    )
    P = build_taylor_propagator(system.A_sparse_padded, layout.h_step, layout.taylor_order, drop_tol=args.propagator_drop_tol)
    C = build_boundary_reduced_matrix(P, layout)
    dummy_rhs = build_boundary_rhs(np.ones(system.state_dim_padded, dtype=complex), layout)
    target_info = vqls_module.build_vqls_target(C, dummy_rhs, args.target, bcow_module)
    H_den_qml, denom_meta = restarted_module.build_denominator_hamiltonian(
        target_info.A_target,
        coefficient_tol=args.coefficient_tol,
        max_qubits=args.max_vqls_qubits,
        vqls_module=vqls_module,
    )
    reduced = ReducedOperators(
        layout=layout,
        P=P,
        C=C,
        H_den_qml=H_den_qml,
        denominator_metadata=denom_meta,
        target_dim=target_info.dim,
        n_qubits=target_info.n_qubits,
    )
    cache[key] = reduced
    return reduced



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
    restarted_module,
    vqls_module,
    qml,
    system,
    reduced: ReducedOperators,
    y_start_padded: np.ndarray,
    t_start: float,
    t_end: float,
    window_index: int,
    n_windows: int,
    retry_index: int,
    rng: np.random.Generator,
    theta_previous: np.ndarray | None,
) -> ReducedWindowResult:
    layout = reduced.layout
    rhs = build_boundary_rhs(y_start_padded, layout)
    target_info = vqls_module.build_vqls_target(reduced.C, rhs, args.target, bcow_module)

    recurrence_boundary: np.ndarray | None = None
    initial_guess_key = args.initial_state_guess.lower().replace("-", "_")
    if args.compare_direct or initial_guess_key in {"recurrence", "reference", "reference_solution", "boundary_recurrence"} or args.carry_mode == "recurrence":
        recurrence_boundary = solve_boundary_recurrence(reduced.P, y_start_padded, layout)

    t_ham_start = time.perf_counter()
    H_num_qml, num_meta = restarted_module.build_numerator_hamiltonian(
        target_info.A_target,
        target_info.rhs_target,
        coefficient_tol=args.coefficient_tol,
        max_qubits=args.max_vqls_qubits,
        vqls_module=vqls_module,
    )
    hamiltonian_build_seconds = float(reduced.denominator_metadata.get("denominator_build_seconds", 0.0)) + float(
        num_meta.get("numerator_build_seconds", 0.0)
    )
    numerator_build_seconds_this_call = time.perf_counter() - t_ham_start

    # The existing make_initial_state already knows how to embed a recurrence
    # vector into the Hermitian-dilation target.  Treat boundary_recurrence as
    # the same mode but with the reduced boundary history.
    init_guess_for_existing = args.initial_state_guess
    if initial_guess_key == "boundary_recurrence":
        init_guess_for_existing = "recurrence"

    init_state = vqls_module.make_initial_state(
        target_info.dim,
        init_guess_for_existing,
        rng,
        target_info.rhs_target,
        recurrence_history=recurrence_boundary,
        target=target_info.target,
    )

    n_params = restarted_module.parameter_count(args.ansatz, args.layers, target_info.n_qubits, vqls_module)
    theta0 = initial_theta_for_window(
        args=args,
        rng=rng,
        n_params=n_params,
        theta_previous=theta_previous,
        retry_index=retry_index,
    )

    estimator = restarted_module.RestartedPennyLaneVQLS(
        n_qubits=target_info.n_qubits,
        initial_state=init_state,
        ansatz=args.ansatz,
        layers=args.layers,
        H_denominator=reduced.H_den_qml,
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
    boundary_vec = vqls_module.extract_history_from_target_solution(solution_target, target_info)
    y_end_approx, boundary_norm, final_block_probability = extract_final_boundary_vector(boundary_vec, layout)

    residual_vec = target_info.A_target @ solution_target - target_info.rhs_target
    relative_residual = float(np.linalg.norm(residual_vec) / max(np.linalg.norm(target_info.rhs_target), 1e-15))

    relative_boundary_error_vs_recurrence: float | None = None
    relative_final_error_vs_recurrence: float | None = None
    if recurrence_boundary is not None:
        recurrence_vec = recurrence_boundary.reshape(-1)
        relative_boundary_error_vs_recurrence = float(
            np.linalg.norm(boundary_vec - recurrence_vec) / max(np.linalg.norm(recurrence_vec), 1e-15)
        )
        recurrence_final, *_ = extract_final_boundary_vector(recurrence_vec, layout)
        relative_final_error_vs_recurrence = float(
            np.linalg.norm(y_end_approx - recurrence_final) / max(np.linalg.norm(recurrence_final), 1e-15)
        )

    relative_local_error_vs_expm: float | None = None
    if should_compute_exact_for_window(args, window_index, n_windows):
        local_exact_padded = bcow_module.expm_times_vec(system.A_sparse_padded, t_end - t_start, y_start_padded)
        relative_local_error_vs_expm = float(
            np.linalg.norm(y_end_approx - local_exact_padded) / max(np.linalg.norm(local_exact_padded), 1e-15)
        )

    carry_mode = args.carry_mode.lower().replace("-", "_")
    if carry_mode == "vqls":
        y_end_padded = y_end_approx
    elif carry_mode == "recurrence":
        if recurrence_boundary is None:
            recurrence_boundary = solve_boundary_recurrence(reduced.P, y_start_padded, layout)
        y_end_padded, *_ = extract_final_boundary_vector(recurrence_boundary.reshape(-1), layout)
    elif carry_mode == "expm":
        y_end_padded = bcow_module.expm_times_vec(system.A_sparse_padded, t_end - t_start, y_start_padded)
    else:
        raise ValueError("carry_mode must be 'vqls', 'recurrence', or 'expm'.")

    return ReducedWindowResult(
        window_index=int(window_index),
        attempt_index=int(retry_index),
        t_start=float(t_start),
        t_end=float(t_end),
        dt=float(t_end - t_start),
        segments=int(layout.segments),
        taylor_order=int(layout.taylor_order),
        boundary_clock_dim=int(layout.boundary_clock_dim),
        padded_boundary_clock_dim=int(layout.padded_boundary_clock_dim),
        vqls_qubits=int(target_info.n_qubits),
        ansatz_parameter_count=int(n_params),
        initial_cost=float(initial_metrics["cost"]),
        initial_sqrt_cost=float(initial_metrics["sqrt_cost"]),
        final_cost=float(final_metrics["cost"]),
        final_sqrt_cost=float(final_metrics["sqrt_cost"]),
        relative_residual=float(relative_residual),
        alpha_real=float(np.real(alpha)),
        alpha_imag=float(np.imag(alpha)),
        boundary_history_norm=float(boundary_norm),
        final_block_probability=float(final_block_probability),
        relative_boundary_error_vs_recurrence=relative_boundary_error_vs_recurrence,
        relative_final_error_vs_recurrence=relative_final_error_vs_recurrence,
        relative_local_error_vs_expm=relative_local_error_vs_expm,
        optimizer_success=bool(opt.success),
        optimizer_message=str(opt.message),
        optimizer_iterations=int(opt.nit),
        optimizer_wall_time_seconds=float(opt_seconds),
        hamiltonian_build_seconds=float(hamiltonian_build_seconds),
        pauli_terms_denominator=int(reduced.denominator_metadata["pauli_terms_denominator"]),
        pauli_terms_numerator=int(num_meta["pauli_terms_numerator"]),
        theta_opt=np.asarray(opt.x, dtype=float),
        y_end_padded=np.asarray(y_end_padded, dtype=complex),
        boundary_vec=np.asarray(boundary_vec, dtype=complex),
        progress_records=progress_records,
        optimizer_history=opt.history,
    )



def solve_window_with_retries(
    *,
    args: argparse.Namespace,
    bcow_module,
    restarted_module,
    vqls_module,
    qml,
    system,
    reduced: ReducedOperators,
    y_start_padded: np.ndarray,
    t_start: float,
    t_end: float,
    window_index: int,
    n_windows: int,
    rng: np.random.Generator,
    theta_previous: np.ndarray | None,
) -> ReducedWindowResult:
    best_result: ReducedWindowResult | None = None
    for retry_index in range(args.window_retries + 1):
        result = solve_window_once(
            args=args,
            bcow_module=bcow_module,
            restarted_module=restarted_module,
            vqls_module=vqls_module,
            qml=qml,
            system=system,
            reduced=reduced,
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
                f"target {args.residual_target:.6e}; retrying."
            )

    assert best_result is not None
    if best_result.relative_residual > args.residual_target:
        msg = (
            f"window {window_index} best residual {best_result.relative_residual:.6e} "
            f"exceeds --residual-target {args.residual_target:.6e}"
        )
        if args.stop_on_window_failure:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return best_result


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


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
        "boundary_clock_dim",
        "padded_boundary_clock_dim",
        "vqls_qubits",
        "ansatz_parameter_count",
        "initial_sqrt_cost",
        "final_sqrt_cost",
        "relative_residual",
        "final_block_probability",
        "relative_boundary_error_vs_recurrence",
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



def save_outputs(
    *,
    args: argparse.Namespace,
    system,
    metadata: dict[str, object],
    window_results: Sequence[ReducedWindowResult],
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
        payload["window_final_block_probabilities"] = np.asarray([w.final_block_probability for w in window_results], dtype=float)
        if args.save_window_solutions:
            for w in window_results:
                payload[f"boundary_solution_window_{w.window_index:04d}"] = np.asarray(w.boundary_vec, dtype=complex)

    if final_exact_original is not None:
        payload["x_exact"] = np.asarray(final_exact_original, dtype=complex)
        payload["u_exact"] = np.real_if_close(final_exact_original[:n_mech], tol=1e8)
        payload["v_exact"] = np.real_if_close(final_exact_original[n_mech:], tol=1e8)
        table = np.column_stack([system.X_all, payload["u_exact"], u_approx])
        displacement_csv_path = prefix.with_name(prefix.name + "_displacement.csv")
        np.savetxt(displacement_csv_path, table, delimiter=",", header="X_all,u_exact,u_boundary_reduced_vqls", comments="")

    summary_rows = [w.summary_dict() for w in window_results]
    global_by_window = {int(g["window_index"]): g for g in metadata.get("global_metrics", [])}
    for row in summary_rows:
        gm = global_by_window.get(int(row["window_index"]))
        if gm is not None:
            row.update({k: v for k, v in gm.items() if k not in {"window_index", "time"}})

    np.savez_compressed(npz_path, **payload)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    write_summary_csv(summary_csv_path, summary_rows)
    return npz_path, json_path, summary_csv_path, displacement_csv_path


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def solve_boundary_reduced_restarted_vqls(args: argparse.Namespace) -> dict[str, object]:
    bcow_module = load_module_from_path(args.bcow_module_path, "bcow_dynamic_module_reduced")
    vqls_module = load_module_from_path(args.vqls_module_path, "bcow_vqls_module_reduced")
    restarted_module = load_module_from_path(args.restarted_module_path, "bcow_restarted_module_reduced")
    qml = vqls_module.qml

    if args.energy_estimator != "pauli_hamiltonian":
        raise ValueError("This reduced driver currently supports --energy-estimator pauli_hamiltonian only.")
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

    times = build_time_grid(args.t, args.restart_window, args.restart_steps)
    n_windows = len(times) - 1
    if n_windows < 1:
        raise ValueError("The restart time grid must contain at least one window.")

    print("PennyLane version:", qml.__version__)
    print("Algorithm: boundary-reduced restarted BCOW + PennyLane VQLS")
    print("Mechanics case: unchanged 1D dynamic MD-FE coupling")
    print("Mechanical DOFs:", system.n_mech)
    print("Original state dimension:", system.state_dim_original)
    print("Padded ODE system dimension:", system.state_dim_padded)
    print("Total final time T:", f"{args.t:.8e}")
    print("Restart windows:", n_windows)
    print("Requested restart window:", f"{args.restart_window:.8e}")
    print("Taylor order k:", args.taylor_order)
    print("Target:", args.target)
    print("PennyLane device requested:", args.device)
    print("Optimizer:", args.optimizer)
    print("Ansatz:", args.ansatz)
    print("Ansatz layers:", args.layers)
    print("Shots: analytic mode; no finite shot count is set")

    operator_cache: dict[tuple[object, ...], ReducedOperators] = {}
    window_results: list[ReducedWindowResult] = []
    restart_states = [np.asarray(system.y0_padded, dtype=complex).copy()]
    theta_previous: np.ndarray | None = None
    global_metrics: list[dict[str, object]] = []

    for w in range(1, n_windows + 1):
        t_start = float(times[w - 1])
        t_end = float(times[w])
        dt = t_end - t_start
        y_start_padded = restart_states[-1]
        segments, norm_A = resolve_segments_for_window(args, bcow_module, system.A_sparse_padded, dt)
        reduced = get_reduced_operators(
            cache=operator_cache,
            args=args,
            bcow_module=bcow_module,
            restarted_module=restarted_module,
            vqls_module=vqls_module,
            system=system,
            dt=dt,
            segments=segments,
        )
        print(
            f"Window {w:04d}/{n_windows}: t=[{t_start:.8e}, {t_end:.8e}], "
            f"dt={dt:.8e}, m={segments}, ||A h||~{norm_A * abs(reduced.layout.h_step):.4e}, "
            f"boundary_clock={reduced.layout.padded_boundary_clock_dim}, "
            f"VQLS_qubits={reduced.n_qubits}, "
            f"den_terms={reduced.denominator_metadata['pauli_terms_denominator']}"
        )

        result = solve_window_with_retries(
            args=args,
            bcow_module=bcow_module,
            restarted_module=restarted_module,
            vqls_module=vqls_module,
            qml=qml,
            system=system,
            reduced=reduced,
            y_start_padded=y_start_padded,
            t_start=t_start,
            t_end=t_end,
            window_index=w,
            n_windows=n_windows,
            rng=rng,
            theta_previous=theta_previous,
        )
        window_results.append(result)
        restart_states.append(result.y_end_padded)
        theta_previous = result.theta_opt.copy()

        msg = (
            f"  done: residual={result.relative_residual:.6e}, "
            f"sqrt_cost={result.final_sqrt_cost:.6e}, "
            f"final_block_prob={result.final_block_probability:.6e}"
        )
        if result.relative_final_error_vs_recurrence is not None:
            msg += f", err_vs_reduced_recurrence={result.relative_final_error_vs_recurrence:.6e}"
        if result.relative_local_error_vs_expm is not None:
            msg += f", local_err_vs_expm={result.relative_local_error_vs_expm:.6e}"
        print(msg)

        if should_compute_exact_for_window(args, w, n_windows):
            exact_from_start = bcow_module.expm_times_vec(system.A_sparse_padded, t_end, system.y0_padded)
            global_rel = float(
                np.linalg.norm(result.y_end_padded - exact_from_start) / max(np.linalg.norm(exact_from_start), 1e-15)
            )
            dist, overlap = normalized_state_metrics(result.y_end_padded, exact_from_start)
            gm = GlobalMetric(
                window_index=w,
                time=t_end,
                global_relative_error_vs_expm=global_rel,
                global_state_distance_vs_expm=dist,
                global_overlap_abs_vs_expm=overlap,
            )
            global_metrics.append(asdict(gm))
            print(
                f"  cumulative exact check: global_rel_err={global_rel:.6e}, "
                f"state_distance={dist:.6e}, overlap={overlap:.8e}"
            )

    restart_states_array = np.asarray(restart_states, dtype=complex)

    final_exact_original: np.ndarray | None = None
    if not args.skip_exact:
        final_exact_padded = bcow_module.expm_times_vec(system.A_sparse_padded, args.t, system.y0_padded)
        final_exact_original = final_exact_padded[: system.state_dim_original]

    total_seconds = time.perf_counter() - total_start
    final_state = restart_states_array[-1]
    final_original = final_state[: system.state_dim_original]
    final_global_error: float | None = None
    final_state_distance: float | None = None
    final_overlap: float | None = None
    if final_exact_original is not None:
        final_exact_padded = bcow_module.expm_times_vec(system.A_sparse_padded, args.t, system.y0_padded)
        final_global_error = float(np.linalg.norm(final_state - final_exact_padded) / max(np.linalg.norm(final_exact_padded), 1e-15))
        final_state_distance, final_overlap = normalized_state_metrics(final_state, final_exact_padded)

    metadata: dict[str, object] = {
        "algorithm": "boundary-reduced restarted BCOW + PennyLane VQLS",
        "reduction": "internal BCOW Taylor variables eliminated; VQLS solves boundary-only linear system",
        "total_time": float(args.t),
        "restart_times": times.tolist(),
        "restart_windows": int(n_windows),
        "mechanics": {
            "n1": int(args.n1),
            "n2": int(args.n2),
            "n_mech": int(system.n_mech),
            "state_dim_original": int(system.state_dim_original),
            "state_dim_padded": int(system.state_dim_padded),
            "pulse_atoms": int(args.pulse_atoms),
        },
        "vqls": {
            "target": args.target,
            "device": args.device,
            "energy_estimator": args.energy_estimator,
            "optimizer": args.optimizer,
            "maxiter": int(args.maxiter),
            "ansatz": args.ansatz,
            "layers": int(args.layers),
            "initial_state_guess": args.initial_state_guess,
            "theta_warm_start": args.theta_warm_start,
            "theta_jitter": float(args.theta_jitter),
            "coefficient_tol": float(args.coefficient_tol),
            "max_vqls_qubits": int(args.max_vqls_qubits),
        },
        "boundary_reduced_layouts_seen": [
            {
                "key": list(map(str, key)),
                "segments": op.layout.segments,
                "taylor_order": op.layout.taylor_order,
                "h_step": op.layout.h_step,
                "boundary_clock_dim": op.layout.boundary_clock_dim,
                "padded_boundary_clock_dim": op.layout.padded_boundary_clock_dim,
                "system_dim": op.layout.system_dim,
                "target_dim": op.target_dim,
                "vqls_qubits": op.n_qubits,
                "P_nnz": int(op.P.nnz),
                "C_nnz": int(op.C.nnz),
                "pauli_terms_denominator": int(op.denominator_metadata["pauli_terms_denominator"]),
            }
            for key, op in operator_cache.items()
        ],
        "window_summaries": [w.summary_dict() for w in window_results],
        "global_metrics": global_metrics,
        "final_global_relative_error_vs_expm": final_global_error,
        "final_global_state_distance_vs_expm": final_state_distance,
        "final_global_overlap_abs_vs_expm": final_overlap,
        "total_wall_time_seconds": float(total_seconds),
    }

    npz_path, json_path, summary_csv_path, displacement_csv_path = save_outputs(
        args=args,
        system=system,
        metadata=metadata,
        window_results=window_results,
        restart_states=restart_states_array,
        final_exact_original=final_exact_original,
    )

    print("\nBoundary-reduced restarted VQLS complete.")
    print("Final state norm:", f"{np.linalg.norm(final_original):.8e}")
    if final_global_error is not None:
        print("Final global relative error vs expm_multiply:", f"{final_global_error:.8e}")
        print("Final normalized state distance vs expm_multiply:", f"{final_state_distance:.8e}")
        print("Final overlap |<exact|approx>|:", f"{final_overlap:.8e}")
    print("Saved NPZ:", npz_path)
    print("Saved JSON:", json_path)
    print("Saved window summary CSV:", summary_csv_path)
    if displacement_csv_path is not None:
        print("Saved displacement CSV:", displacement_csv_path)

    return metadata


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Boundary-reduced restarted BCOW + shot-free PennyLane VQLS for the 1D dynamic MD-FE ODE system."
    )

    # Project file paths.
    parser.add_argument("--bcow-module-path", default="1Ddynamic_BCOW_ODE_solver.py")
    parser.add_argument("--vqls-module-path", default="1Ddynamic_BCOW_VQLS_pennylane.py")
    parser.add_argument("--restarted-module-path", default="1Ddynamic_BCOW_VQLS_restarted_pennylane.py")

    # Mechanics model; aliases mimic the user's static VQLS command style.
    parser.add_argument("--n1", "--md-atoms", dest="n1", type=int, default=5)
    parser.add_argument("--n2", "--fe-elems", dest="n2", type=int, default=2)
    parser.add_argument("--u-right", type=float, default=None, help="Accepted for CLI compatibility only; not used.")
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--h-factor", type=float, default=1.0)
    parser.add_argument("--k-spring", type=float, default=1.0)
    parser.add_argument("--mass-atom", type=float, default=1.0)
    parser.add_argument("--pulse-atoms", type=int, default=3)
    parser.add_argument("--pulse-amp-factor", type=float, default=0.01)
    parser.add_argument("--pulse-wavelength-factor", type=float, default=8.0)
    parser.add_argument("--no-pad", action="store_true", help="Do not pad the ODE system dimension to a power of two.")

    # Time and reduced BCOW layout.
    parser.add_argument("--t", type=float, default=0.5, help="Total final time T.")
    parser.add_argument("--restart-window", type=float, default=0.1, help="Approximate window size for restarting.")
    parser.add_argument("--restart-steps", type=int, default=0, help="If >0, use exactly this many equal restart windows.")
    parser.add_argument("--segments", type=int, default=1, help="BCOW segments m per restart window. Use 0 for norm-based automatic choice.")
    parser.add_argument("--segment-norm-bound", type=float, default=0.5)
    parser.add_argument("--norm-exact-dim-limit", type=int, default=512)
    parser.add_argument("--taylor-order", type=int, default=4)
    parser.add_argument("--no-pad-clock", action="store_true", help="Do not pad the reduced boundary clock to a power of two.")
    parser.add_argument(
        "--propagator-drop-tol",
        type=float,
        default=0.0,
        help="Drop tiny entries while building P_k(hA). Keep 0.0 for exact small tests.",
    )

    # VQLS target and estimator.
    parser.add_argument("--target", choices=["direct-c", "hermitian-dilation"], default="direct-c")
    parser.add_argument("--device", default="lightning.gpu")
    parser.add_argument("--no-device-fallback", action="store_true")
    parser.add_argument("--energy-estimator", choices=["pauli_hamiltonian"], default="pauli_hamiltonian")
    parser.add_argument("--coefficient-tol", type=float, default=1e-10)
    parser.add_argument("--max-vqls-qubits", type=int, default=12)

    # Ansatz and optimizer.
    parser.add_argument(
        "--ansatz",
        choices=["hardware_efficient", "real_ry", "real_ry_identity", "hardware_efficient_identity"],
        default="real_ry_identity",
    )
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--initial-state-guess", default="rhs")
    parser.add_argument("--init-scale", type=float, default=0.0)
    parser.add_argument("--theta-warm-start", choices=["none", "previous"], default="previous")
    parser.add_argument("--theta-jitter", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["spsa", "scipy-powell", "scipy-lbfgsb"], default="scipy-lbfgsb")
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--spsa-a", type=float, default=0.05)
    parser.add_argument("--spsa-c", type=float, default=0.08)
    parser.add_argument("--rng-seed", type=int, default=1234)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--verbose-windows", action="store_true")

    # Restart/error controls.
    parser.add_argument("--residual-target", type=float, default=1e-2)
    parser.add_argument("--window-retries", type=int, default=0)
    parser.add_argument("--stop-on-window-failure", action="store_true")
    parser.add_argument("--carry-mode", choices=["vqls", "recurrence", "expm"], default="vqls")
    parser.add_argument("--compare-direct", action="store_true", help="Compare each VQLS window with reduced classical recurrence.")
    parser.add_argument("--skip-exact", action="store_true", help="Skip expm_multiply exact checks.")
    parser.add_argument("--exact-every", type=int, default=10)

    # Output.
    parser.add_argument("--output-prefix", default="boundary_reduced_bcow_vqls")
    parser.add_argument("--save-window-solutions", action="store_true")

    return parser.parse_args()



def main() -> None:
    args = parse_args()
    solve_boundary_reduced_restarted_vqls(args)


if __name__ == "__main__":
    main()
