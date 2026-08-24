#!/usr/bin/env python3
"""
BCOW-style quantum linear-ODE rewrite for Case II: 1D dynamic MD-FE coupling.

This script keeps the mechanics scenario from the Level-2.5 LCU code unchanged:

    y_dot = A y,    y = [u; v],    A = [[0, I], [-M^{-1}K, 0]].

The algorithmic layer is changed from measured LCU Taylor time marching to the
Berry-Childs-Ostrander-Wang (BCOW) construction.  BCOW encodes the whole Taylor
segmented evolution into one sparse linear system

    C_{m,k,p}(A h) |history> = |rhs>,      h = t_final / m,

where m is the number of time segments, k is the Taylor order per segment, and p
adds repeated final-time blocks so that measuring the clock register has a
non-negligible probability of returning the final state.

What this script does in a classical Python environment:
  * builds the same mechanical A and initial condition y0;
  * builds the BCOW clock-register layout and, optionally, the sparse matrix C;
  * solves the BCOW linear system either by a sparse direct solve (small tests)
    or by a matrix-free recurrence that is algebraically equivalent to the
    lower-triangular BCOW system (large tests / reference emulator);
  * extracts the final-time block(s), compares with expm_multiply, and saves the
    mechanics displacement output.

How this maps to a quantum implementation:
  * replace the classical sparse-direct/recurrence backend by a quantum linear
    systems algorithm (QLSA/QSVT linear-system solver) applied to C, or to the
    Hermitian dilation [[0, C], [C^†, 0]] generated with --save-hermitian-dilation;
  * measure the BCOW clock register in the final-time subspace and use amplitude
    amplification/estimation if a normalized final state or observables are the
    desired outputs;
  * do not reconstruct all components unless the application truly requires a
    full classical displacement field.

Example quick validation:
    python 1Ddynamic_BCOW_ODE_solver.py --n1 6 --n2 1 --t 1.0 \
        --segments 4 --taylor-order 6 --padding-steps 4 \
        --backend sparse-direct --output-prefix bcow_small

Notebook-size mechanics case with the BCOW recurrence emulator:
    python 1Ddynamic_BCOW_ODE_solver.py --n1 106 --n2 21 --t 106 \
        --segments 0 --taylor-order 8 --padding-steps -1 \
        --backend recurrence --output-prefix bcow_caseII_full

Export a QLSA-ready Hermitian dilation for a small case:
    python 1Ddynamic_BCOW_ODE_solver.py --n1 6 --n2 1 --t 1.0 \
        --segments 4 --taylor-order 6 --backend none \
        --save-linear-system --save-hermitian-dilation \
        --output-prefix bcow_small_export
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import scipy.linalg as dense_la
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires SciPy. Install with: pip install scipy"
    ) from exc


# -----------------------------------------------------------------------------
# Mechanics: same 1D dynamic MD-FE coupling model as the uploaded Case II code
# -----------------------------------------------------------------------------


@dataclass
class DynamicSystem:
    n1: int
    n2: int
    a: float
    h: float
    k_spring: float
    mass_atom: float
    n_mech: int
    state_dim_original: int
    state_dim_padded: int
    A_original: np.ndarray
    A_padded: np.ndarray
    A_sparse_original: sp.csr_matrix
    A_sparse_padded: sp.csr_matrix
    y0_original: np.ndarray
    y0_padded: np.ndarray
    K: np.ndarray
    M_diag: np.ndarray
    X_atoms: np.ndarray
    X_fe: np.ndarray
    X_all: np.ndarray


def next_power_of_two(value: int) -> int:
    if value < 1:
        return 1
    return 1 << int(math.ceil(math.log2(value)))


def build_dynamic_md_fe_system(
    n1: int = 106,
    n2: int = 21,
    a: float = 1.0,
    h_factor: float = 5.0,
    k_spring: float = 1.0,
    mass_atom: float = 1.0,
    pulse_atoms: int = 50,
    pulse_amp_factor: float = 0.02,
    pulse_wavelength_factor: float = 200.0,
    pad_to_power2: bool = True,
) -> DynamicSystem:
    """Build exactly the same mechanics model used in the LCU Case II script."""
    if n1 < 2:
        raise ValueError("n1 must be >= 2.")
    if n2 < 1:
        raise ValueError("n2 must be >= 1.")

    h = h_factor * a
    area = 1.0
    c0 = a * math.sqrt(k_spring / mass_atom)
    rho = mass_atom / (a * area)
    young = rho * c0**2
    k_fe = young * area / h

    X_atoms = np.arange(n1, dtype=float) * a
    X_fe = n1 * a + np.arange(n2 + 1, dtype=float) * h
    X_all = np.concatenate([X_atoms, X_fe])

    u_atoms = np.zeros(n1, dtype=float)
    v_atoms = np.zeros(n1, dtype=float)
    u_fe = np.zeros(n2 + 1, dtype=float)
    v_fe = np.zeros(n2 + 1, dtype=float)

    M_fe = np.zeros(n2 + 1, dtype=float)
    M_e = rho * area * h
    for j in range(n2):
        M_fe[j] += 0.5 * M_e
        M_fe[j + 1] += 0.5 * M_e
    M_fe[0] = 0.5 * M_e
    M_atoms = mass_atom * np.ones(n1, dtype=float)

    pulse_atoms = int(min(max(1, pulse_atoms), n1))
    amp = pulse_amp_factor * a
    wavelength = pulse_wavelength_factor * a
    k_lo = 2.0 * math.pi / wavelength
    if pulse_atoms == 1:
        u_atoms[0] += amp * math.cos(k_lo * X_atoms[0])
    else:
        for idx in range(pulse_atoms):
            smooth = 0.5 * (1.0 + math.cos(math.pi * idx / (pulse_atoms - 1)))
            u_atoms[idx] += smooth * amp * math.cos(k_lo * X_atoms[idx])

    def x_atoms(u: np.ndarray) -> np.ndarray:
        return X_atoms + u

    def x_fe(u: np.ndarray) -> np.ndarray:
        return X_fe + u

    def internal_forces(ua: np.ndarray, uf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fA = np.zeros_like(ua)
        fF = np.zeros_like(uf)
        xa = x_atoms(ua)
        xf = x_fe(uf)

        for i in range(n1 - 1):
            ext = xa[i + 1] - xa[i] - a
            force = k_spring * ext
            fA[i] += force
            fA[i + 1] -= force

        for j in range(n2):
            ext = uf[j + 1] - uf[j]
            force = k_fe * ext
            fF[j] += force
            fF[j + 1] -= force

        ia = n1 - 1
        ext_md = xf[0] - xa[ia] - a
        f_md = k_spring * ext_md
        ext_fe = uf[1] - uf[0]
        f_fe = k_fe * ext_fe
        f_hand = 0.5 * f_md + 0.5 * f_fe
        fA[ia] += f_hand
        fF[0] -= f_hand
        return fA, fF

    def global_force(u_full: np.ndarray) -> np.ndarray:
        ua = u_full[:n1]
        uf = u_full[n1:]
        fA, fF = internal_forces(ua, uf)
        return np.concatenate([fA, fF])

    n_mech = n1 + (n2 + 1)
    K = np.zeros((n_mech, n_mech), dtype=float)
    probe = np.zeros(n_mech, dtype=float)
    for j in range(n_mech):
        probe[:] = 0.0
        probe[j] = 1.0
        K[:, j] = -global_force(probe)

    M_diag = np.concatenate([M_atoms, M_fe])
    MinvK = (K.T / M_diag).T
    Z = np.zeros((n_mech, n_mech), dtype=float)
    I = np.eye(n_mech, dtype=float)
    A_original = np.block([[Z, I], [-MinvK, Z]])

    u0 = np.concatenate([u_atoms, u_fe])
    v0 = np.concatenate([v_atoms, v_fe])
    y0_original = np.concatenate([u0, v0])

    original_dim = A_original.shape[0]
    padded_dim = next_power_of_two(original_dim) if pad_to_power2 else original_dim
    if padded_dim != original_dim:
        A_padded = np.zeros((padded_dim, padded_dim), dtype=complex)
        A_padded[:original_dim, :original_dim] = A_original.astype(complex)
        y0_padded = np.zeros(padded_dim, dtype=complex)
        y0_padded[:original_dim] = y0_original.astype(complex)
    else:
        if original_dim & (original_dim - 1):
            raise ValueError(
                f"State dimension {original_dim} is not a power of two. "
                "Use pad_to_power2=True."
            )
        A_padded = A_original.astype(complex)
        y0_padded = y0_original.astype(complex)

    return DynamicSystem(
        n1=n1,
        n2=n2,
        a=a,
        h=h,
        k_spring=k_spring,
        mass_atom=mass_atom,
        n_mech=n_mech,
        state_dim_original=original_dim,
        state_dim_padded=padded_dim,
        A_original=A_original,
        A_padded=A_padded,
        A_sparse_original=sp.csr_matrix(A_original.astype(complex)),
        A_sparse_padded=sp.csr_matrix(A_padded),
        y0_original=y0_original.astype(complex),
        y0_padded=y0_padded.astype(complex),
        K=K,
        M_diag=M_diag,
        X_atoms=X_atoms,
        X_fe=X_fe,
        X_all=X_all,
    )


# -----------------------------------------------------------------------------
# BCOW clock-register linear system
# -----------------------------------------------------------------------------


@dataclass
class BCOWLayout:
    segments: int
    taylor_order: int
    padding_steps: int
    h_step: float
    active_clock_dim: int
    padded_clock_dim: int
    system_dim: int
    clock_qubits: int
    system_qubits: int
    final_start_block: int
    total_vector_dim: int
    total_qubits_without_dilation: int
    total_qubits_with_hermitian_dilation: int


@dataclass
class BCOWDiagnostics:
    backend: str
    wall_time_seconds: float
    history_norm: float
    final_norm: float
    final_subspace_probability: float
    final_copy_relative_mismatch: float
    residual_relative_norm: float | None
    vector_abs_error_vs_expm: float | None
    vector_rel_error_vs_expm: float | None
    normalized_state_distance_vs_expm: float | None
    overlap_abs_vs_expm: float | None


def build_bcow_layout(
    *,
    system_dim: int,
    t_final: float,
    segments: int,
    taylor_order: int,
    padding_steps: int,
    pad_clock_to_power2: bool = True,
) -> BCOWLayout:
    if segments < 1:
        raise ValueError("segments must be >= 1.")
    if taylor_order < 1:
        raise ValueError("taylor_order must be >= 1.")
    if padding_steps < 0:
        raise ValueError("padding_steps must be >= 0 after default resolution.")
    if system_dim < 1:
        raise ValueError("system_dim must be positive.")

    h_step = float(t_final) / int(segments)
    active_clock_dim = int(segments) * (int(taylor_order) + 1) + int(padding_steps) + 1
    padded_clock_dim = next_power_of_two(active_clock_dim) if pad_clock_to_power2 else active_clock_dim
    if not pad_clock_to_power2 and active_clock_dim & (active_clock_dim - 1):
        warnings.warn(
            "Clock dimension is not a power of two. This is fine for classical sparse "
            "emulation, but a quantum clock register would normally be padded.",
            RuntimeWarning,
            stacklevel=2,
        )
    clock_qubits = int(math.ceil(math.log2(padded_clock_dim))) if padded_clock_dim > 1 else 0
    system_qubits = int(math.ceil(math.log2(system_dim))) if system_dim > 1 else 0
    final_start_block = int(segments) * (int(taylor_order) + 1)
    total_vector_dim = padded_clock_dim * system_dim
    return BCOWLayout(
        segments=int(segments),
        taylor_order=int(taylor_order),
        padding_steps=int(padding_steps),
        h_step=h_step,
        active_clock_dim=active_clock_dim,
        padded_clock_dim=padded_clock_dim,
        system_dim=system_dim,
        clock_qubits=clock_qubits,
        system_qubits=system_qubits,
        final_start_block=final_start_block,
        total_vector_dim=total_vector_dim,
        total_qubits_without_dilation=clock_qubits + system_qubits,
        total_qubits_with_hermitian_dilation=1 + clock_qubits + system_qubits,
    )


def estimate_operator_norm(A: sp.spmatrix | np.ndarray, exact_dim_limit: int = 512) -> float:
    """Estimate ||A||_2 for selecting h with ||A h|| <= O(1)."""
    n = A.shape[0]
    if exact_dim_limit > 0 and n <= exact_dim_limit:
        dense = A.toarray() if sp.issparse(A) else np.asarray(A)
        return float(np.linalg.norm(dense, ord=2))

    # Cheap rigorous-ish upper bound: ||A||_2 <= sqrt(||A||_1 ||A||_inf).
    if sp.issparse(A):
        norm_1 = float(abs(A).sum(axis=0).max())
        norm_inf = float(abs(A).sum(axis=1).max())
    else:
        norm_1 = float(np.linalg.norm(A, ord=1))
        norm_inf = float(np.linalg.norm(A, ord=np.inf))
    return math.sqrt(max(norm_1 * norm_inf, 0.0))


def choose_segments_from_norm(
    A: sp.spmatrix | np.ndarray,
    t_final: float,
    segment_norm_bound: float = 1.0,
    exact_norm_dim_limit: int = 512,
) -> tuple[int, float]:
    if segment_norm_bound <= 0:
        raise ValueError("segment_norm_bound must be positive.")
    norm_A = estimate_operator_norm(A, exact_dim_limit=exact_norm_dim_limit)
    segments = max(1, int(math.ceil(abs(float(t_final)) * norm_A / segment_norm_bound)))
    return segments, norm_A


def build_bcow_rhs(
    y0: np.ndarray,
    layout: BCOWLayout,
    b: np.ndarray | None = None,
) -> np.ndarray:
    """Right-hand side for C_{m,k,p}(Ah)|history> = rhs.

    The uploaded mechanics case is homogeneous, so b defaults to zero.  The
    inhomogeneous term is included to keep the implementation faithful to BCOW.
    """
    y0 = np.asarray(y0, dtype=complex)
    if y0.shape != (layout.system_dim,):
        raise ValueError("y0 has the wrong shape for the BCOW layout.")
    if b is None:
        b_vec = np.zeros(layout.system_dim, dtype=complex)
    else:
        b_vec = np.asarray(b, dtype=complex)
        if b_vec.shape != (layout.system_dim,):
            raise ValueError("b has the wrong shape for the BCOW layout.")

    rhs = np.zeros(layout.total_vector_dim, dtype=complex)
    rhs[0 : layout.system_dim] = y0
    if np.linalg.norm(b_vec) > 0:
        for i in range(layout.segments):
            block = i * (layout.taylor_order + 1) + 1
            lo = block * layout.system_dim
            rhs[lo : lo + layout.system_dim] += layout.h_step * b_vec
    return rhs


def build_bcow_sparse_matrix(A: sp.spmatrix, layout: BCOWLayout) -> sp.csr_matrix:
    r"""Build C_{m,k,p}(Ah) using clock Kronecker products.

    C = I_clock \\otimes I_system
        - sum_{i=0}^{m-1} sum_{j=1}^k |i(k+1)+j><i(k+1)+j-1| \\otimes (h A / j)
        - sum_{i=0}^{m-1} sum_{j=0}^k |(i+1)(k+1)><i(k+1)+j| \\otimes I
        - sum_{j=1}^p |m(k+1)+j><m(k+1)+j-1| \\otimes I.
    """
    A = sp.csr_matrix(A, dtype=complex)
    if A.shape != (layout.system_dim, layout.system_dim):
        raise ValueError("A shape does not match layout.system_dim.")

    cdim = layout.padded_clock_dim
    n = layout.system_dim
    eye_clock = sp.eye(cdim, dtype=complex, format="csr")
    eye_sys = sp.eye(n, dtype=complex, format="csr")

    rows_A: list[int] = []
    cols_A: list[int] = []
    data_A: list[complex] = []
    rows_I: list[int] = []
    cols_I: list[int] = []
    data_I: list[complex] = []

    k = layout.taylor_order
    m = layout.segments
    h = layout.h_step

    # Taylor recurrence rows: x_{i,j} - (hA/j) x_{i,j-1} = h b delta_{j,1}.
    for i in range(m):
        base = i * (k + 1)
        for j in range(1, k + 1):
            rows_A.append(base + j)
            cols_A.append(base + j - 1)
            data_A.append(-h / float(j))

    # Segment-transition rows: x_{i+1,0} - sum_j x_{i,j} = 0.
    for i in range(m):
        row = (i + 1) * (k + 1)
        base = i * (k + 1)
        for j in range(k + 1):
            rows_I.append(row)
            cols_I.append(base + j)
            data_I.append(-1.0)

    # Final padding rows: x_{m,j} - x_{m,j-1} = 0.
    final_start = layout.final_start_block
    for j in range(1, layout.padding_steps + 1):
        rows_I.append(final_start + j)
        cols_I.append(final_start + j - 1)
        data_I.append(-1.0)

    clock_A = sp.coo_matrix((data_A, (rows_A, cols_A)), shape=(cdim, cdim), dtype=complex).tocsr()
    clock_I = sp.coo_matrix((data_I, (rows_I, cols_I)), shape=(cdim, cdim), dtype=complex).tocsr()

    C = sp.kron(eye_clock, eye_sys, format="csr")
    C = C + sp.kron(clock_A, A, format="csr")
    C = C + sp.kron(clock_I, eye_sys, format="csr")
    return C.tocsr()


def build_hermitian_dilation(C: sp.spmatrix) -> sp.csr_matrix:
    """Return [[0, C], [C^†, 0]], the standard QLSA Hermitian embedding."""
    C = sp.csr_matrix(C, dtype=complex)
    zero = sp.csr_matrix(C.shape, dtype=complex)
    return sp.bmat([[zero, C], [C.conjugate().transpose(), zero]], format="csr")


def build_hermitian_rhs(rhs: np.ndarray) -> np.ndarray:
    """Right-hand side for the Hermitian dilation: [rhs; 0]."""
    rhs = np.asarray(rhs, dtype=complex)
    return np.concatenate([rhs, np.zeros_like(rhs)])


def solve_bcow_by_recurrence(
    A: sp.spmatrix,
    y0: np.ndarray,
    layout: BCOWLayout,
    b: np.ndarray | None = None,
) -> np.ndarray:
    """Matrix-free solve of the lower-triangular BCOW system.

    This is a classical emulator of the BCOW linear-system solution.  It should
    not be confused with the quantum route; the quantum route replaces this
    function with a QLSA applied to C or its Hermitian dilation.
    """
    A = sp.csr_matrix(A, dtype=complex)
    y0 = np.asarray(y0, dtype=complex)
    if y0.shape != (layout.system_dim,):
        raise ValueError("y0 has the wrong shape for the BCOW layout.")
    if b is None:
        b_vec = np.zeros(layout.system_dim, dtype=complex)
    else:
        b_vec = np.asarray(b, dtype=complex)
        if b_vec.shape != (layout.system_dim,):
            raise ValueError("b has the wrong shape for the BCOW layout.")

    history = np.zeros((layout.padded_clock_dim, layout.system_dim), dtype=complex)
    history[0] = y0

    k = layout.taylor_order
    h = layout.h_step
    for i in range(layout.segments):
        base = i * (k + 1)
        for j in range(1, k + 1):
            history[base + j] = (h / float(j)) * (A @ history[base + j - 1])
            if j == 1 and np.linalg.norm(b_vec) > 0:
                history[base + j] += h * b_vec
        next_base = (i + 1) * (k + 1)
        history[next_base] = np.sum(history[base : base + k + 1], axis=0)

    final_start = layout.final_start_block
    for j in range(1, layout.padding_steps + 1):
        history[final_start + j] = history[final_start + j - 1]

    return history


def solve_bcow_sparse_direct(C: sp.spmatrix, rhs: np.ndarray, layout: BCOWLayout) -> np.ndarray:
    """Classical sparse direct solve C x = rhs, useful for small validation."""
    sol = spla.spsolve(C.tocsc(), np.asarray(rhs, dtype=complex))
    return np.asarray(sol, dtype=complex).reshape((layout.padded_clock_dim, layout.system_dim))


def bcow_residual_relative_norm(
    C: sp.spmatrix,
    rhs: np.ndarray,
    history: np.ndarray,
) -> float:
    vec = np.asarray(history, dtype=complex).reshape(-1)
    residual = C @ vec - np.asarray(rhs, dtype=complex)
    return float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-15))


def extract_final_vector(history: np.ndarray, layout: BCOWLayout, mode: str = "average") -> tuple[np.ndarray, float, float, float]:
    """Extract y(t) and BCOW success diagnostics from a solved history state."""
    final_blocks = history[
        layout.final_start_block : layout.final_start_block + layout.padding_steps + 1
    ]
    first = final_blocks[0]
    if mode == "first":
        final = first.copy()
    elif mode == "last":
        final = final_blocks[-1].copy()
    elif mode == "average":
        final = np.mean(final_blocks, axis=0)
    else:
        raise ValueError("mode must be one of: first, last, average")

    history_norm = float(np.linalg.norm(history.reshape(-1)))
    final_norm = float(np.linalg.norm(final))
    final_subspace_norm_sq = float(np.sum(np.linalg.norm(final_blocks, axis=1) ** 2))
    final_subspace_probability = final_subspace_norm_sq / max(history_norm**2, 1e-30)

    if final_norm <= 1e-15:
        final_copy_relative_mismatch = 0.0
    else:
        final_copy_relative_mismatch = float(
            max(np.linalg.norm(block - first) for block in final_blocks) / max(final_norm, 1e-15)
        )
    return final, history_norm, final_subspace_probability, final_copy_relative_mismatch


# -----------------------------------------------------------------------------
# Reference, metrics, and output
# -----------------------------------------------------------------------------


def expm_times_vec(A: sp.spmatrix | np.ndarray, t: float, y0: np.ndarray) -> np.ndarray:
    """Reference exp(A t)y0 using scipy.sparse.linalg.expm_multiply."""
    A_sparse = sp.csr_matrix(A, dtype=complex) if not sp.issparse(A) else A.tocsr()
    return np.asarray(spla.expm_multiply(A_sparse * float(t), np.asarray(y0, dtype=complex)))


def normalized_state_metrics(approx: np.ndarray, exact: np.ndarray) -> tuple[float, float]:
    """Return phase-insensitive normalized state distance and |overlap|."""
    approx = np.asarray(approx, dtype=complex)
    exact = np.asarray(exact, dtype=complex)
    na = np.linalg.norm(approx)
    ne = np.linalg.norm(exact)
    if na <= 1e-15 or ne <= 1e-15:
        return float("nan"), float("nan")
    a = approx / na
    e = exact / ne
    overlap = complex(np.vdot(e, a))
    overlap_abs = float(min(1.0, abs(overlap)))
    distance = math.sqrt(max(0.0, 2.0 - 2.0 * overlap_abs))
    return distance, overlap_abs


def dense_condition_number_if_small(C: sp.spmatrix, max_dim: int) -> float | None:
    if C.shape[0] > max_dim:
        return None
    dense = C.toarray()
    return float(np.linalg.cond(dense))


def save_results(
    output_prefix: str,
    system: DynamicSystem,
    layout: BCOWLayout,
    final_approx_padded: np.ndarray | None,
    x_exact_original: np.ndarray | None,
    metadata: dict[str, object],
    history: np.ndarray | None = None,
) -> tuple[Path, Path | None, Path]:
    prefix = Path(output_prefix)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")

    payload: dict[str, object] = {
        "metadata": json.dumps(metadata, indent=2),
        "layout_json": json.dumps(asdict(layout), indent=2),
        "A_original": np.asarray(system.A_original),
        "y0_original": np.asarray(system.y0_original),
        "X_all": np.asarray(system.X_all),
    }

    csv_path: Path | None = None
    if final_approx_padded is not None:
        final_approx_original = final_approx_padded[: system.state_dim_original]
        payload["x_approx"] = np.asarray(final_approx_original)
        payload["x_approx_padded"] = np.asarray(final_approx_padded)
        n_mech = system.n_mech
        payload["u_approx"] = np.real_if_close(final_approx_original[:n_mech], tol=1e8)
        payload["v_approx"] = np.real_if_close(final_approx_original[n_mech:], tol=1e8)

    if x_exact_original is not None:
        payload["x_exact"] = np.asarray(x_exact_original)
        n_mech = system.n_mech
        payload["u_exact"] = np.real_if_close(x_exact_original[:n_mech], tol=1e8)
        payload["v_exact"] = np.real_if_close(x_exact_original[n_mech:], tol=1e8)

    if history is not None:
        payload["bcow_history"] = np.asarray(history)

    np.savez_compressed(npz_path, **payload)

    if final_approx_padded is not None and x_exact_original is not None:
        n_mech = system.n_mech
        final_approx_original = final_approx_padded[: system.state_dim_original]
        u_approx = np.real_if_close(final_approx_original[:n_mech], tol=1e8)
        u_exact = np.real_if_close(x_exact_original[:n_mech], tol=1e8)
        table = np.column_stack([system.X_all, u_exact, u_approx])
        csv_path = prefix.with_name(prefix.name + "_displacement.csv")
        np.savetxt(
            csv_path,
            table,
            delimiter=",",
            header="X_all,u_exact,u_bcow",
            comments="",
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return npz_path, csv_path, json_path


def save_sparse_exports(
    output_prefix: str,
    C: sp.spmatrix | None,
    rhs: np.ndarray | None,
    save_hermitian_dilation: bool,
) -> dict[str, str]:
    prefix = Path(output_prefix)
    paths: dict[str, str] = {}
    if C is not None:
        c_path = prefix.with_name(prefix.name + "_bcow_C.npz")
        sp.save_npz(c_path, C.tocsr())
        paths["bcow_C_sparse_npz"] = str(c_path)
    if rhs is not None:
        rhs_path = prefix.with_name(prefix.name + "_bcow_rhs.npy")
        np.save(rhs_path, np.asarray(rhs, dtype=complex))
        paths["bcow_rhs_npy"] = str(rhs_path)
    if save_hermitian_dilation:
        if C is None:
            raise ValueError("C must be built before saving the Hermitian dilation.")
        H = build_hermitian_dilation(C)
        h_path = prefix.with_name(prefix.name + "_bcow_hermitian_dilation.npz")
        sp.save_npz(h_path, H)
        paths["bcow_hermitian_dilation_sparse_npz"] = str(h_path)
        if rhs is not None:
            hrhs_path = prefix.with_name(prefix.name + "_bcow_hermitian_rhs.npy")
            np.save(hrhs_path, build_hermitian_rhs(rhs))
            paths["bcow_hermitian_rhs_npy"] = str(hrhs_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BCOW quantum linear-ODE construction for Case II 1D dynamic MD-FE coupling."
    )
    parser.add_argument("--n1", type=int, default=106, help="Number of MD atoms.")
    parser.add_argument("--n2", type=int, default=21, help="Number of FE elements.")
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--h-factor", type=float, default=5.0)
    parser.add_argument("--k-spring", type=float, default=1.0)
    parser.add_argument("--mass-atom", type=float, default=1.0)
    parser.add_argument("--pulse-atoms", type=int, default=50)
    parser.add_argument("--pulse-amp-factor", type=float, default=0.02)
    parser.add_argument("--pulse-wavelength-factor", type=float, default=200.0)
    parser.add_argument(
        "--no-pad",
        action="store_true",
        help="Require the original system dimension to be a power of two instead of padding.",
    )

    parser.add_argument("--t", type=float, default=106.0, help="Final time T.")
    parser.add_argument(
        "--segments",
        type=int,
        default=0,
        help=(
            "BCOW segment count m. Use 0 for automatic selection from "
            "||A||_2 h <= --segment-norm-bound."
        ),
    )
    parser.add_argument(
        "--segment-norm-bound",
        type=float,
        default=1.0,
        help="Automatic segment rule: choose m so ||A||_2 * T / m <= this value.",
    )
    parser.add_argument(
        "--norm-exact-dim-limit",
        type=int,
        default=0,
        help="Use dense exact ||A||_2 only when system dimension is at most this value. Default 0 uses the fast sqrt(||A||_1||A||_inf) upper bound.",
    )
    parser.add_argument(
        "--taylor-order",
        type=int,
        default=8,
        help="BCOW Taylor order k inside each segment. The paper's analysis uses k >= 5.",
    )
    parser.add_argument(
        "--padding-steps",
        type=int,
        default=-1,
        help="BCOW final-time copy count p. Use -1 to set p=m.",
    )
    parser.add_argument(
        "--no-pad-clock",
        action="store_true",
        help="Do not pad the BCOW clock register dimension to a power of two.",
    )

    parser.add_argument(
        "--backend",
        choices=["recurrence", "sparse-direct", "none"],
        default="recurrence",
        help=(
            "Classical emulator backend. 'recurrence' solves the lower-triangular BCOW "
            "system matrix-free; 'sparse-direct' builds C and calls scipy.spsolve for "
            "small validation; 'none' only builds/exports the quantum linear-system instance."
        ),
    )
    parser.add_argument(
        "--extract-final",
        choices=["first", "last", "average"],
        default="average",
        help="How to extract the repeated final BCOW blocks from the solved history state.",
    )
    parser.add_argument("--skip-exact", action="store_true", help="Skip expm_multiply reference.")
    parser.add_argument(
        "--residual-check",
        action="store_true",
        help="Build C and compute ||C history - rhs||/||rhs|| even for recurrence backend.",
    )
    parser.add_argument(
        "--condition-number-max-dim",
        type=int,
        default=0,
        help="If >0, compute dense condition number of C only when dim(C) is at most this value.",
    )
    parser.add_argument(
        "--save-linear-system",
        action="store_true",
        help="Save sparse BCOW C and rhs for an external QLSA/block-encoding workflow.",
    )
    parser.add_argument(
        "--save-hermitian-dilation",
        action="store_true",
        help="Also save [[0,C],[C^†,0]] and [rhs,0], the standard Hermitian QLSA embedding.",
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save the full BCOW history array in the output NPZ. Can be large.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="bcow_caseII")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress = not args.quiet

    if args.taylor_order < 5:
        warnings.warn(
            "BCOW's condition/error analysis is usually stated for k >= 5. "
            "Small k is useful for debugging but not recommended for final runs.",
            RuntimeWarning,
            stacklevel=2,
        )

    t_build_start = time.perf_counter()
    system = build_dynamic_md_fe_system(
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
    mechanics_build_seconds = time.perf_counter() - t_build_start

    if args.segments == 0:
        segments, norm_A_for_segments = choose_segments_from_norm(
            system.A_sparse_padded,
            args.t,
            segment_norm_bound=args.segment_norm_bound,
            exact_norm_dim_limit=args.norm_exact_dim_limit,
        )
    else:
        segments = int(args.segments)
        norm_A_for_segments = estimate_operator_norm(
            system.A_sparse_padded,
            exact_dim_limit=args.norm_exact_dim_limit,
        )
    if segments < 1:
        raise ValueError("segments must be >= 1, or 0 for automatic selection.")

    padding_steps = int(args.padding_steps)
    if padding_steps < 0:
        padding_steps = segments

    layout = build_bcow_layout(
        system_dim=system.state_dim_padded,
        t_final=args.t,
        segments=segments,
        taylor_order=args.taylor_order,
        padding_steps=padding_steps,
        pad_clock_to_power2=not args.no_pad_clock,
    )

    if progress:
        print("Algorithm: Berry-Childs-Ostrander-Wang linear-ODE construction")
        print("Mechanics case: unchanged 1D dynamic MD-FE coupling")
        print("Mechanical DOFs:", system.n_mech)
        print("Original state dimension:", system.state_dim_original)
        print("Padded system dimension:", system.state_dim_padded)
        print("System qubits:", layout.system_qubits)
        print("BCOW segments m:", layout.segments)
        print("BCOW Taylor order k:", layout.taylor_order)
        print("BCOW padding steps p:", layout.padding_steps)
        print("Time step h = T/m:", f"{layout.h_step:.8e}")
        print("Estimated ||A||_2:", f"{norm_A_for_segments:.8e}")
        print("Estimated ||A h||_2:", f"{norm_A_for_segments * abs(layout.h_step):.8e}")
        print("Active clock blocks d+1:", layout.active_clock_dim)
        print("Padded clock blocks:", layout.padded_clock_dim)
        print("Clock qubits:", layout.clock_qubits)
        print("QLSA vector dimension:", layout.total_vector_dim)
        print("Qubits without Hermitian dilation:", layout.total_qubits_without_dilation)
        print("Qubits with Hermitian dilation:", layout.total_qubits_with_hermitian_dilation)
        print("Classical emulator backend:", args.backend)

    C: sp.csr_matrix | None = None
    rhs: np.ndarray | None = None
    needs_C = (
        args.backend == "sparse-direct"
        or args.residual_check
        or args.save_linear_system
        or args.save_hermitian_dilation
        or args.condition_number_max_dim > 0
    )
    if needs_C:
        if progress:
            print("Building sparse BCOW matrix C ...")
        t0 = time.perf_counter()
        C = build_bcow_sparse_matrix(system.A_sparse_padded, layout)
        rhs = build_bcow_rhs(system.y0_padded, layout)
        if progress:
            print(
                "Sparse C:",
                f"shape={C.shape}, nnz={C.nnz}, build_time={time.perf_counter() - t0:.3f}s",
            )
    else:
        rhs = None

    history: np.ndarray | None = None
    final_approx_padded: np.ndarray | None = None
    backend_seconds = 0.0
    residual_rel: float | None = None
    history_norm = float("nan")
    final_norm = float("nan")
    final_subspace_probability = float("nan")
    final_copy_relative_mismatch = float("nan")

    if args.backend == "recurrence":
        if progress:
            print("Solving BCOW system by matrix-free lower-triangular recurrence emulator ...")
        t0 = time.perf_counter()
        history = solve_bcow_by_recurrence(system.A_sparse_padded, system.y0_padded, layout)
        backend_seconds = time.perf_counter() - t0
    elif args.backend == "sparse-direct":
        if C is None or rhs is None:
            C = build_bcow_sparse_matrix(system.A_sparse_padded, layout)
            rhs = build_bcow_rhs(system.y0_padded, layout)
        if progress:
            print("Solving sparse BCOW linear system with scipy.spsolve ...")
        t0 = time.perf_counter()
        history = solve_bcow_sparse_direct(C, rhs, layout)
        backend_seconds = time.perf_counter() - t0
    elif args.backend == "none":
        if progress:
            print("No classical solve requested; exported/metadata-only run.")
    else:  # pragma: no cover
        raise ValueError(f"Unknown backend {args.backend!r}.")

    if history is not None:
        final_approx_padded, history_norm, final_subspace_probability, final_copy_relative_mismatch = (
            extract_final_vector(history, layout, mode=args.extract_final)
        )
        final_norm = float(np.linalg.norm(final_approx_padded))

        if args.residual_check or args.backend == "sparse-direct":
            if C is None:
                C = build_bcow_sparse_matrix(system.A_sparse_padded, layout)
            if rhs is None:
                rhs = build_bcow_rhs(system.y0_padded, layout)
            residual_rel = bcow_residual_relative_norm(C, rhs, history)

    condition_number: float | None = None
    if C is not None and args.condition_number_max_dim > 0:
        condition_number = dense_condition_number_if_small(C, args.condition_number_max_dim)

    x_exact_original: np.ndarray | None = None
    abs_err: float | None = None
    rel_err: float | None = None
    state_distance: float | None = None
    overlap_abs: float | None = None
    if not args.skip_exact and final_approx_padded is not None:
        if progress:
            print("Computing expm_multiply reference ...")
        x_exact_original = expm_times_vec(system.A_sparse_original, args.t, system.y0_original)
        final_original = final_approx_padded[: system.state_dim_original]
        abs_err = float(np.linalg.norm(final_original - x_exact_original))
        rel_err = float(abs_err / max(np.linalg.norm(x_exact_original), 1e-15))
        state_distance, overlap_abs = normalized_state_metrics(final_original, x_exact_original)

    export_paths: dict[str, str] = {}
    if args.save_linear_system or args.save_hermitian_dilation:
        if C is None:
            C = build_bcow_sparse_matrix(system.A_sparse_padded, layout)
        if rhs is None:
            rhs = build_bcow_rhs(system.y0_padded, layout)
        export_paths = save_sparse_exports(
            args.output_prefix,
            C,
            rhs,
            save_hermitian_dilation=args.save_hermitian_dilation,
        )

    diagnostics = BCOWDiagnostics(
        backend=args.backend,
        wall_time_seconds=backend_seconds,
        history_norm=history_norm,
        final_norm=final_norm,
        final_subspace_probability=final_subspace_probability,
        final_copy_relative_mismatch=final_copy_relative_mismatch,
        residual_relative_norm=residual_rel,
        vector_abs_error_vs_expm=abs_err,
        vector_rel_error_vs_expm=rel_err,
        normalized_state_distance_vs_expm=state_distance,
        overlap_abs_vs_expm=overlap_abs,
    )

    metadata: dict[str, object] = {
        "algorithm": "Berry-Childs-Ostrander-Wang BCOW linear ODE construction",
        "mechanics_case": "1D dynamic MD-FE coupling, y_dot = A y, y=[u;v]",
        "mechanics_builder_changed": False,
        "classical_backend_note": (
            "The selected backend is a classical emulator of the BCOW linear system. "
            "A hardware/fault-tolerant quantum implementation would replace it with a QLSA "
            "on C or the Hermitian dilation."
        ),
        "n1": args.n1,
        "n2": args.n2,
        "mechanical_dofs": system.n_mech,
        "state_dim_original": system.state_dim_original,
        "state_dim_padded": system.state_dim_padded,
        "sparsity_A_original_nnz": int(system.A_sparse_original.nnz),
        "sparsity_A_padded_nnz": int(system.A_sparse_padded.nnz),
        "estimated_A_norm_2": norm_A_for_segments,
        "estimated_Ah_norm_2": norm_A_for_segments * abs(layout.h_step),
        "t_final": args.t,
        "layout": asdict(layout),
        "diagnostics": asdict(diagnostics),
        "condition_number_C_if_computed": condition_number,
        "mechanics_build_seconds": mechanics_build_seconds,
        "sparse_C_shape": None if C is None else list(C.shape),
        "sparse_C_nnz": None if C is None else int(C.nnz),
        "export_paths": export_paths,
    }

    npz_path, csv_path, json_path = save_results(
        args.output_prefix,
        system,
        layout,
        final_approx_padded,
        x_exact_original,
        metadata,
        history=history if args.save_history else None,
    )

    if progress:
        print("\nFinal BCOW results")
        print("Backend:", args.backend)
        print("Backend wall time:", f"{backend_seconds:.3f}s")
        if history is not None:
            print("History norm:", f"{history_norm:.8e}")
            print("Final vector norm:", f"{final_norm:.8e}")
            print("Final clock-subspace probability:", f"{final_subspace_probability:.8e}")
            print("Final copy relative mismatch:", f"{final_copy_relative_mismatch:.8e}")
        if residual_rel is not None:
            print("Relative BCOW linear-system residual:", f"{residual_rel:.8e}")
        if condition_number is not None:
            print("Dense condition number of C:", f"{condition_number:.8e}")
        if abs_err is not None and rel_err is not None:
            print("Absolute error vs expm_multiply:", f"{abs_err:.8e}")
            print("Relative error vs expm_multiply:", f"{rel_err:.8e}")
            print("Normalized quantum-state distance:", f"{state_distance:.8e}")
            print("|Normalized overlap|:", f"{overlap_abs:.8e}")
        print("Saved numerical results to", npz_path)
        print("Saved metadata JSON to", json_path)
        if csv_path is not None:
            print("Saved displacement CSV to", csv_path)
        for label, path in export_paths.items():
            print(f"Saved {label} to {path}")


if __name__ == "__main__":
    main()
