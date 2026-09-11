"""Postprocess retained Case III data into a nonlinear error budget.

This script does not run VQLS optimization.  It combines the retained VQLS
displacement history with four inexpensive classical trajectories:

1. high-accuracy integration of the full Lennard--Jones equations;
2. high-accuracy integration of the quadratic-force nonlinear equations;
3. exact continuous evolution of the second-order Carleman system; and
4. classical backward-Euler evolution of that Carleman system.

Consecutive comparisons isolate force-expansion, Carleman-truncation,
time-discretization, and VQLS/optimization errors.  The retained VQLS artifact
does not contain a velocity history, so velocity curves are reconstructed from
every displacement trajectory with backward differences for frames 1 through
150 and the prescribed zero initial velocity at frame 0. This corrects the
first-revision postprocessor as documented in the second-round response.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = PACKAGE_DIR / "inputs" / "200steps_result_cpu_diagnostics.npz"
DEFAULT_H5 = PACKAGE_DIR / "inputs" / "200steps_result_cpu.h5"
OUTPUT_DIR = PACKAGE_DIR / "reproduced"
DEFAULT_BY_TIME = OUTPUT_DIR / "case3_error_decomposition_by_time.csv"
DEFAULT_SUMMARY = OUTPUT_DIR / "case3_error_decomposition_summary.csv"
DEFAULT_TRAJECTORIES = OUTPUT_DIR / "case3_validation_trajectories.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_H5)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--spacing", type=float, default=1.0, help="Equilibrium spacing a in nm.")
    parser.add_argument("--epsilon", type=float, default=1.65)
    parser.add_argument("--mass", type=float, default=1.993)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--atol", type=float, default=1.0e-14)
    parser.add_argument("--by-time-output", type=Path, default=DEFAULT_BY_TIME)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--trajectories-output", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--states-output", type=Path, default=OUTPUT_DIR / "case3_reference_states.npz")
    return parser.parse_args()


def lj_derivative(r: np.ndarray, sigma: float, epsilon: float) -> np.ndarray:
    """Return U'(r) for the Lennard--Jones potential."""
    return 4.0 * epsilon * (-12.0 * sigma**12 / r**13 + 6.0 * sigma**6 / r**7)


def lj_taylor_coefficients(spacing: float, sigma: float, epsilon: float) -> tuple[float, float]:
    """Return c1=U''(a) and c2=U'''(a)/2."""
    a = spacing
    c1 = 4.0 * epsilon * (156.0 * sigma**12 / a**14 - 42.0 * sigma**6 / a**8)
    c2 = 0.5 * 4.0 * epsilon * (-2184.0 * sigma**12 / a**15 + 336.0 * sigma**6 / a**9)
    return float(c1), float(c2)


def propagate(step_matrix: np.ndarray, initial: np.ndarray, count: int) -> np.ndarray:
    states = np.empty((count, initial.size), dtype=float)
    states[0] = np.asarray(initial, dtype=float)
    for index in range(1, count):
        states[index] = step_matrix @ states[index - 1]
    return states


def metric_row(
    comparison: str,
    interpretation: str,
    left_q: np.ndarray,
    right_q: np.ndarray,
    left_v: np.ndarray,
    right_v: np.ndarray,
    final_time: float,
) -> dict[str, object]:
    dq = np.asarray(left_q - right_q, dtype=float)
    dv = np.asarray(left_v - right_v, dtype=float)
    return {
        "time_interval_ps": f"0-{final_time:g}",
        "comparison": comparison,
        "error_isolated": interpretation,
        "max_abs_displacement_error_nm": float(np.max(np.abs(dq))),
        "rms_displacement_error_nm": float(np.sqrt(np.mean(dq**2))),
        "mean_abs_displacement_error_nm": float(np.mean(np.abs(dq))),
        "max_abs_velocity_error_nm_per_ps": float(np.max(np.abs(dv))),
        "rms_velocity_error_nm_per_ps": float(np.sqrt(np.mean(dv**2))),
        "mean_abs_velocity_error_nm_per_ps": float(np.mean(np.abs(dv))),
    }


def main() -> None:
    args = parse_args()

    with np.load(args.diagnostics, allow_pickle=False) as data:
        carleman_matrix = np.asarray(data["A"], dtype=float)
        backward_euler_matrix = np.asarray(data["M"], dtype=float)
        metadata = json.loads(str(data["metadata"].item()))

    with h5py.File(args.trajectory, "r") as data:
        saved_times = np.asarray(data["t"][: args.n_steps + 1], dtype=float)
        vqls_q = np.asarray(data["y"][: args.n_steps + 1], dtype=float)

    if len(saved_times) != args.n_steps + 1 or len(vqls_q) != args.n_steps + 1:
        raise ValueError("The retained trajectory does not contain the requested number of steps.")
    if saved_times.size < 2:
        raise ValueError("At least two retained time samples are required.")
    step_sizes = np.diff(saved_times)
    dt = float(np.median(step_sizes))
    if not np.allclose(step_sizes, dt, rtol=0.0, atol=1.0e-12):
        raise ValueError("This postprocessor expects uniformly spaced retained times.")

    atom_count = int(metadata["N"])
    if vqls_q.shape[1] != atom_count:
        raise ValueError("The retained displacement width does not match metadata['N'].")
    if carleman_matrix.shape != backward_euler_matrix.shape:
        raise ValueError("Stored Carleman and backward-Euler matrices have inconsistent shapes.")
    if not np.allclose(
        backward_euler_matrix,
        np.eye(carleman_matrix.shape[0]) - dt * carleman_matrix,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("The stored backward-Euler matrix is inconsistent with A and dt.")

    spacing = float(args.spacing)
    sigma = spacing / (2.0 ** (1.0 / 6.0))
    c1, c2 = lj_taylor_coefficients(spacing, sigma, float(args.epsilon))
    if not np.isclose(c1, float(metadata["c1"]), rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("Reconstructed c1 does not match the retained run metadata.")
    if not np.isclose(c2, float(metadata["c2"]), rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("Reconstructed c2 does not match the retained run metadata.")

    incidence = np.zeros((atom_count - 1, atom_count), dtype=float)
    for index in range(atom_count - 1):
        incidence[index, index] = -1.0
        incidence[index, index + 1] = 1.0
    bond_to_atom = -incidence.T

    initial_q = vqls_q[0].copy()
    initial_z = np.concatenate([initial_q, np.zeros(atom_count, dtype=float)])
    initial_y = np.concatenate([initial_z, np.kron(initial_z, initial_z)])
    if initial_y.size != carleman_matrix.shape[0]:
        raise ValueError("The reconstructed lifted initial state has the wrong dimension.")

    def full_lj_rhs(_time: float, state: np.ndarray) -> np.ndarray:
        q = state[:atom_count]
        velocity = state[atom_count:]
        bond_extension = incidence @ q
        bond_force = lj_derivative(spacing + bond_extension, sigma, float(args.epsilon))
        acceleration = (bond_to_atom @ bond_force) / float(args.mass)
        return np.concatenate([velocity, acceleration])

    def quadratic_rhs(_time: float, state: np.ndarray) -> np.ndarray:
        q = state[:atom_count]
        velocity = state[atom_count:]
        bond_extension = incidence @ q
        bond_force = c1 * bond_extension + c2 * bond_extension**2
        acceleration = (bond_to_atom @ bond_force) / float(args.mass)
        return np.concatenate([velocity, acceleration])

    integration_options = {
        "method": "DOP853",
        "t_eval": saved_times,
        "rtol": float(args.rtol),
        "atol": float(args.atol),
    }
    full_lj_solution = solve_ivp(
        full_lj_rhs,
        (float(saved_times[0]), float(saved_times[-1])),
        initial_z,
        **integration_options,
    )
    quadratic_solution = solve_ivp(
        quadratic_rhs,
        (float(saved_times[0]), float(saved_times[-1])),
        initial_z,
        **integration_options,
    )
    if not full_lj_solution.success or not quadratic_solution.success:
        raise RuntimeError("At least one high-accuracy classical integration failed.")

    full_lj_q = np.asarray(full_lj_solution.y[:atom_count].T, dtype=float)
    quadratic_q = np.asarray(quadratic_solution.y[:atom_count].T, dtype=float)
    continuous_step = expm(dt * carleman_matrix)
    continuous_y = propagate(continuous_step, initial_y, len(saved_times))
    backward_euler_step = np.linalg.solve(backward_euler_matrix, np.eye(backward_euler_matrix.shape[0]))
    backward_euler_y = propagate(backward_euler_step, initial_y, len(saved_times))
    continuous_q = continuous_y[:, :atom_count]
    backward_euler_q = backward_euler_y[:, :atom_count]

    trajectories = {
        "full_lj": full_lj_q,
        "quadratic_nonlinear": quadratic_q,
        "continuous_k2_carleman": continuous_q,
        "backward_euler_k2_carleman": backward_euler_q,
        "vqls": vqls_q,
    }
    velocities = {
        name: np.vstack((np.zeros((1, atom_count)), np.diff(q, axis=0) / dt))
        for name, q in trajectories.items()
    }

    comparisons = [
        (
            "Quadratic-force nonlinear vs full Lennard-Jones",
            "Force expansion",
            "quadratic_nonlinear",
            "full_lj",
        ),
        (
            "Continuous K=2 Carleman vs quadratic-force nonlinear",
            "Carleman truncation",
            "continuous_k2_carleman",
            "quadratic_nonlinear",
        ),
        (
            "Backward Euler K=2 vs continuous K=2 Carleman",
            "Time discretization",
            "backward_euler_k2_carleman",
            "continuous_k2_carleman",
        ),
        (
            "VQLS vs backward Euler K=2",
            "VQLS/optimization",
            "vqls",
            "backward_euler_k2_carleman",
        ),
        (
            "VQLS vs full Lennard-Jones",
            "Total",
            "vqls",
            "full_lj",
        ),
    ]

    summary_rows: list[dict[str, object]] = []
    by_time_rows: list[dict[str, object]] = []
    for comparison, interpretation, left_name, right_name in comparisons:
        left_q = trajectories[left_name]
        right_q = trajectories[right_name]
        left_v = velocities[left_name]
        right_v = velocities[right_name]
        summary_rows.append(
            metric_row(
                comparison,
                interpretation,
                left_q,
                right_q,
                left_v,
                right_v,
                float(saved_times[-1]),
            )
        )
        for index, time in enumerate(saved_times):
            dq = left_q[index] - right_q[index]
            dv = left_v[index] - right_v[index]
            by_time_rows.append(
                {
                    "time_ps": float(time),
                    "comparison": comparison,
                    "error_isolated": interpretation,
                    "max_abs_displacement_error_nm": float(np.max(np.abs(dq))),
                    "rms_displacement_error_nm": float(np.sqrt(np.mean(dq**2))),
                    "max_abs_velocity_error_nm_per_ps": float(np.max(np.abs(dv))),
                    "rms_velocity_error_nm_per_ps": float(np.sqrt(np.mean(dv**2))),
                }
            )

    trajectory_rows: list[dict[str, object]] = []
    for time_index, time in enumerate(saved_times):
        for atom_index in range(atom_count):
            row: dict[str, object] = {
                "time_ps": float(time),
                "atom_index": atom_index,
            }
            for name in trajectories:
                row[f"{name}_displacement_nm"] = float(trajectories[name][time_index, atom_index])
                row[f"{name}_reconstructed_velocity_backward_nm_per_ps"] = float(velocities[name][time_index, atom_index])
            trajectory_rows.append(row)

    for output in (args.by_time_output, args.summary_output, args.trajectories_output, args.states_output):
        output.parent.mkdir(parents=True, exist_ok=True)

    with args.by_time_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_time_rows[0]))
        writer.writeheader()
        writer.writerows(by_time_rows)

    with args.summary_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    with args.trajectories_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    np.savez_compressed(
        args.states_output, time_ps=saved_times, A_C=carleman_matrix, M_C=backward_euler_matrix,
        initial_z=initial_z, initial_Y=initial_y,
        full_lj_native_q_v=np.asarray(full_lj_solution.y.T),
        quadratic_nonlinear_native_q_v=np.asarray(quadratic_solution.y.T),
        continuous_k2_lifted_state=continuous_y,
        backward_euler_k2_lifted_state=backward_euler_y,
        vqls_displacement_nm=vqls_q,
    )

    print(
        f"High-accuracy references: DOP853, rtol={args.rtol:g}, atol={args.atol:g}; "
        f"{len(saved_times)} frames on 0-{saved_times[-1]:g} ps."
    )
    print("All Table 7 velocities use backward differences and prescribed zero initial velocities.")
    for row in summary_rows:
        print(
            f"{row['error_isolated']}: "
            f"max/RMS |dq|={row['max_abs_displacement_error_nm']:.8e}/"
            f"{row['rms_displacement_error_nm']:.8e} nm, "
            f"max/RMS |dv|={row['max_abs_velocity_error_nm_per_ps']:.8e}/"
            f"{row['rms_velocity_error_nm_per_ps']:.8e} nm/ps"
        )


if __name__ == "__main__":
    main()
