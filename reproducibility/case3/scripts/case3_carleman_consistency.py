"""Audit Case III Carleman consistency without rerunning VQLS optimization.

The reported algorithm propagates the complete unprojected lifted state.  This
postprocessor compares that classical backward-Euler update with continuous
K=2 Carleman evolution and with a sensitivity variant that re-lifts
Y2 <- kron(z, z) after every classical backward-Euler step.  It also evaluates
the single complete terminal VQLS lifted state retained in the 200-step NPZ
artifact.  No variational circuit or optimizer is executed.
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
DEFAULT_BY_TIME = OUTPUT_DIR / "case3_carleman_consistency_by_time.csv"
DEFAULT_SUMMARY = OUTPUT_DIR / "case3_carleman_consistency_summary.csv"
DEFAULT_SENSITIVITY = OUTPUT_DIR / "case3_projection_sensitivity_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_H5)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1.65)
    parser.add_argument("--mass", type=float, default=1.993)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--atol", type=float, default=1.0e-14)
    parser.add_argument("--by-time-output", type=Path, default=DEFAULT_BY_TIME)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--sensitivity-output", type=Path, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--states-output", type=Path, default=OUTPUT_DIR / "case3_consistency_states.npz")
    parser.add_argument("--trajectories-output", type=Path, default=OUTPUT_DIR / "case3_projection_trajectories.csv")
    return parser.parse_args()


def consistency(state: np.ndarray, first_order_dimension: int) -> tuple[float, float]:
    z = np.asarray(state[:first_order_dimension], dtype=float)
    lifted = np.asarray(state[first_order_dimension:], dtype=float)
    exact_lift = np.kron(z, z)
    absolute = float(np.linalg.norm(lifted - exact_lift))
    scale = max(float(np.linalg.norm(lifted)), float(np.linalg.norm(exact_lift)), np.finfo(float).eps)
    return absolute, absolute / scale


def propagate(step_matrix: np.ndarray, initial: np.ndarray, count: int) -> np.ndarray:
    states = np.empty((count, initial.size), dtype=float)
    states[0] = np.asarray(initial, dtype=float)
    for index in range(1, count):
        states[index] = step_matrix @ states[index - 1]
    return states


def difference_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    difference = np.asarray(left - right, dtype=float)
    return (
        float(np.max(np.abs(difference))),
        float(np.sqrt(np.mean(difference**2))),
        float(np.max(np.abs(difference[-1]))),
    )


def main() -> None:
    args = parse_args()

    with np.load(args.diagnostics, allow_pickle=False) as data:
        carleman_matrix = np.asarray(data["A"], dtype=float)
        backward_euler_matrix = np.asarray(data["M"], dtype=float)
        retained_vqls_terminal = np.asarray(data["Y_final"], dtype=float)
        metadata = json.loads(str(data["metadata"].item()))

    with h5py.File(args.trajectory, "r") as data:
        all_saved_times = np.asarray(data["t"], dtype=float)
        saved_times = np.asarray(data["t"][: args.n_steps + 1], dtype=float)
        saved_displacements = np.asarray(data["y"][: args.n_steps + 1], dtype=float)

    if len(saved_times) != args.n_steps + 1:
        raise ValueError("The retained HDF5 file does not contain the requested time interval.")
    time_steps = np.diff(saved_times)
    dt = float(np.median(time_steps))
    if not np.allclose(time_steps, dt, rtol=0.0, atol=1.0e-12):
        raise ValueError("This audit expects uniformly spaced Case III frames.")
    if not np.isclose(float(metadata["dt"]), dt, rtol=0.0, atol=1.0e-12):
        raise ValueError("The HDF5 time step is inconsistent with the retained metadata.")
    if not np.allclose(
        backward_euler_matrix,
        np.eye(carleman_matrix.shape[0]) - dt * carleman_matrix,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("The stored backward-Euler matrix is inconsistent with A and dt.")

    atom_count = int(metadata["N"])
    first_order_dimension = 2 * atom_count
    initial_z = np.concatenate([saved_displacements[0], np.zeros(atom_count, dtype=float)])
    initial_lifted = np.concatenate([initial_z, np.kron(initial_z, initial_z)])
    if initial_lifted.size != carleman_matrix.shape[0]:
        raise ValueError("The reconstructed initial lifted state has an inconsistent dimension.")

    continuous_step = expm(dt * carleman_matrix)
    backward_euler_step = np.linalg.solve(
        backward_euler_matrix,
        np.eye(backward_euler_matrix.shape[0]),
    )
    continuous_states = propagate(continuous_step, initial_lifted, len(saved_times))
    unprojected_states = propagate(backward_euler_step, initial_lifted, len(saved_times))

    relifted_states = np.empty_like(unprojected_states)
    relifted_states[0] = initial_lifted
    relifted_raw_states = np.empty_like(unprojected_states)
    relifted_raw_states[0] = initial_lifted
    relifted_preprojection_absolute = np.zeros(len(saved_times), dtype=float)
    relifted_preprojection_relative = np.zeros(len(saved_times), dtype=float)
    for index in range(1, len(saved_times)):
        raw_state = backward_euler_step @ relifted_states[index - 1]
        relifted_raw_states[index] = raw_state
        absolute, relative = consistency(raw_state, first_order_dimension)
        relifted_preprojection_absolute[index] = absolute
        relifted_preprojection_relative[index] = relative
        z = raw_state[:first_order_dimension]
        relifted_states[index] = np.concatenate([z, np.kron(z, z)])

    continuous_consistency = np.asarray(
        [consistency(state, first_order_dimension) for state in continuous_states]
    )
    unprojected_consistency = np.asarray(
        [consistency(state, first_order_dimension) for state in unprojected_states]
    )
    relifted_post_consistency = np.asarray(
        [consistency(state, first_order_dimension) for state in relifted_states]
    )

    retained_step_count = int(metadata["n_steps"])
    retained_terminal_time = retained_step_count * dt
    if len(all_saved_times) != retained_step_count + 1:
        raise ValueError("The retained terminal lifted state and HDF5 frame count are inconsistent.")
    if not np.isclose(all_saved_times[-1], retained_terminal_time, rtol=0.0, atol=1.0e-12):
        raise ValueError("The retained terminal lifted state time cannot be verified.")
    retained_vqls_absolute, retained_vqls_relative = consistency(
        retained_vqls_terminal,
        first_order_dimension,
    )

    by_time_rows: list[dict[str, object]] = []
    for index, time in enumerate(saved_times):
        unprojected_relifted_q = (
            unprojected_states[index, :atom_count] - relifted_states[index, :atom_count]
        )
        unprojected_relifted_v = (
            unprojected_states[index, atom_count:first_order_dimension]
            - relifted_states[index, atom_count:first_order_dimension]
        )
        by_time_rows.append(
            {
                "time_ps": float(time),
                "continuous_k2_eta": float(continuous_consistency[index, 0]),
                "continuous_k2_eta_relative": float(continuous_consistency[index, 1]),
                "unprojected_be_eta": float(unprojected_consistency[index, 0]),
                "unprojected_be_eta_relative": float(unprojected_consistency[index, 1]),
                "relifted_be_preprojection_eta": float(relifted_preprojection_absolute[index]),
                "relifted_be_preprojection_eta_relative": float(relifted_preprojection_relative[index]),
                "relifted_be_postprojection_eta": float(relifted_post_consistency[index, 0]),
                "relifted_be_postprojection_eta_relative": float(relifted_post_consistency[index, 1]),
                "max_abs_unprojected_vs_relifted_displacement_nm": float(
                    np.max(np.abs(unprojected_relifted_q))
                ),
                "max_abs_unprojected_vs_relifted_native_velocity_nm_per_ps": float(
                    np.max(np.abs(unprojected_relifted_v))
                ),
            }
        )

    summary_rows: list[dict[str, object]] = [
        {
            "trajectory": "Continuous K=2 Carleman",
            "time_interval_ps": f"0-{saved_times[-1]:g}",
            "update_or_evaluation_point": "Continuous truncated evolution",
            "max_eta": float(np.max(continuous_consistency[:, 0])),
            "terminal_eta": float(continuous_consistency[-1, 0]),
            "max_eta_relative": float(np.max(continuous_consistency[:, 1])),
            "terminal_eta_relative": float(continuous_consistency[-1, 1]),
        },
        {
            "trajectory": "Backward Euler K=2, unprojected",
            "time_interval_ps": f"0-{saved_times[-1]:g}",
            "update_or_evaluation_point": "Complete lifted state retained",
            "max_eta": float(np.max(unprojected_consistency[:, 0])),
            "terminal_eta": float(unprojected_consistency[-1, 0]),
            "max_eta_relative": float(np.max(unprojected_consistency[:, 1])),
            "terminal_eta_relative": float(unprojected_consistency[-1, 1]),
        },
        {
            "trajectory": "Backward Euler K=2, re-lifted (before projection)",
            "time_interval_ps": f"0-{saved_times[-1]:g}",
            "update_or_evaluation_point": "Raw state before Y2 <- kron(z,z)",
            "max_eta": float(np.max(relifted_preprojection_absolute)),
            "terminal_eta": float(relifted_preprojection_absolute[-1]),
            "max_eta_relative": float(np.max(relifted_preprojection_relative)),
            "terminal_eta_relative": float(relifted_preprojection_relative[-1]),
        },
        {
            "trajectory": "Backward Euler K=2, re-lifted (after projection)",
            "time_interval_ps": f"0-{saved_times[-1]:g}",
            "update_or_evaluation_point": "Projected state passed to next step",
            "max_eta": float(np.max(relifted_post_consistency[:, 0])),
            "terminal_eta": float(relifted_post_consistency[-1, 0]),
            "max_eta_relative": float(np.max(relifted_post_consistency[:, 1])),
            "terminal_eta_relative": float(relifted_post_consistency[-1, 1]),
        },
        {
            "trajectory": "Retained VQLS terminal lifted state",
            "time_interval_ps": f"terminal-only at {retained_terminal_time:g}",
            "update_or_evaluation_point": "Saved Y_final from retained 200-step artifact",
            "max_eta": "",
            "terminal_eta": retained_vqls_absolute,
            "max_eta_relative": "",
            "terminal_eta_relative": retained_vqls_relative,
        },
    ]

    spacing = float(args.spacing)
    sigma = spacing / (2.0 ** (1.0 / 6.0))
    incidence = np.zeros((atom_count - 1, atom_count), dtype=float)
    for index in range(atom_count - 1):
        incidence[index, index] = -1.0
        incidence[index, index + 1] = 1.0
    bond_to_atom = -incidence.T

    def full_lj_rhs(_time: float, state: np.ndarray) -> np.ndarray:
        q = state[:atom_count]
        velocity = state[atom_count:]
        bond_length = spacing + incidence @ q
        bond_force = 4.0 * float(args.epsilon) * (
            -12.0 * sigma**12 / bond_length**13 + 6.0 * sigma**6 / bond_length**7
        )
        acceleration = (bond_to_atom @ bond_force) / float(args.mass)
        return np.concatenate([velocity, acceleration])

    full_lj_solution = solve_ivp(
        full_lj_rhs,
        (float(saved_times[0]), float(saved_times[-1])),
        initial_z,
        method="DOP853",
        t_eval=saved_times,
        rtol=float(args.rtol),
        atol=float(args.atol),
    )
    if not full_lj_solution.success:
        raise RuntimeError("The full Lennard--Jones reference integration failed.")
    full_lj = np.asarray(full_lj_solution.y.T, dtype=float)

    physical_trajectories = {
        "Re-lifted BE vs unprojected BE": (
            "Projection choice",
            relifted_states[:, :first_order_dimension],
            unprojected_states[:, :first_order_dimension],
        ),
        "Unprojected BE vs full Lennard-Jones": (
            "Current classical accuracy",
            unprojected_states[:, :first_order_dimension],
            full_lj,
        ),
        "Re-lifted BE vs full Lennard-Jones": (
            "Re-lifted sensitivity accuracy",
            relifted_states[:, :first_order_dimension],
            full_lj,
        ),
    }
    sensitivity_rows: list[dict[str, object]] = []
    for comparison, (interpretation, left, right) in physical_trajectories.items():
        q_max, q_rms, q_terminal = difference_metrics(
            left[:, :atom_count],
            right[:, :atom_count],
        )
        v_max, v_rms, v_terminal = difference_metrics(
            left[:, atom_count:first_order_dimension],
            right[:, atom_count:first_order_dimension],
        )
        sensitivity_rows.append(
            {
                "time_interval_ps": f"0-{saved_times[-1]:g}",
                "comparison": comparison,
                "interpretation": interpretation,
                "max_abs_displacement_error_nm": q_max,
                "rms_displacement_error_nm": q_rms,
                "terminal_max_abs_displacement_error_nm": q_terminal,
                "max_abs_native_velocity_error_nm_per_ps": v_max,
                "rms_native_velocity_error_nm_per_ps": v_rms,
                "terminal_max_abs_native_velocity_error_nm_per_ps": v_terminal,
            }
        )

    for output in (args.by_time_output, args.summary_output, args.sensitivity_output,
                   args.states_output, args.trajectories_output):
        output.parent.mkdir(parents=True, exist_ok=True)

    with args.by_time_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_time_rows[0]))
        writer.writeheader()
        writer.writerows(by_time_rows)

    with args.summary_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    with args.sensitivity_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    np.savez_compressed(
        args.states_output, time_ps=saved_times, initial_Y=initial_lifted,
        continuous_k2_lifted_state=continuous_states,
        unprojected_be_lifted_state=unprojected_states,
        relifted_be_preprojection_state=relifted_raw_states,
        relifted_be_postprojection_state=relifted_states,
        full_lj_native_q_v=full_lj,
        retained_vqls_terminal_state=retained_vqls_terminal,
        retained_vqls_terminal_time_ps=retained_terminal_time,
    )
    trajectory_rows = []
    native = {"full_lj": full_lj, "continuous_k2": continuous_states,
              "unprojected_be": unprojected_states, "relifted_be": relifted_states}
    for index, time in enumerate(saved_times):
        for atom in range(atom_count):
            row = {"time_ps": float(time), "atom_index": atom}
            for name, states in native.items():
                row[name + "_displacement_nm"] = float(states[index, atom])
                row[name + "_native_velocity_nm_per_ps"] = float(states[index, atom_count + atom])
            trajectory_rows.append(row)
    with args.trajectories_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    print(f"Classical consistency interval: 0-{saved_times[-1]:g} ps ({len(saved_times)} frames).")
    for row in summary_rows:
        print(
            f"{row['trajectory']}: max eta={row['max_eta']}, terminal eta={row['terminal_eta']}, "
            f"max eta_rel={row['max_eta_relative']}, terminal eta_rel={row['terminal_eta_relative']}"
        )
    for row in sensitivity_rows:
        print(
            f"{row['comparison']}: max/RMS |dq|="
            f"{row['max_abs_displacement_error_nm']:.8e}/"
            f"{row['rms_displacement_error_nm']:.8e} nm, max/RMS native |dv|="
            f"{row['max_abs_native_velocity_error_nm_per_ps']:.8e}/"
            f"{row['rms_native_velocity_error_nm_per_ps']:.8e} nm/ps"
        )


if __name__ == "__main__":
    main()
