"""Verify numerical reproduction and independent mechanics/state identities."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy.linalg import expm
import h5py

PACKAGE = Path(__file__).resolve().parents[1]


def verify_manifest():
    manifest = json.loads((PACKAGE / "manifest_sha256.json").read_text())
    for relative, expected in manifest["files"].items():
        file = (PACKAGE / relative).resolve()
        assert file.is_relative_to(PACKAGE.resolve()), relative
        assert hashlib.sha256(file.read_bytes()).hexdigest() == expected, relative
    return len(manifest["files"])


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def compare_csv(reference, actual):
    left, right = rows(reference), rows(actual)
    assert len(left) == len(right), reference.name
    maximum = 0.0
    for a, b in zip(left, right):
        assert a.keys() == b.keys(), reference.name
        for key in a:
            try:
                x, y = float(a[key]), float(b[key])
            except ValueError:
                assert a[key] == b[key], (reference.name, key, a[key], b[key])
            else:
                assert np.isclose(x, y, rtol=1e-9, atol=1e-12), (reference.name, key, x, y)
                maximum = max(maximum, abs(x - y))
    return maximum


def independent_checks(output):
    with np.load(output / "case3_reference_states.npz", allow_pickle=False) as data:
        A, M = data["A_C"], data["M_C"]
        t, initial = data["time_ps"], data["initial_Y"]
        continuous = data["continuous_k2_lifted_state"]
        backward = data["backward_euler_k2_lifted_state"]
        vqls_q = data["vqls_displacement_nm"]
    config = json.loads((PACKAGE / "configuration.json").read_text())
    assert len(t) == 151 and np.isclose(t[-1], 0.300)
    assert np.array_equal(initial[:6], np.array(config["initial_displacement_nm"] + config["initial_velocity_nm_per_ps"]))
    assert np.allclose(M, np.eye(42) - config["dt_ps"] * A, rtol=1e-12, atol=1e-12)
    # Reconstruct the ordered K=2 matrix independently from the physical inputs.
    a, epsilon, mass = config["spacing_nm"], config["epsilon_aJ"], config["mass_in_1e_minus24_kg"]
    sigma = a / 2 ** (1 / 6)
    c1 = 4 * epsilon * (156 * sigma ** 12 / a ** 14 - 42 * sigma ** 6 / a ** 8)
    c2 = 2 * epsilon * (-2184 * sigma ** 12 / a ** 15 + 336 * sigma ** 6 / a ** 9)
    T = np.array([[-1., 1., 0.], [0., -1., 1.]])
    S = -T.T
    A11 = np.zeros((6, 6))
    A11[:3, 3:] = np.eye(3)
    A11[3:, :3] = c1 / mass * (S @ T)
    L2 = sum(S[:, i:i+1] @ np.kron(T[i:i+1], T[i:i+1]) for i in range(2))
    E = np.hstack((np.eye(3), np.zeros((3, 3))))
    A12 = np.vstack((np.zeros((3, 36)), c2 / mass * L2 @ np.kron(E, E)))
    reconstructed_A = np.block([[A11, A12], [np.zeros((36, 6)),
                               np.kron(np.eye(6), A11) + np.kron(A11, np.eye(6))]])
    assert np.allclose(A, reconstructed_A, rtol=1e-12, atol=1e-12)
    assert np.allclose(continuous[-1], expm(t[-1] * A) @ initial, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(backward[1:] @ M.T - backward[:-1])) < 1e-12
    with h5py.File(PACKAGE / "inputs/200steps_result_cpu.h5") as data:
        assert np.array_equal(vqls_q, data["y"][:151])
        assert len(data["t"]) == 201 and np.isclose(data["t"][-1], 0.4)
    with np.load(output / "case3_consistency_states.npz", allow_pickle=False) as data:
        raw = data["relifted_be_preprojection_state"]
        projected = data["relifted_be_postprojection_state"]
        assert np.array_equal(raw[:, :6], projected[:, :6])
        lifts = np.array([np.kron(z, z) for z in projected[:, :6]])
        assert np.array_equal(projected[:, 6:], lifts)
        assert np.max(np.abs(raw[1:] @ M.T - projected[:-1])) < 1e-12
        assert np.isclose(float(data["retained_vqls_terminal_time_ps"]), 0.4)
    original_figure6 = rows(PACKAGE / "inputs/published_figure6_series.csv")
    regenerated_figure6 = rows(output / "figure6_series.csv")
    mapping = {"vqls_displacement": "vqls_displacement_nm",
               "classical_displacement": "backward_euler_k2_displacement_nm",
               "vqls_velocity_reconstructed": "vqls_reconstructed_velocity_backward_nm_per_ps",
               "classical_velocity": "backward_euler_k2_reconstructed_velocity_backward_nm_per_ps"}
    assert len(original_figure6) == len(regenerated_figure6) == 453
    for original, regenerated in zip(original_figure6, regenerated_figure6):
        assert int(original["frame"]) == int(regenerated["frame"])
        assert int(original["atom"]) == int(regenerated["atom"])
        for old, new in mapping.items():
            assert np.isclose(float(original[old]), float(regenerated[new]), rtol=0, atol=5e-13)
    old_costs = rows(PACKAGE / "inputs/published_figure7_cost.csv")
    new_costs = rows(output / "figure7_cost_full_precision.csv")
    assert len(old_costs) == len(new_costs) == 150
    for old, new in zip(old_costs, new_costs):
        assert int(old["step"]) == int(new["step"])
        assert np.isclose(float(old["time"]), float(new["time_ps"]), rtol=0, atol=1e-12)
        assert format(float(old["vqls_cost"]), ".2e") == format(float(new["vqls_cost"]), ".2e")
    return {
        "matrix_rebuilt_from_physical_parameters": True,
        "continuous_terminal_matches_direct_matrix_exponential": True,
        "backward_euler_recurrence_verified": True,
        "retained_vqls_displacements_unchanged": True,
        "projection_preserves_first_order_state": True,
        "projected_second_order_state_exact_kron": True,
        "retained_terminal_state_is_0_400_ps_not_table8": True,
        "public_figure6_series_reproduced": True,
        "public_figure7_costs_match_at_published_precision": True,
    }


def verify(output, reference=PACKAGE / "results"):
    discrepancies = {}
    for file in sorted(reference.iterdir()):
        actual = output / file.name
        assert actual.is_file(), file.name
        if file.suffix == ".csv":
            discrepancies[file.name] = compare_csv(file, actual)
        elif file.suffix == ".npz":
            maximum = 0.0
            with np.load(file, allow_pickle=False) as a, np.load(actual, allow_pickle=False) as b:
                assert a.files == b.files
                for key in a.files:
                    assert a[key].shape == b[key].shape, (file.name, key)
                    assert np.allclose(a[key], b[key], rtol=1e-9, atol=1e-12), (file.name, key)
                    maximum = max(maximum, float(np.max(np.abs(a[key] - b[key]))))
            discrepancies[file.name] = maximum
        elif file.suffix == ".tex":
            assert file.read_text() == actual.read_text(), file.name
        elif file.suffix == ".json":
            a, b = json.loads(file.read_text()), json.loads(actual.read_text())
            assert a.keys() == b.keys()
            assert all(np.isclose(a[key], b[key], rtol=1e-9, atol=1e-12) for key in a)
    return {"all_archived_results_reproduced": True,
            "rtol": 1e-9, "atol": 1e-12,
            "maximum_absolute_differences": discrepancies,
            "independent_checks": independent_checks(output)}
