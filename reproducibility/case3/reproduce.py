"""Regenerate the complete Case III validation package without VQLS optimization."""
from pathlib import Path
import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "reproduced")
    parser.add_argument("--build-reference", action="store_true",
                        help="Maintainer option: generate a new archived result set without comparing against results/")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == (PACKAGE / "results").resolve() and not args.build_reference:
        parser.error("Use a separate output directory to preserve the archived reference results.")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((PACKAGE / "configuration.json").read_text())
    spec = importlib.util.spec_from_file_location("verify_results", PACKAGE / "scripts/verify_results.py")
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    if not args.build_reference:
        count = checks.verify_manifest()
        print(f"Verified SHA-256 hashes of {count} archived files.", flush=True)
    common = ["--diagnostics", str(PACKAGE / "inputs/200steps_result_cpu_diagnostics.npz"),
              "--trajectory", str(PACKAGE / "inputs/200steps_result_cpu.h5"),
              "--n-steps", str(config["n_steps"]), "--spacing", str(config["spacing_nm"]),
              "--epsilon", str(config["epsilon_aJ"]), "--mass", str(config["mass_in_1e_minus24_kg"]),
              "--rtol", str(config["rtol"]), "--atol", str(config["atol"])]
    start = time.perf_counter()
    commands = [
        [sys.executable, "-B", str(PACKAGE / "scripts/case3_error_decomposition.py"), *common,
         "--by-time-output", str(output / "case3_error_decomposition_by_time.csv"),
         "--summary-output", str(output / "case3_error_decomposition_summary.csv"),
         "--trajectories-output", str(output / "case3_validation_trajectories.csv"),
         "--states-output", str(output / "case3_reference_states.npz")],
        [sys.executable, "-B", str(PACKAGE / "scripts/case3_carleman_consistency.py"), *common,
         "--by-time-output", str(output / "case3_carleman_consistency_by_time.csv"),
         "--summary-output", str(output / "case3_carleman_consistency_summary.csv"),
         "--sensitivity-output", str(output / "case3_projection_sensitivity_summary.csv"),
         "--states-output", str(output / "case3_consistency_states.npz"),
         "--trajectories-output", str(output / "case3_projection_trajectories.csv")],
        [sys.executable, "-B", str(PACKAGE / "scripts/build_tables.py"), "--results-dir", str(output)],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    result = checks.independent_checks(output) if args.build_reference else checks.verify(output)
    import numpy, scipy, h5py
    report = {"release_id": config["release_id"], "python": platform.python_version(),
              "platform": platform.platform(), "numpy": numpy.__version__,
              "scipy": scipy.__version__, "h5py": h5py.__version__,
              "elapsed_seconds": time.perf_counter() - start, "verification": result}
    if not args.build_reference:
        (output / "reproduction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
