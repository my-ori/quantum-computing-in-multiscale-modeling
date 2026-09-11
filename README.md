# Quantum Computing in Multiscale Modeling

Reference implementations for three multiscale-mechanics case studies using
shot-free [PennyLane](https://pennylane.ai/) simulations of variational quantum
linear solvers (VQLS). The examples cover static and dynamic molecular
dynamics–finite element (MD–FE) coupling and a nonlinear Lennard-Jones chain
treated with Carleman linearization.

The quantum circuits are evaluated analytically (`shots=None`). These programs
are intended for algorithm development, verification, and reproducibility—not
as evidence of a quantum speedup on classical simulators.

## Case studies

| Case | Mechanical model | Quantum formulation | Main entry point |
| --- | --- | --- | --- |
| 1 | Static 1D MD–FE coupling | Positive-amplitude VQLS for the reduced stiffness system | `case1/pennylane_vqls_shot_free.py` |
| 2 | Dynamic 1D MD–FE coupling | Boundary-reduced, restarted BCOW linear-ODE system with VQLS | `case2/1Ddynamic_BCOW_VQLS_boundary_reduced_pennylane.py` |
| 3 | Three-atom 1D Lennard-Jones chain | Order-2 Carleman lifting, backward Euler, and signed-amplitude VQLS | `case3/case3_carleman_pennylane_vqls_direct_signed.py` |

BCOW refers to the Berry–Childs–Ostrander–Wang linear-system construction for
linear ordinary differential equations.

## Repository layout

```text
.
├── case1/
│   ├── pennylane_vqls_shot_free.py
│   └── runcommand_case1.txt
├── case2/
│   ├── 1Ddynamic_BCOW_ODE_solver.py
│   ├── 1Ddynamic_BCOW_VQLS_pennylane.py
│   ├── 1Ddynamic_BCOW_VQLS_restarted_pennylane.py
│   ├── 1Ddynamic_BCOW_VQLS_boundary_reduced_pennylane.py
│   └── runcommand_case2.txt
├── case3/
│   ├── case3_carleman_pennylane_vqls_direct_signed.py
│   └── runcommand_case3.txt
├── data_of_figures/
│   └── figure*.csv
└── reproducibility/
    └── case3/
        ├── reproduce.py
        ├── inputs/
        ├── scripts/
        ├── results/
        └── provenance/
```

The additional Case 2 programs provide the classical BCOW construction, a
single-window VQLS implementation, and a full restarted VQLS implementation.
Keep all four Case 2 Python files together because the VQLS drivers load sibling
modules at runtime.

## Nonlinear validation for the second revision

The [Case III validation archive](reproducibility/case3/README.md) contains the exact retained inputs, full Lennard-Jones and quadratic-force DOP853 integrations, continuous and backward-Euler Carleman trajectories, and the retained VQLS trajectory. It also includes complete projected and unprojected classical states, re-lifted sensitivity trajectories, and the scripts and machine-readable data needed to regenerate Tables 7 and 8.

From the repository root, run:

```bash
python -m pip install -r reproducibility/case3/requirements.txt
python -B reproducibility/case3/reproduce.py
```

This command verifies the SHA-256 manifest, regenerates the validation results and table fragments, and compares all archived result files. Generated files and a new verification report are written to `reproducibility/case3/reproduced/`. Only NumPy, SciPy, and h5py are required after Python is installed; the command reads the retained VQLS results without repeating the optimizations.

The archive uses the original PennyLane 0.42.3 `lightning.qubit` accuracy artifact. The subsequent PennyLane 0.45.0 GPU run supplied resource measurements. The archive README explains these sources, the 0-0.300 ps reporting interval, and the common backward-difference velocity reconstruction. Its table builder also generates the Figure 6 comparison and full-precision Figure 7 costs. Existing public series are in `data_of_figures/figure6a_6b_data.csv` and `data_of_figures/figure7_data.csv`.

For the manuscript citation, use the immutable repository commit that contains this validation archive. The earlier solver snapshot listed in the provenance file is a separate source record.

## Environment

The target environment used by the original VQLS solver scripts is:

- Python 3.12.13
- PennyLane 0.45.0
- PennyLane Lightning GPU 0.45.0 for `lightning.gpu`
- NumPy, SciPy, Matplotlib, and h5py

The supplied reproduction commands request `lightning.gpu`. Install a
CUDA-compatible PennyLane Lightning GPU package for GPU execution. For a CPU
run, replace `--device lightning.gpu` with `--device lightning.qubit`. The Case
1 and Case 3 scripts can also fall back to `lightning.qubit` or
`default.qubit`; the supplied Case 2 command disables fallback with
`--no-device-fallback`.

## Running the cases

Clone the repository and run each command from its case directory so that
outputs remain grouped with the corresponding program.

## Computational scope

The VQLS objectives use analytic PennyLane expectation values and dense Pauli
decompositions. Pauli expansion grows exponentially with the number of qubits,
so these implementations are suited to small validation systems and selected
reproduction cases. Larger BCOW or Carleman systems require scalable
block-encoding, local-cost, QLSA, or QSVT formulations rather than dense
classical Pauli expansion.
