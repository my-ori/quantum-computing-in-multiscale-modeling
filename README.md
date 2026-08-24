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
└── case3/
    ├── case3_carleman_pennylane_vqls_direct_signed.py
    └── runcommand_case3.txt
```

The additional Case 2 programs provide the classical BCOW construction, a
single-window VQLS implementation, and a full restarted VQLS implementation.
Keep all four Case 2 Python files together because the VQLS drivers load sibling
modules at runtime.

## Environment

The target environment used by the scripts is:

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
