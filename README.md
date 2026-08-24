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

A minimal CPU-capable environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "pennylane==0.45.0" numpy scipy matplotlib h5py
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The supplied reproduction commands request `lightning.gpu`. Install a
CUDA-compatible PennyLane Lightning GPU package for GPU execution. For a CPU
run, replace `--device lightning.gpu` with `--device lightning.qubit`. The Case
1 and Case 3 scripts can also fall back to `lightning.qubit` or
`default.qubit`; the supplied Case 2 command disables fallback with
`--no-device-fallback`.

## Running the cases

Clone the repository and run each command from its case directory so that
outputs remain grouped with the corresponding program.

### Case 1: static MD–FE coupling

This case assembles the weighted atomistic–continuum stiffness system and solves
the reduced static problem with an analytic PennyLane VQLS energy functional.

```bash
cd case1
python pennylane_vqls_shot_free.py \
  --md-atoms 850 \
  --fe-elems 177 \
  --u-right 87 \
  --device lightning.gpu \
  --energy-estimator pauli_hamiltonian \
  --optimizer scipy-lbfgsb \
  --initial-state-guess index_ramp \
  --maxiter 100 \
  --output-prefix f1
```

Expected outputs:

- `f1.npz`: numerical solution and diagnostics
- `f1_scatter.png`: reference-versus-VQLS displacement comparison

Use `--no-plot` to skip figure generation. The alternative
`probability_laplacian` estimator is a lower-overhead analytic reference for
this positive-amplitude 1D mechanics problem.

### Case 2: dynamic MD–FE coupling

The dynamic state `y = [u; v]` satisfies `dy/dt = Ay`. The recommended driver
compresses each BCOW history system to its segment-boundary variables and uses
short restarted time windows to keep individual VQLS systems tractable.

```bash
cd case2
python 1Ddynamic_BCOW_VQLS_boundary_reduced_pennylane.py \
  --no-pad --no-device-fallback \
  --target direct-c \
  --device lightning.gpu \
  --energy-estimator pauli_hamiltonian \
  --compare-direct \
  --n1 5 --n2 2 --h-factor 5 --pulse-atoms 3 \
  --t 5 \
  --restart-window 0.1 \
  --segments 1 \
  --taylor-order 4 \
  --coefficient-tol 1e-10 \
  --max-vqls-qubits 12 \
  --ansatz real_ry_identity \
  --layers 8 \
  --init-scale 0.0 \
  --initial-state-guess rhs \
  --theta-warm-start previous \
  --theta-jitter 0.01 \
  --optimizer scipy-lbfgsb \
  --maxiter 300 \
  --residual-target 1e-2 \
  --window-retries 2 \
  --progress-interval 25 \
  --exact-every 10 \
  --output-prefix boundary_reduced
```

Expected outputs include:

- `boundary_reduced.npz`: solution arrays and window diagnostics
- `boundary_reduced.json`: run metadata
- `boundary_reduced_window_summary.csv`: per-window convergence metrics
- `boundary_reduced_displacement.csv`: exact and VQLS displacements when the
  exact comparison is enabled

### Case 3: Carleman-lifted Lennard-Jones chain

This case applies order-2 Carleman lifting to a three-atom 1D Lennard-Jones
chain. Every backward-Euler step is solved with a direct signed-amplitude VQLS;
`--classical-check-interval 1` evaluates a dense classical reference at every
step for diagnostics.

```bash
cd case3
python case3_carleman_pennylane_vqls_direct_signed.py \
  --n-steps 150 \
  --device lightning.gpu \
  --energy-estimator pauli_hamiltonian \
  --optimizer scipy-lbfgsb \
  --initial-state-guess uniform \
  --maxiter 300 \
  --classical-check-interval 1 \
  --output-prefix 150steps_result
```

Expected outputs:

- `150steps_result.h5`: trajectory data
- `150steps_result_diagnostics.npz`: matrices, final state, optimizer results,
  residuals, and classical comparisons

Add `--write-xyz` to export trajectory frames as XYZ files.

## Reproducibility notes

- The canonical commands are also stored in `runcommand_case1.txt`,
  `runcommand_case2.txt`, and `runcommand_case3.txt`.
- Optimizer seeds are controlled by `--rng-seed` (default: `7` for Cases 1 and
  3; `1234` for the boundary-reduced Case 2 driver).
- The SciPy optimizers are deterministic for a fixed initial state and software
  environment. GPU and library differences can still introduce small numerical
  variations.
- Output files are written relative to the current working directory unless an
  explicit path is included in `--output-prefix`.

## Computational scope

The VQLS objectives use analytic PennyLane expectation values and dense Pauli
decompositions. Pauli expansion grows exponentially with the number of qubits,
so these implementations are suited to small validation systems and selected
reproduction cases. Larger BCOW or Carleman systems require scalable
block-encoding, local-cost, QLSA, or QSVT formulations rather than dense
classical Pauli expansion.
