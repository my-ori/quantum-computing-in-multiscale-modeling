# Reproduce the nonlinear validation in Tables 7 and 8

This archive reproduces the nonlinear validation of the three-atom Lennard–Jones benchmark, including the velocity correction in the second revision. It contains the exact retained VQLS input files, runnable validation scripts, all five comparison trajectories, complete classical Carleman states, projected and unprojected sensitivity trajectories, and the code and numerical data for Tables 7–8. The VQLS optimizations do not need to be repeated.

The release identifier is `case3-validation-second-revision-20260910`. `manifest_sha256.json` identifies the exact archived files. `provenance/source_provenance.json` identifies the earlier public repository snapshot separately from this validation release. For an immutable GitHub citation, use the commit containing this archive rather than the moving `main` branch.

## Run the complete reproduction

Use Python 3.12. The validation was executed with Python 3.12.13, NumPy 2.4.6, SciPy 1.17.1, and h5py 3.16.0. From the repository root:

```bash
python -m pip install -r reproducibility/case3/requirements.txt
python -B reproducibility/case3/reproduce.py
```

The command verifies the archive's SHA-256 hashes, reruns both classical nonlinear integrations and both Carleman propagations, computes the projection-sensitivity trajectories, rebuilds Tables 7–8, and compares the regenerated files with `results/`. It also checks the physical matrix construction, backward-Euler recurrence, exact post-projection lift, and agreement with the released Figure 6/7 series. It writes new files and `reproduction_report.json` to `reproduced/`, leaving `results/` intact.

The script resolves its inputs relative to its own location and also runs from an unrelated working directory. To choose another output location:

```bash
python -B reproducibility/case3/reproduce.py --output-dir /path/to/check
```

PennyLane, CUDA, a GPU, and network access are not required to run the validation after its three Python dependencies are installed. `--build-reference` is a maintainer option for generating a new reference set; omit it for verification against this release.

## Files that support the manuscript

| Result | Runnable calculation | Archived data |
|---|---|---|
| Full Lennard–Jones reference | `scripts/case3_error_decomposition.py`, `solve_ivp` with DOP853 | `results/case3_validation_trajectories.csv`; `full_lj_native_q_v` in `results/case3_reference_states.npz` |
| Quadratic-force nonlinear reference | Same script, DOP853 without Carleman truncation | Same files, `quadratic_nonlinear` columns and `quadratic_nonlinear_native_q_v` array |
| Continuous second-order Carleman reference | Same script, matrix-exponential propagation | Same files, `continuous_k2_carleman` columns and `continuous_k2_lifted_state` array |
| Classical backward-Euler Carleman reference | Same script, repeated solution of the fixed linear system | Same files, `backward_euler_k2_carleman` columns and `backward_euler_k2_lifted_state` array |
| Retained VQLS trajectory | Read directly from the archived HDF5 input | Same CSV, `vqls` columns; `vqls_displacement_nm` array |
| Table 7 error decomposition | First script, then `scripts/build_tables.py` | `results/case3_error_decomposition_summary.csv`, time-resolved CSV, and `table7.csv` |
| Table 8 consistency defects | `scripts/case3_carleman_consistency.py`, then table builder | `results/case3_carleman_consistency_by_time.csv`, summary CSV and `table8.csv` |
| Projection sensitivity | Consistency script, backward Euler followed by re-lifting after every step | `results/case3_projection_trajectories.csv`, sensitivity summary CSV, and `case3_consistency_states.npz` |
| Figure 6 displacement and velocity series | Table builder, using the five-level trajectory CSV | Generated as `reproduced/figure6_series.csv` |
| Figure 7 VQLS cost series | Table builder, reading the retained diagnostics | Generated as `reproduced/figure7_cost_full_precision.csv` and `reproduced/figure7_cost_summary.json` |

`scripts/build_tables.py` performs the rounding used in the manuscript and writes both machine-readable CSV tables and LaTeX tabular fragments. The 12 archived CSV/NPZ files are in `results/`; the LaTeX fragments and figure exports are generated in the selected output directory (default `reproduced/`). The fragments contain the numerical table content; manuscript-specific column widths and revision colors are applied in the manuscript source.

## Physical and numerical inputs

`configuration.json` gives the complete settings used by the reproduction command: three atoms with free chain ends, nearest-neighbor Lennard–Jones bonds, equilibrium spacing 1 nm, epsilon 1.65 aJ, mass 1.993 in units of 10^-24 kg, initial displacement (0.01, 0, 0) nm, and zero initial velocity. Sigma is a/2^(1/6). The quadratic force coefficients are c1 = 118.8 aJ/nm² and c2 = -1247.4 aJ/nm³. Both DOP853 integrations use rtol = 10^-12 and atol = 10^-14.

The reported interval consists of 150 steps of 0.002 ps, with 151 frames from 0 to 0.300 ps. The initial lift is Y0 = [z0; kron(z0,z0)], with z = [q0,q1,q2,v0,v1,v2], giving 42 unpadded components. Later states are stored as Y = [z; Y2]; Y2 need not equal kron(z,z) unless explicitly projected. The continuous trajectory is evaluated by repeated application of exp(dt A_C); the verifier also checks its final state against exp(t_final A_C) Y0. The classical backward-Euler update uses M_C = I - dt A_C. The retained NPZ contains both matrices, and the verifier independently rebuilds A_C from the physical parameters.

The VQLS solution direction was embedded in 64 amplitudes on six qubits. Original circuit and optimizer metadata are preserved in `provenance/retained_vqls_metadata.json`; the retained run used an analytic signed controlled-Ry tree, seed 7, SciPy L-BFGS-B, and a maximum of 300 optimizer iterations per step. These settings describe the retained input, not a new optimization performed by this archive.

## Velocity reconstruction and state conventions

All five velocity histories in `case3_validation_trajectories.csv` use the corrected common rule: reconstructed velocity is the prescribed zero initial velocity at frame 0, and [q(n) - q(n-1)] / 0.002 ps at every later frame, including the last. Columns explicitly use the suffix `_reconstructed_velocity_backward_nm_per_ps`. Table 7 maximum and RMS differences include all 3 × 151 = 453 atom–frame samples.

The classical NPZ state arrays additionally preserve native velocity components. `case3_projection_trajectories.csv` labels those components `_native_velocity_nm_per_ps`. Table 8 evaluates the consistency of the native full lifted state; replacing its first-order velocity entries with reconstructed velocities would change the diagnostic and is not done. The sensitivity summary's optional native-velocity metrics are distinct from Table 7's common reconstructed-velocity metrics.

The full classical lifted states are archived before and after every projection. For each state, eta = norm(Y2 - kron(z,z)); the normalized defect divides by max(norm(Y2), norm(kron(z,z)), machine epsilon). In this benchmark the epsilon safeguard is inactive. After projection, Y2 = kron(z,z) exactly, while the first-order z is unchanged. See `DATA_DICTIONARY.md` for array names, shapes, and indexing.

## Retained run and time-interval provenance

`inputs/200steps_result_cpu.h5` and `inputs/200steps_result_cpu_diagnostics.npz` are byte-for-byte copies of the original retained validation inputs. They contain 201 displacement frames through 0.400 ps. Tables 7–8 and Figures 6–7 use the first 151 frames or 150 step records, through 0.300 ps.

The retained run's metadata record **PennyLane 0.42.3 and `lightning.qubit`**. The subsequent PennyLane 0.45.0 `lightning.gpu` run supplied resource-accounting measurements; it is not substituted for this retained accuracy trajectory. The original HDF5 file has no software attributes, so the companion NPZ metadata are the authoritative software record. The retained artifacts do not record the historical solver's Git commit; no historical commit is inferred. An exact snapshot of the public Case III solver is included in `provenance/public_case3_solver.py`, together with its verified public commit and hash, as a separate source record.

Only one complete native VQLS lifted state is retained: `Y_final` at **0.400 ps**. Its consistency is exported to `retained_vqls_terminal_consistency.csv` as an auxiliary terminal-only diagnostic. It is not a Table 8 row, not a 0.300 ps state, and not a per-step VQLS consistency history. The four Table 8 rows are the classical continuous, unprojected, pre-projection, and post-projection comparisons over 0–0.300 ps.

## Corrected validation release

Use the runnable validation scripts in `scripts/`. They apply the common backward-difference velocity reconstruction described above and export complete classical states. The force laws, integration tolerances, Carleman propagation, projection operation, and consistency definitions are those used for the reported validation. The hashes of the earlier first-revision scripts are recorded in `provenance/source_provenance.json`; those historical script files are not included in this reduced release.

The regenerated Table 7 reproduces the corrected manuscript values. Table 8 and the reported projection-sensitivity displacement values are unchanged from the first revision. Verification compares all 12 archived CSV/NPZ result files with rtol = 10^-9 and atol = 10^-12. The table builder generates LaTeX fragments using the manuscript's three-significant-figure formatting. `provenance/verification_report.json` records a completed reproduction of this reduced release.

The two `inputs/published_figure*.csv` files are the historical snapshots used by the verifier. The current repository names for these figures are `data_of_figures/figure6a_6b_data.csv` and `data_of_figures/figure7_data.csv`. The earlier snapshot commit and the repository state inspected before upload are distinguished in `provenance/source_provenance.json`.
