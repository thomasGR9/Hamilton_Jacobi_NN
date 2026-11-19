# Hamilton–Jacobi Networks (HJN)

Code for **"Hamilton–Jacobi Networks: Unifying Conserved-Quantity Discovery and Symplectic Trajectory Prediction"**
(T. N. Saltzis, A. Tefas).





## 1. Repository layout

| File | What it contains |
|---|---|
| `modules.py` | **Original n = 1 implementation.** HJL layers (`Step_1`, `Step_2`, `CombinedHamiltonianLayer`, `SimpleStackedHamiltonianNetwork`), their inverses, n = 1 data loader, HSIC and prediction losses, `train_model`, inference protocol, all n = 1 evaluation functions, the **local Lyapunov-time estimator** (used for every task) and the n = 1 constancy experiment. |
| `modules_general.py` | **General-n implementation** (any number of degrees of freedom). Same architecture with a trailing DOF axis (`*_general`), segment data loader, masked loss, `train_model_general`, general-n inference, evaluation and constancy functions. |
| `train_model_low_samples.py` | Training script, Task 1 (mass–spring), **seed 42**, original n = 1 code. |
| `train_model_low_samples_pendulum.py` | Training script, Task 2 (ideal pendulum), **seed 42**, original n = 1 code. |
| `train_model_mass_spring_generalized.py` | Training script, Task 1, **seeds 52 and 62**, general code. |
| `train_model_ideal_pendulum_generalized.py` | Training script, Task 2, **seeds 52 and 62**, general code. |
| `train_model_real_pendulum_generalized.py` | Training script, Task 3 (real pendulum), general code. |
| `train_model_HH.py` | Training script, Task 4 (Hénon–Heiles), all seeds, the λ_HSIC = 0 run and the T_max ablation, general code. |
| `data_generator_and_sorting.ipynb` | Data generation, train/val/test splits, segmentation, T_max estimation for Tasks 3–4, Benettin check, HSIC-sweep pool, T_max-ablation segmentations. |
| `training_results_low_samples.ipynb` | All Task 1 evaluations (all seeds, λ_HSIC = 0). Also the Task 1 T_max estimate. |
| `training_results_low_samples_pendulum.ipynb` | All Task 2 evaluations (all seeds, λ_HSIC = 0). Also the Task 2 T_max estimate. |
| `training_results_real_pendulum.ipynb` | Task 3 evaluations. |
| `training_results_HH.ipynb` | All Task 4 evaluations, HSIC frame sweep, λ_HSIC = 0 run, T_max ablation evaluation. |
| `final_plots.ipynb` | Reads the saved result pickles and prints every table / draws every figure of the paper. |
| `run_benchmark.sh`, `run_benchmark_general.sh`, `time_one_sweep_pytorch_general.py`, `hjn_config_general.py` | Inference runtime / GPU-memory benchmark (Appendix "Inference runtime and GPU memory"). |
| `time_one_sweep_pytorch.py`, `hjn_config.py`, `aggregate_sweeps.py` | n = 1 benchmark script and config, and the aggregator called after every benchmark. Needed for the runtime table. |





## 2. Environment

Results in the paper were produced with Python 3 + PyTorch 2.5.1 + CUDA 12.4 on one NVIDIA RTX 3060 (12 GB).

* HJN code: `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib` (`pyarrow` is optional; it
  is only used to log per-segment progress).
* Hénon–Heiles data generation and the Benettin check: `heyoka` (used in a separate conda environment).
  This is why Task 4 data are stored as *portable* pickles of plain numpy arrays, which load under any pandas version.
* Reference model (Noether's Razor): JAX/jaxlib 0.10.0.
* * **Environments:** the exact software versions used are provided as exported environment files. Two separate environments were used:
  * HJN code (this repository): Python 3.11.13, PyTorch 2.5.1. Files: `environment_hjn.yml` (full conda export) and `requirements_hjn.txt` (pip package list).
  * Noether's Razor reference: Python 3.11.15, JAX 0.10.0 with CUDA 12 support. Files: `environment_noethers_razor.yml` and `requirements_noethers_razor.txt`.




## 3. Data, checkpoints and saved results

All datasets, trained checkpoints and saved results are archived on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24377012.svg)](https://doi.org/10.5281/zenodo.24377012)

> Saltzis, T. N. (2026). *Data, checkpoints and saved results for "Hamilton–Jacobi Networks: Unifying Conserved-Quantity Discovery and Symplectic Trajectory Prediction"* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.24377013


`saved_results_noethers_razor.zip` contains a top-level `noethers-razor/` folder; unzip it anywhere (e.g. next to this repository).
`final_plots.ipynb` reads both result sets: set `HJN_ROOT` (this repository) and `BASE_ROOT` (the unzipped `noethers-razor/` folder) in its first cell.

| Artifact | Archive | Files / directories |
|---|---|---|
| Task 1 data | `data_task1.zip` | `all_dataframes_low_samples.pkl` (pool → train/val), `all_dataframes_mass_spring_portable.pkl` (same train/val as segments, for the general code), `all_dataframes_low_samples_multiple_periods.pkl` (Test A), `df_high_energy_low_samples.pkl` (Test B) |
| Task 2 data | `data_task2.zip` | `all_dataframes_low_samples_pendulum.pkl`, `all_dataframes_pendulum_portable.pkl`, `all_dataframes_low_samples_multiple_periods_pendulum.pkl` (Test A), `df_high_energy_low_samples_pendulum.pkl` (Test B) |
| Task 3 data | not archived | Not redistributed. Download `invar_datasets.zip` from the *Science* supplement of Schmidt & Lipson (2009) into the repository root; data notebook cells 65–82 rebuild `all_dataframes_real_pendulum.pkl` and `all_dataframes_real_pendulum_generalized.pkl` deterministically. |
| Task 4 data | `data_task4.zip` | `all_dataframes_HH.pkl` / `all_dataframes_HH_portable.pkl` (pool, splits, segments), `all_dataframes_HH_portable_high_energy.pkl` (Test B), `dataframes_HH_HSIC_TEST_portable.pkl` (HSIC-sweep pool), `all_dataframes_HH_portable_Tmax_{2.14075,4.2815,17.126,50}.pkl` (T_max ablation) |
| Checkpoints (`best_model.pt` + `loss_scales.pkl`) | `checkpoints.zip` | Task 1: `save_directory_noisy_full_pred_low_samples_4` (seed 42), `save_directory_mass_spring_seed_{52,62}`, `save_directory_mass_spring_NO_HSIC_3` (λ_HSIC = 0). Task 2: `save_directory_noisy_full_pred_low_samples_pendulum_4` (seed 42), `save_directory_ideal_pendulum_seed_{52,62}`, `save_directory_ideal_pendulum_NO_HSIC_1` (λ_HSIC = 0). Task 3: `save_directory_real_pendulum_generalized_1`. Task 4: `save_directory_HH_general_1` (seed 42), `save_directory_HH_seed_{52,62}`, `save_directory_HH_NO_HSIC_1` (λ_HSIC = 0), `save_directory_HH_Tmax_{2.14075,4.2815,17.126,50}_1` (T_max ablation) |
| HJN saved results read by `final_plots.ipynb` | `saved_results_hjn.zip` | `prediction_results_all_trajectories/`, `prediction_results_single_trajectory/`, `constancy_results/`, `conserved_quantities_results/`, `variance_results/`, `seed_results/`, `no_HSIC_ablation_results/`, `HH_Tmax_ablation_results/`, `benchmark_outputs/`, `frame_sweep_res.pkl` |
| Noether's Razor baseline results read by `final_plots.ipynb` | `saved_results_noethers_razor.zip` | `noethers-razor/prediction_results_all_trajectories/`, `noethers-razor/prediction_results_single_trajectory/`, `noethers-razor/variance_results/`, `noethers-razor/benchmark_outputs/` |

## 4. Method (Section "Proposed Method", Algorithms 1–2)

| Paper element | n = 1 code (`modules.py`) | General code (`modules_general.py`) |
|---|---|---|
| HJL Step 1, Eqs. (hjl_s1q)–(hjl_s1p) | `Step_1` | `Step_1_general` |
| HJL Step 2, Eqs. (hjl_s2p)–(hjl_s2q) | `Step_2` | `Step_2_general` |
| One HJL = Step 1 ∘ Step 2 | `CombinedHamiltonianLayer` | `CombinedHamiltonianLayer_general` |
| Mapping network 𝓜 (L stacked HJLs, linear output only in the last HJL) | `SimpleStackedHamiltonianNetwork` | `SimpleStackedHamiltonianNetwork_general` |
| Inverse network 𝓜⁻¹ (closed-form, shared weights) | `ReverseStep1/2`, `ReverseCombinedHamiltonianLayer`, `InverseStackedHamiltonianNetwork` | `ReverseStep1/2_general`, `ReverseCombinedHamiltonianLayer_general`, `InverseStackedHamiltonianNetwork_general` |
| Bounded reparameterisation of `a` (App. "HJN Architecture and Initialisation") | `Step_1.a` / `Step_2.a`, `solve_a_raw`, `init_step_params` | `solve_a_raw`, `a_raw_std_from_induced`, `init_step_params_general` |
| Prediction loss L_pred (masked MAE) | `prediction_loss(..., loss_type="mae")` | `prediction_loss_masked` |
| Independence loss L_HSIC (unbiased, RBF, median bandwidth, zeroed diagonal, clamped ≥ 0) | `hsic_loss`, `_compute_rbf_kernel` | `hsic_loss_general` |
| H_max = max(HSIC(s, s), 0.05), s = [1..N], per batch | inside `train_model` (cached per N) | `hsic_max_for_N` |
| Total loss, Eq. (loss_total) | `compute_total_loss` (`hsic_loss_max_want` = λ_HSIC) | `compute_total_loss_general` |
| Alg. 1, Phase I (calibration of L̄_pred, median if \|μ/s\| < 2) | `calculate_losses_scale_on_untrained` | `calibrate_scales` |
| Alg. 1, Phase II (one observation per segment, decode every sample) | `SimpleHarmonicDataLoader` + `train_model` | `SegmentDataLoader_general` + `training_step` + `train_model_general` |
| Validation criterion (encode first point of each segment, predict the segment) | `validation_forward_pass` inside `train_model` | `make_validation_batch`, `validation_step` |
| Segmentation into windows of length ≤ T_max, segment-local time | one segment per trajectory (`build_seg_frames` in the data notebook) | `create_segments` (data notebook and `modules_general.py`), `create_segments_1d` (Task 3) |
| Alg. 2, autoregressive hand-off inference | `Autoregressive_Handoff_Inference_Protocol`, `Batched_Autoregressive_Handoff_Inference_Protocol` | `..._general` versions |
| T_max rule, Eq. (tmax_rule), and local Lyapunov-time estimator (App. "Specification of the local Lyapunov-time estimator") | `local_lyapunov_times`, `ensemble_scaling_slope`, `_ensemble_slopes`, `_half_max_x`, `theiler_window_from_autocorr`, `canonical_pair_scaling`, `eps_grid_from_pairwise`, `summarize_task`, `row_to_latex` | — (the n = 1 module handles any n) |

## 5. Datasets and training runs

### 5.1 Datasets (Table "Task specifications", App. "Initial-condition sampling")

Cell numbers below are 0-based, counting markdown cells, as of commit `4a1799d`.

| Task | Notebook cells (`data_generator_and_sorting.ipynb`) |
|---|---|
| Task 1 pool (74 trajectories, 32 samples), stratified split → 24 train / 25 val | cells 6–20; segments for the general code: cells 21–22 |
| Task 1 Test A (separate set: 24 trajectories × 97 samples, E ∈ [0.2, 1]) | cells 23–32 |
| Task 1 Test B (high-energy set, 50 % of it used as test) | cells 33–43 |
| Task 2 pool (75 trajectories, 30 samples), DOP853 `atol=1e-12, rtol=1e-9, max_step=0.1` | cells 44–56 |
| Task 2 Test A (separate set: 20 × 144) / Test B (13 of 26 high-energy trajectories) | cells 57–59 / 60–64 |
| Task 3 (first 444 samples train, last 112 test = validation) | cells 65–70, T_max estimate 75–78, segmentation 79–80, save 82 |
| Task 4 pool (80 trajectories, heyoka, seed 42) → 48 / 16 / 16 | cells 84–89; T_max estimate cell 91; segmentation 93–96 |
| Task 4 Test B (16 trajectories, E ∈ [0.13, 1/6)) | cells 97–100 |
| Benettin maximal Lyapunov exponents (App. "Application to the training data") | cell 103 |
| HSIC frame-sweep pool (80 trajectories on [0, 3 T_max]) | cells 104–107 |
| T_max-ablation segmentations, epoch budgets and scheduler patience (Tables "T_max ablation: segmentation…") | cells 108–113 |

**T_max estimates (Table "Local Lyapunov-time estimation"):**
Task 1: `training_results_low_samples.ipynb` cell 7; Task 2: `training_results_low_samples_pendulum.ipynb`
cell 7; Task 3: data notebook cells 75–77; Task 4: data notebook cell 91.

**Noise selection (App. "Noise selection"):** per-channel variances and SNRs are printed by
`training_results_low_samples.ipynb` cell 6, `training_results_low_samples_pendulum.ipynb` cell 6 and
`training_results_HH.ipynb` cell 7.

### 5.2 Training runs (Table "Per-task training configuration")

Common to all runs: L = 10, hidden widths [33, 60, 32], GELU, orthogonal init, AdamW (β = (0.9, 0.999),
weight decay on MLP weights only), `ReduceLROnPlateau(mode="min", factor=0.5)` stepped once per epoch on the
validation criterion, λ_pred = 2.

The values committed in the training scripts are those of the **last** run launched with each script. The values
below are read from each run's own record: the optimiser and scheduler state in `best_model.pt`, `loss_scales.pkl`,
`run_*.json`, and the learning rate logged in `epoch_*/epoch_metrics.json`. To reproduce a run, copy them into the
"RUN-SPECIFIC KNOBS" block of the script and run `python <script>` from the repository root.

| Run | Code | σ² | λ_HSIC | λ_WD | Initial LR | Plateau patience / threshold | Grad clip | Epochs run | Selected epoch | L̄_pred | Initial std of `a` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1, seed 42 | n = 1 | 0.1 | 0.1 | 1e-4 | 3e-3 | 50 / 1e-4 rel. | 60 | 543 | 507 | 1.7133 | 10 % |
| T1, seed 52 | general | 0.1 | 0.1 | 1e-4 | 3e-3 | 50 / 1e-4 rel. | 60 | 507 | 501 | 1.8177 | 10 % |
| T1, seed 62 | general | 0.1 | 0.1 | 1e-4 | 3e-3 | 50 / 1e-4 rel. | 60 | 507 | 496 | 1.7845 | 10 % |
| T1, λ_HSIC = 0, seed 42 | general | 0.1 | 0 | 1e-4 | 3e-3 | 50 / 1e-4 rel. | 60 | 600 | 597 | 1.8190 | 10 % |
| T2, seed 42 | n = 1 | 0.1 | 0.35 | 1e-6 | 3e-3 | 100 / 1e-4 rel. | 60 | 1600 | 1157 | 2.4217 | 10 % |
| T2, seed 52 | general | 0.1 | 0.35 | 1e-6 | 3e-3 | 100 / 1e-4 rel. | 60 | 1157 | 1105 | 2.4920 | 10 % |
| T2, seed 62 | general | 0.1 | 0.35 | 1e-6 | 3e-3 | 100 / 1e-4 rel. | 60 | 1157 | 1121 | 2.5271 | 10 % |
| T2, λ_HSIC = 0, seed 42 | general | 0.1 | 0 | 1e-6 | 3e-3 | 100 / 1e-4 rel. | 60 | 1200 | 1084 | 2.4813 | 10 % |
| T3 | general, seed 42 | 0 | 0.05 | 1e-4 | 1e-4 | 200 / 1e-3 abs. | 60 | 2769 | 2202 | 1.8850 | 10 % |
| T4, seed 42 | general | 0.0045 | 0.35 | 1e-6 | 3e-3 | 200 / 1e-3 abs. | 60 | 2911 | 2111 | 0.5308 | 10 % |
| T4, seed 52 | general | 0.0045 | 0.35 | 1e-6 | 3e-3 | 200 / 1e-3 abs. | 60 | 2111 | 2110 | 0.5022 | 10 % |
| T4, seed 62 | general | 0.0045 | 0.35 | 1e-6 | 3e-3 | 200 / 1e-3 abs. | 60 | 2111 | 248 | 0.5120 | 10 % |
| T4, λ_HSIC = 0, seed 42 | general | 0.0045 | 0 | 1e-6 | 3e-3 | 200 / 1e-3 abs. | 60 | 2200 | 1994 | 0.5308 | 10 % |
| T4, T_max = 2.14075, seed 42 | general | 0.0045 | 0.35 | 1e-6 | 3e-3 | 773 / 1e-3 abs. | 60 | 8163 | 3697 | 0.2839 | 10 % |
| T4, T_max = 4.2815, seed 42 | general | 0.0045 | 0.35 | 1e-6 | 3e-3 | 400 / 1e-3 abs. | 60 | 4222 | 2265 | 0.4498 | 10 % |
| T4, T_max = 17.126, seed 42 | general | 0.0045 | 0.35 | 1e-6 | 3e-3 | 101 / 1e-3 abs. | 60 | 1065 | 33 | 0.5814 | 10 % |
| T4, T_max = 50, seed 42 | general | 0.0045 | 0.35 | 1e-6 | 3e-3 | 35 / 1e-3 abs. | 60 | 367 | 356 | 0.5658 | 10 % |

Scripts and directories:

| Run | Script | `SAVE_DIR` |
|---|---|---|
| T1 seed 42 | `train_model_low_samples.py` | `save_directory_noisy_full_pred_low_samples_4` |
| T1 seeds 52 / 62, λ_HSIC = 0 | `train_model_mass_spring_generalized.py` | `save_directory_mass_spring_seed_{52,62}`, `save_directory_mass_spring_NO_HSIC_3` |
| T2 seed 42 | `train_model_low_samples_pendulum.py` | `save_directory_noisy_full_pred_low_samples_pendulum_4` |
| T2 seeds 52 / 62, λ_HSIC = 0 | `train_model_ideal_pendulum_generalized.py` | `save_directory_ideal_pendulum_seed_{52,62}`, `save_directory_ideal_pendulum_NO_HSIC_1` |
| T3 | `train_model_real_pendulum_generalized.py` | `save_directory_real_pendulum_generalized_1` |
| T4 (all runs) | `train_model_HH.py` (`DATA_PICKLE=all_dataframes_HH_portable_Tmax_<T>.pkl` for the ablation) | `save_directory_HH_general_1`, `save_directory_HH_seed_{52,62}`, `save_directory_HH_NO_HSIC_1`, `save_directory_HH_Tmax_<T>_1` |


## 6. Experiments (Section "Experiments") and appendix

The evaluation notebooks write result pickles; `final_plots.ipynb` reads them and prints the tables and figures.
In each evaluation notebook the seed-42 model comes first, followed by sections headed **SEED 52**, **SEED 62** and
**NO HSIC ABLATION** that repeat the same calls for the other checkpoints.

| Paper element | Produced by (evaluation) | Printed / drawn by |
|---|---|---|
| Tables "Within-segment constancy" (D₀, δ_rms) and the round-trip bound 5.6·10⁻⁶ | `run_constancy_experiment_1d` (T1: `training_results_low_samples.ipynb` cell 39; T2: pendulum notebook cell 38) and `run_constancy_experiment` (T4: `training_results_HH.ipynb` cell 40) → `constancy_results/`, `seed_results/constancy_results_seed_*` | `final_plots.ipynb` cell 28 |
| Tables "Energy recovery", "Phase recovery", "Time-shift test" (energy, phase), "Anharmonic frequency curve" | recovery/time-shift code defined inline: T1 cell 46, T2 cell 45, T4 cells 46–51 → `conserved_quantities_results/`, `seed_results/conserved_quantities_results_seed_*` | `final_plots.ipynb` cell 31 (also the positive-definiteness / Ê_min check) |
| Figure "Energy and phase signatures" (`energy_recovery_time_shift_both_tasks.pdf`) | same pickles (seed 42) | `final_plots.ipynb` cell 32 |
| Table "Aggregate prediction accuracy", Figures "Per-task accuracy structure" (`exp1_main_multiple_periods.pdf`, `exp1_appendix_high_energy.pdf`) | `test_model_all_observations_all_trajectories` (T1, T2) / `..._general` (T4) → `prediction_results_all_trajectories/`, `seed_results/prediction_results_all_trajectories_seed_*` | `final_plots.ipynb` cells 3–7 (BCa bootstrap, 10 000 resamples, RNG seed 0) |
| Table "Single-trajectory inference diagnostics", Figures `exp2_main.pdf`, `exp2_appendix.pdf` | `test_model_varying_observation_single_trajectory` (T1, T2) / `..._general` (T3, T4) on the representative trajectory (`median_representative_id` of the aggregate run) → `prediction_results_single_trajectory/` | `final_plots.ipynb` cells 8–13 |
| Table "Per-point symplectic residual" | `test_canonical_transformation_on_trajectory` (T1 cell 14, T2 cell 13) / `..._general` (T3 cell 10, T4 cell 13) | printed in those cells |
| Table "Local Lyapunov-time estimation" | see §5.1 | printed as LaTeX rows by `row_to_latex` |
| HSIC motivation: Table "Representative frames", Figure `hsic_frame_sweep.pdf` | `training_results_HH.ipynb` cells 78–88 (`run_frame_sweep`) → `frame_sweep_res.pkl` | `final_plots.ipynb` cells 33–37 (saves `exp4_frame_sweep.pdf`) |
| Table "Penalised and unpenalised models at S = id" | `training_results_HH.ipynb` cell 97 (`frameless_row`) for the λ_HSIC = 0 model; the λ_HSIC > 0 row is the S = id row of the sweep | printed in that cell |
| Tables "Results without the independence term" (constancy, recovery, prediction) | NO HSIC ABLATION sections of the three evaluation notebooks → `no_HSIC_ablation_results/` | `final_plots.ipynb` cells 39–41 (prediction), 47 (constancy), 48 (recovery) |
| Tables "T_max ablation" (setup and results) | data notebook cells 108–113; `training_results_HH.ipynb` cells 99–100 → `HH_Tmax_ablation_results/` | `training_results_HH.ipynb` cell 101 prints the means and NFE; its confidence intervals come from an unseeded bootstrap. Recompute them with the seeded BCa of `final_plots.ipynb` (RNG seed 0), as for Table "Aggregate prediction accuracy". |
| Table "Inference runtime and GPU memory" | `run_benchmark.sh` (T1, T2) / `run_benchmark_general.sh` (T4): 1 warm-up + 20 timed sweeps, each in its own process, saved to `benchmark_outputs/hjn_task{1,2,4}_sweeps[_high_energy]/sweep_*.pkl`, then `aggregate_sweeps.py` → `benchmark_outputs/hjn_task*_{multiple_periods_final,high_energy}.pkl`  | `final_plots.ipynb` cell 6 |
| Parameter counts (82 560 / 83 260) | `count_parameters`, printed in each evaluation notebook | — |




## 7. Code not used in the paper

**Notebook cells whose output is not in the paper** (0-based cell numbers; the paper's figures and tables all come
from `final_plots.ipynb` and the cells listed in §6):

| Notebook | Cells | What they do |
|---|---|---|
| `training_results_low_samples.ipynb` (T1) | 21–24, 37–38, 61–62, 72–73, 83–84 | Variance experiment: `test_model_variance_with_varying_observed_points` + `plot_variance_results` |
| | 10 ||
| | 20, 36, 30, 45 | Notebook previews of the accuracy and single-trajectory results (`plot_summary_statistics`, `plot_varying_observation_results`); the paper versions are drawn by `final_plots.ipynb` |
| | 47–50, 54–55 | Earlier quadratic-minimum check and recovery figures with hard-coded, rounded coefficients; superseded by `final_plots.ipynb` cells 31–32 |
| `training_results_low_samples_pendulum.ipynb` (T2) | 20–23, 36–37, 54–55, 65–66, 76–77 | Variance experiment (as above) |
| | 10, 19, 35, 29, 44 | |
| | 48 (figure part) | Earlier per-task pendulum time-shift figures; the paper figure is `final_plots.ipynb` cell 32 |
| `training_results_HH.ipynb` (T4) | 21–24, 38–39, 60–61, 71–72, 92–93 | Variance experiment (`..._general` versions) |
| | 8, 9, 19, 36, 29, 44 | per-segment training progress, notebook previews |
| | 45, 53 | Column statistics; "normalised latent motion" check |
| `training_results_real_pendulum.ipynb` (T3) | 6, 16 | notebook preview |
| `final_plots.ipynb` | 14–26, 42–46 | Variance experiment figures and tables ("Experiment 3", `exp3_main.pdf`), main and λ_HSIC = 0 |

The variance experiment measures the within-segment variance of the *raw* latents as observed points are added, for
Tasks 1, 2 and 4. The paper's constancy experiment (D₀, δ_rms) is a different computation
(`run_constancy_experiment*`). **TODO:** keep these cells (clearly labelled) or remove them.

**Functions not used for the paper:** `time_inference_gpu*` (in-process timing; the paper uses the multi-process
benchmark), `train_model_real_pendulum`, `resume_training_from_checkpoint_real_pendulum`,
`prepare_prediction_inputs_for_real_pendulum_2`, `generate_prediction_labels_for_real_pendulum_2` (earlier n = 1 Task 3
pipeline, also data notebook cells 71–72), `analyze_mapping_net`, `find_valid_combinations`,
`get_trajectory_ids_by_energies`, and the diagnostics `visualize_epoch_metrics`, `load_segment_progress`,
`plot_segment_progress`, `summarise_segment_progress`.



## 8. License

The code in this repository is released under the MIT License (see `LICENSE`).
The datasets, checkpoints and saved results on Zenodo (https://doi.org/10.5281/zenodo.24377012) are released under Creative Commons Attribution 4.0 International (CC BY 4.0).
The Task 3 real-pendulum data are from Schmidt & Lipson (2009) and are not redistributed here; they remain subject to their original terms.


