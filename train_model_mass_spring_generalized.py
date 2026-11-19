"""
Training runner for the general-n Hamilton-Jacobi Network (n=2 task).

Uses the paper's shared HJN backbone and initialisation as closely as possible:
  - L = 10 Hamilton-Jacobi Layers, MLP potentials F,G with hidden widths [33,60,32],
    GELU activations, linear output on the FINAL HJL only.
  - Orthogonal (Saxe) weight init.
  - a in [0.5, 2.0], a_k = 0.1, induced-a mean 1.6, std = 10% of mean.
  - gamma ~ N(2.0, 0.04);  c-shifts ~ N(+/-0.1, 0.01).
  - AdamW with weight decay on MLP weights only.
  - Per-batch unbiased HSIC with median-heuristic bandwidth (our verified losses).
  - Prediction loss: masked MAE; total loss normalised by calibrated L_bar_pred and
    per-batch H_max.

Data: expects the pickle with train/val segment dataframes (see LOADING below).
Run:  python train_runner_general.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch

from modules_general import (
    SimpleStackedHamiltonianNetwork_general,
    InverseStackedHamiltonianNetwork_general,
    SegmentDataLoader_general,
    train_model_general,
    resume_training_from_checkpoint_general,
    calibrate_scales,
    make_validation_batch,)



# ===========================================================================
# RUN-SPECIFIC KNOBS (edit these)
# ===========================================================================
RESUME            = False       # False = fresh training; True = resume from checkpoint
RESUME_MODE_A     = True        # (resume only) True = load optimizer/scheduler state (Mode A)
RESUME_NUM_EPOCHS = 2000         # (resume only) additional epochs

DATA_PICKLE   = "all_dataframes_mass_spring_portable.pkl"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 62
SAVE_DIR      = "./save_directory_mass_spring_seed_62"

# columns 
POS_COLS      = ("x",)
MOM_COLS      = ("u",)

# loss / training hyperparameters
ADD_NOISE       = True          # paper adds N(0, sigma^2) noise to encode inputs (Tasks 1-2)
NOISE_VARIANCE  = 0.1          # paper sigma^2 for noisy tasks
LAMBDA_PRED     = 2.0           # paper lambda_pred (Tasks 1-2)
LAMBDA_HSIC     = 0.1          # paper lambda_HSIC (Task 2 value; adjust per task)
WEIGHT_DECAY    = 0.0001         # paper lambda_WD on MLP weights (Task 2 value)
LEARNING_RATE   = 0.003
GRAD_CLIP       = 60.0
NUM_EPOCHS      = 507
NOISE_THRESHOLD = 2.0           # calibration mean/std switch (kept from old code)



# early stopping
EARLY_STOPPING = False
PATIENCE       = 800
MIN_DELTA      = 0
SCHEDULER_THRESHOLD = 0.001 

# scheduler (plateau, as in the old runner)
SCHEDULER_PARAMS = {'mode': 'min', 'factor': 0.5, 'patience': 50, 'verbose': True}

# checkpoint / logging
SAVE_BEST_ONLY   = False
SAVE_FREQ_EPOCHS = 20
LOG_FREQ_BATCHES = 10
VERBOSE          = 1


# ===========================================================================
# MODEL CONFIG (paper backbone + initialisation)
# ===========================================================================
def build_model_config(n):
    return dict(
        n=n,
        n_layers=10,
        hidden_dims=[33, 60, 32],
        n_hidden_layers=None,               # ignored when hidden_dims is a list
        activation='gelu',
        activation_params=None,
        final_activation=None,              # linear output on final HJL
        final_activation_only_on_final_layer=True,
        weight_init='orthogonal',           # Saxe init (paper)
        weight_init_params=None,
        bias_init='zeros',
        bias_init_value=0.0,
        use_bias=True,
        use_layer_norm=False,
        # a reparam: a in [0.5, 2.0], a_k = 0.1, induced-a mean 1.6, std = 10% of mean
        a_eps_min=0.5,
        a_eps_max=2.0,
        a_k=0.1,
        step_1_a_mean_innit=1.6,
        step_2_a_mean_innit=1.6,
        a_std_ratio=0.1,
        # gamma ~ N(2.0, 0.04)  (variance 0.04)
        step_1_gamma_mean_innit=2.0,
        step_2_gamma_mean_innit=2.0,
        gamma_variance=0.04,
        # c1 ~ N(+0.1, 0.01) for step_1 / N(-0.1, 0.01) for step_2 (paper: c_{1,i}, c'_{1,i})
        step_1_c1_mean_innit=0.1,
        step_2_c1_mean_innit=-0.1,
        c1_variance=0.01,
        # c2 ~ N(-0.1, 0.01) for step_1 / N(+0.1, 0.01) for step_2 (paper: c_{2,i}, c'_{2,i})
        step_1_c2_mean_innit=-0.1,
        step_2_c2_mean_innit=0.1,
        c2_variance=0.01,
        seed=SEED,
    )


# ===========================================================================
# DATA LOADING
# ===========================================================================
def load_dataframes(path):
    with open(path, "rb") as f:
        portable = pickle.load(f)
    loaded = {}
    for name, d in portable.items():
        df = pd.DataFrame(d["data"], columns=d["columns"])
        df.index = pd.Index(d["index_values"], name=d["index_name"])
        loaded[name] = df
    return (loaded["train_seg_df"], loaded["train_seg_id_df"],
            loaded["val_seg_df"], loaded["val_seg_id_df"])

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_seg_df, train_seg_id_df, val_seg_df, val_seg_id_df = load_dataframes(DATA_PICKLE)
    n = len(POS_COLS)
    print(f"Loaded data: n={n}, train rows={len(train_seg_df)}, "
          f"train segments={train_seg_df['segment_id'].nunique()}, "
          f"val segments={val_seg_df['segment_id'].nunique()}, device={DEVICE}")

    # ----- loaders / batches -----
    train_loader = SegmentDataLoader_general(
        train_seg_df, pos_cols=POS_COLS, mom_cols=MOM_COLS,
        time_col='t', seg_col='segment_id', traj_col='trajectory_id',
        seed=SEED, device=DEVICE)
    val_batch = make_validation_batch(
        val_seg_df, pos_cols=POS_COLS, mom_cols=MOM_COLS,
        time_col='t', seg_col='segment_id', traj_col='trajectory_id', device=DEVICE)
    val_train_batch = make_validation_batch(
        train_seg_df, pos_cols=POS_COLS, mom_cols=MOM_COLS,
        time_col='t', seg_col='segment_id', traj_col='trajectory_id', device=DEVICE)

    # ----- model -----
    model_config = build_model_config(n)
    mapping_net = SimpleStackedHamiltonianNetwork_general(**model_config).to(DEVICE)
    print(f"Model: {sum(p.numel() for p in mapping_net.parameters())} params, "
          f"L={model_config['n_layers']}, hidden={model_config['hidden_dims']}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    loss_scales_path = os.path.join(SAVE_DIR, "loss_scales.pkl")

    if not RESUME:
        # ===== FRESH: calibrate up front, save pred_scale, then train =====
        inverse_net = InverseStackedHamiltonianNetwork_general(mapping_net).to(DEVICE)
        pred_scale = calibrate_scales(
            train_loader, mapping_net, inverse_net,
            add_noise=ADD_NOISE, noise_variance=NOISE_VARIANCE,
            noise_threshold=NOISE_THRESHOLD, device=DEVICE,
            save_path=loss_scales_path)          # <-- saves the pickle right here

        train_model_general(
            train_loader, mapping_net,
            val_batch=val_batch, val_train_batch=val_train_batch,
            lambda_pred=LAMBDA_PRED, lambda_hsic=LAMBDA_HSIC,
            pred_scale=pred_scale,               # <-- pass the fixed, already-saved scale
            add_noise=ADD_NOISE, noise_variance=NOISE_VARIANCE, noise_threshold=NOISE_THRESHOLD,
            use_unbiased_hsic=True, model_config=model_config,
            optimizer_type='AdamW', learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            scheduler_type='plateau', scheduler_params=SCHEDULER_PARAMS,
            grad_clip_value=GRAD_CLIP,
            num_epochs=NUM_EPOCHS, reshuffle_each_epoch=True,
            early_stopping=EARLY_STOPPING, patience=PATIENCE, min_delta=MIN_DELTA,
            save_dir=SAVE_DIR, save_best_only=SAVE_BEST_ONLY, save_freq_epochs=SAVE_FREQ_EPOCHS, log_segment_progress=False,
            auto_rescue=True, log_freq_batches=LOG_FREQ_BATCHES, verbose=VERBOSE, device=DEVICE,
        )
    else:
        # ===== RESUME: load the saved pred_scale, continue from checkpoint =====
        checkpoint_path = os.path.join(SAVE_DIR, "checkpoint_epoch_1599.pt")
        with open(loss_scales_path, "rb") as f:
            saved_pred_scale = pickle.load(f)["pred_scale"]
        print(f"Loaded saved pred_scale = {saved_pred_scale:.6f}")

        resume_training_from_checkpoint_general(
            checkpoint_path, train_loader, mapping_net,
            val_batch=val_batch, val_train_batch=val_train_batch,
            lambda_pred=LAMBDA_PRED, lambda_hsic=LAMBDA_HSIC,
            pred_scale=saved_pred_scale,          # <-- reuse original scale, no recalibration
            add_noise=ADD_NOISE, noise_variance=NOISE_VARIANCE,
            use_unbiased_hsic=True, model_config=model_config,
            load_scheduler_and_optimizer=RESUME_MODE_A,
            learning_rate=LEARNING_RATE,
            weight_decay=(None if RESUME_MODE_A else WEIGHT_DECAY),
            optimizer_type='AdamW', scheduler_type='plateau',
            scheduler_params=SCHEDULER_PARAMS, reset_scheduler_patience=False,
            num_epochs=RESUME_NUM_EPOCHS, grad_clip_value=GRAD_CLIP, reshuffle_each_epoch=True,
            early_stopping=EARLY_STOPPING, patience=PATIENCE, min_delta=MIN_DELTA,
            save_dir=SAVE_DIR, save_best_only=SAVE_BEST_ONLY, save_freq_epochs=SAVE_FREQ_EPOCHS,
            auto_rescue=True, log_freq_batches=LOG_FREQ_BATCHES, verbose=VERBOSE, device=DEVICE,
        )




if __name__ == "__main__":
    main()