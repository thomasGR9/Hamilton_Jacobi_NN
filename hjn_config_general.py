"""
General-n HJN config file, consumed by time_one_sweep_pytorch_general.py
via --config-py.

It must define exactly six things:

  1. N_DOF      : int, number of degrees of freedom (n)
  2. POS_COLS   : tuple of N_DOF coordinate column names in test_df
  3. MOM_COLS   : tuple of N_DOF momentum column names in test_df
  4. build_networks(device, checkpoint_path) -> (mapping_net, inverse_net)
  5. get_data_from_trajectory_id(ids_df, data_df, trajectory_ids)
  6. Batched_Autoregressive_Handoff_Inference_Protocol_general(...)

IMPORTANT — the architecture block in build_networks must match the trained
checkpoint. Structural parameters that MUST agree with training:
    n, n_layers, hidden_dims, n_hidden_layers, activation,
    final_activation, final_activation_only_on_final_layer,
    use_bias, use_layer_norm, a_eps_min, a_eps_max, a_k
Note a_eps_min / a_eps_max / a_k are NOT learned parameters: they define the
reparameterisation a = a_min + (a_max - a_min) * sigmoid(a_k * a_raw). A
mismatch loads cleanly and silently changes every layer's scaling.
The init means / variances / seed are overwritten by the checkpoint and so
do not affect inference.
"""

# ============================================================
# Imports from your project (adjust path as needed)
# ============================================================
import pandas as pd
import sys
sys.path.append("/home/thomasdev/Desktop/Hamilton_Jacobi_NN/Hamilton_Jacobi_NN")  # so the imports below work

import torch
import pickle
from modules_general import (
    SimpleStackedHamiltonianNetwork_general,
    InverseStackedHamiltonianNetwork_general,
    load_checkpoint,
    get_data_from_trajectory_id,
    Batched_Autoregressive_Handoff_Inference_Protocol_general,
)


# ============================================================
# Degrees of freedom and phase-space column names
# ============================================================
N_DOF = 2
POS_COLS = ("x", "y")
MOM_COLS = ("px", "py")
def load_dataframes(path):
    """
    Reconstruct the dataframes from the portable pickle format
    ({"data", "columns", "index_values", "index_name", "shape"} per entry).

    The index restore is REQUIRED: get_data_from_trajectory_id slices with
    data_df.loc[start : end-1], which is label-based, so a reconstruction
    that leaves a default RangeIndex silently returns the wrong rows.
    """
    with open(path, "rb") as f:
        portable = pickle.load(f)

    dfs = {}
    for name, d in portable.items():
        if isinstance(d, pd.DataFrame):        # already a DataFrame
            dfs[name] = d
            continue
        df = pd.DataFrame(d["data"], columns=d["columns"])
        df.index = pd.Index(d["index_values"], name=d["index_name"])
        if "shape" in d and df.shape != tuple(d["shape"]):
            raise RuntimeError(
                f"{name}: reconstructed shape {df.shape} != stored {tuple(d['shape'])}")
        dfs[name] = df
    return dfs

# ============================================================
# Strict checkpoint verification
# ============================================================
def _verify_checkpoint_loaded(mapping_net, checkpoint_path, device):
    """
    load_checkpoint() swallows load_state_dict failures (try/except + print),
    which in a multi-process benchmark would silently time a randomly
    initialised network. This re-reads the checkpoint and asserts that every
    stored tensor is present, shape-compatible, and actually resident in the
    model after the load.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' not in ckpt:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no 'model' entry.")
    sd = ckpt['model']
    model_sd = mapping_net.state_dict()

    missing = [k for k in sd if k not in model_sd]
    extra = [k for k in model_sd if k not in sd]
    bad_shape = [k for k in sd if k in model_sd
                 and tuple(model_sd[k].shape) != tuple(sd[k].shape)]
    if missing or extra or bad_shape:
        raise RuntimeError(
            "Checkpoint / architecture mismatch — the build_networks() block does "
            "not match the trained model.\n"
            f"  in checkpoint but not in model : {missing[:5]}\n"
            f"  in model but not in checkpoint : {extra[:5]}\n"
            f"  shape mismatch                 : "
            f"{[(k, tuple(sd[k].shape), tuple(model_sd[k].shape)) for k in bad_shape[:5]]}"
        )

    not_loaded = [k for k in sd
                  if not torch.equal(model_sd[k].detach().cpu(), sd[k].detach().cpu())]
    if not_loaded:
        raise RuntimeError(
            f"Checkpoint weights were NOT loaded into the model (load_checkpoint "
            f"failed silently). Offending keys: {not_loaded[:5]}"
        )

    n_params = sum(p.numel() for p in mapping_net.parameters())
    print(f"[config] checkpoint verified: {len(sd)} tensors loaded, "
          f"{n_params} parameters, n={getattr(mapping_net, 'n', '?')}")


# ============================================================
# build_networks: signature required by the sweep script
# ============================================================
def build_networks(device, checkpoint_path):
    mapping_net = SimpleStackedHamiltonianNetwork_general(
        n=N_DOF,

        # How many HJL (Step_1 + Step_2) layers to stack
        n_layers=10,

        # MLP architecture
        hidden_dims=[33, 60, 32],
        n_hidden_layers=None,          # ignored when hidden_dims is a list

        # Activations
        activation='gelu',
        activation_params=None,
        final_activation=None,         # linear output on final HJL
        final_activation_only_on_final_layer=True,

        # Initialization (overwritten by the checkpoint; kept for exactness)
        weight_init='orthogonal',
        weight_init_params=None,
        bias_init='zeros',
        bias_init_value=0.0,

        # Architectural choices
        use_bias=True,
        use_layer_norm=False,

        # a reparameterisation — STRUCTURAL, must match training
        a_eps_min=0.5,
        a_eps_max=2.0,
        a_k=0.1,

        # a init
        step_1_a_mean_innit=1.6,
        step_2_a_mean_innit=1.6,
        a_std_ratio=0.1,

        # gamma init
        step_1_gamma_mean_innit=2.0,
        step_2_gamma_mean_innit=2.0,
        gamma_variance=0.04,

        # c1 init
        step_1_c1_mean_innit=0.1,
        step_2_c1_mean_innit=-0.1,
        c1_variance=0.01,

        # c2 init
        step_1_c2_mean_innit=-0.1,
        step_2_c2_mean_innit=0.1,
        c2_variance=0.01,

        seed=42,
    ).to(device)

    load_checkpoint(
        path=checkpoint_path,
        mapping_net=mapping_net,
        device=device,
        optimizer=None,
        scheduler=None,
    )
    _verify_checkpoint_loaded(mapping_net, checkpoint_path, device)

    inverse_net = InverseStackedHamiltonianNetwork_general(forward_network=mapping_net)
    return mapping_net, inverse_net