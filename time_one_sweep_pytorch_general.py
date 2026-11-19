"""
Single-sweep timing script for the general-n HJN (PyTorch) method.

Generalization of time_one_sweep_pytorch.py to n degrees of freedom. The
timing protocol is UNCHANGED (in-process warm-up call, perf_counter with
device synchronization on both sides of every measurement, per-sweep peak
memory relative to the pre-sweep baseline, TF32 disabled, cuDNN autotune +
deterministic). Only the phase-space tensors gain a trailing DOF axis, and
the inference call is the general-n protocol.

Designed to be invoked once per sweep by run_benchmark_general.sh. Loads the
model, does one in-process warm-up call, then runs ONE measured sweep through
all N_obs observation points, saving a per-sweep pickle.

Usage:
    python time_one_sweep_pytorch_general.py \
        --config-py /path/to/hjn_config_general.py \
        --data-pkl /path/to/all_dataframes_henon_heiles.pkl \
        --test-id-key test_id_df_henon_heiles \
        --test-data-key test_df_henon_heiles \
        --checkpoint /path/to/best_model.pt \
        --Tmax 8.563 \
        --sweep-index 3 \
        --output-dir benchmark_outputs/hjn_task3_sweeps/

    # For warmup-only (no pickle saved):
    python time_one_sweep_pytorch_general.py ... --warmup-only

The --config-py file must define:
  - N_DOF, POS_COLS, MOM_COLS
  - build_networks(device, checkpoint_path) -> (mapping_net, inverse_net)
  - get_data_from_trajectory_id (the data accessor function)
  - Batched_Autoregressive_Handoff_Inference_Protocol_general (inference fn)
See hjn_config_general.py for a template.
"""

import argparse
import datetime
import importlib.util
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch


def import_config(config_py_path):
    spec = importlib.util.spec_from_file_location("user_config", config_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config from {config_py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_config"] = module
    spec.loader.exec_module(module)
    required = ["N_DOF", "POS_COLS", "MOM_COLS",
                "build_networks", "get_data_from_trajectory_id", "load_dataframes",
                "Batched_Autoregressive_Handoff_Inference_Protocol_general"]
    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(f"Config file must define {name}")
    n = int(module.N_DOF)
    if len(module.POS_COLS) != n:
        raise RuntimeError(
            f"POS_COLS must have N_DOF={n} entries, got {len(module.POS_COLS)}: {module.POS_COLS}")
    if len(module.MOM_COLS) != n:
        raise RuntimeError(
            f"MOM_COLS must have N_DOF={n} entries, got {len(module.MOM_COLS)}: {module.MOM_COLS}")
    if len(set(list(module.POS_COLS) + list(module.MOM_COLS))) != 2 * n:
        raise RuntimeError(
            f"POS_COLS + MOM_COLS must be distinct, got "
            f"{list(module.POS_COLS) + list(module.MOM_COLS)}")
    return module


def set_fairness_flags():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-py", required=True)
    parser.add_argument("--data-pkl", required=True)
    parser.add_argument("--test-id-key", required=True)
    parser.add_argument("--test-data-key", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--Tmax", type=float, required=True)
    parser.add_argument("--sweep-index", type=int, default=-1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup-only", action="store_true")
    parser.add_argument("--task-name", default="unknown_task")
    parser.add_argument("--method-name", default="HJN")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA.")

    set_fairness_flags()
    user = import_config(args.config_py)

    n = int(user.N_DOF)
    pos_cols = list(user.POS_COLS)
    mom_cols = list(user.MOM_COLS)

    mapping_net, inverse_net = user.build_networks(
        device=device, checkpoint_path=args.checkpoint
    )
    mapping_net.eval()
    inverse_net.eval()

    if hasattr(mapping_net, "n") and mapping_net.n != n:
        raise RuntimeError(f"mapping_net.n={mapping_net.n} does not match N_DOF={n}")
    if hasattr(inverse_net, "n") and inverse_net.n != n:
        raise RuntimeError(f"inverse_net.n={inverse_net.n} does not match N_DOF={n}")



    dfs = user.load_dataframes(args.data_pkl)
    for key in (args.test_id_key, args.test_data_key):
        if key not in dfs:
            raise RuntimeError(f"Key '{key}' not in {args.data_pkl}. Available: {sorted(dfs)}")
    test_id_df = dfs[args.test_id_key]
    test_df = dfs[args.test_data_key]

    trajectory_ids = test_id_df["trajectory_id"].values.astype(int)
    all_x, all_u = [], []
    ref_t = None
    for _, row_data in test_id_df.iterrows():
        traj_id = int(row_data["trajectory_id"])
        td = user.get_data_from_trajectory_id(test_id_df, test_df, trajectory_ids=traj_id)
        missing = [c for c in pos_cols + mom_cols + ["t"] if c not in td.columns]
        if missing:
            raise ValueError(f"Trajectory {traj_id}: columns missing: {missing}")
        x_i = torch.as_tensor(td[pos_cols].to_numpy(dtype=np.float32), device=device)   # (T, n)
        u_i = torch.as_tensor(td[mom_cols].to_numpy(dtype=np.float32), device=device)   # (T, n)
        t_i = torch.as_tensor(td["t"].to_numpy(dtype=np.float32), device=device)        # (T,)
        if x_i.dim() != 2 or u_i.dim() != 2 or x_i.shape[1] != n or u_i.shape[1] != n:
            raise ValueError(
                f"Trajectory {traj_id}: expected (T, {n}) coordinate/momentum tensors, "
                f"got {tuple(x_i.shape)} and {tuple(u_i.shape)}")
        if ref_t is None:
            ref_t = t_i
        else:
            if len(t_i) != len(ref_t) or not torch.allclose(t_i, ref_t, atol=1e-6):
                raise ValueError(f"Trajectory {traj_id} has different time values.")
        all_x.append(x_i)
        all_u.append(u_i)

    x_batch = torch.stack(all_x, dim=0)          # (N_traj, N_obs, n)
    u_batch = torch.stack(all_u, dim=0)          # (N_traj, N_obs, n)
    t_np = ref_t.detach().cpu().numpy()
    N_obs = len(t_np)
    N_traj = len(trajectory_ids)

    query_indices_per_obs = []
    query_times_per_obs = []
    for obs_idx in range(N_obs):
        q_idx = [i for i in range(N_obs) if i != obs_idx]
        q_t = t_np[q_idx]
        query_indices_per_obs.append(q_idx)
        query_times_per_obs.append(q_t)

    def run_single_obs(obs_idx):
        user.Batched_Autoregressive_Handoff_Inference_Protocol_general(
            mapping_net=mapping_net,
            inverse_net=inverse_net,
            x_obs=x_batch[:, obs_idx, :],        # (N_traj, n)
            u_obs=u_batch[:, obs_idx, :],        # (N_traj, n)
            t_obs=t_np[obs_idx],
            t_min=t_np[0],
            t_max=t_np[-1],
            Tmax=args.Tmax,
            query_times=query_times_per_obs[obs_idx],
            query_indices=query_indices_per_obs[obs_idx],
            device=device,
            n=n,
        )

    # In-process warmup: 1 single-obs call. Handles first-call effects
    # (CUDA context init, cuDNN autotune) symmetric with JAX side.
    print(f"[sweep {args.sweep_index}] In-process warmup (1 call, n={n}, "
          f"N_traj={N_traj}, N_obs={N_obs})...")
    run_single_obs(0)
    torch.cuda.synchronize(device)

    if args.warmup_only:
        print(f"[sweep {args.sweep_index}] --warmup-only: not saving.")
        return

    torch.cuda.reset_peak_memory_stats(device)
    mem_baseline_bytes = torch.cuda.memory_allocated(device)

    sweep_times = np.empty(N_obs)
    print(f"[sweep {args.sweep_index}] Measured sweep ({N_obs} obs)...")

    sweep_start = time.perf_counter()
    for obs_idx in range(N_obs):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        run_single_obs(obs_idx)
        torch.cuda.synchronize(device)
        sweep_times[obs_idx] = time.perf_counter() - start
    sweep_total_s = time.perf_counter() - sweep_start

    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_above_baseline = peak_bytes - mem_baseline_bytes

    sweep_result = {
        "method_name": args.method_name,
        "task_name": args.task_name,
        "sweep_index": args.sweep_index,
        "timestamp": datetime.datetime.now().isoformat(),
        "sweep_times": sweep_times,
        "sweep_total_s": sweep_total_s,
        "N_obs": N_obs,
        "N_traj": N_traj,
        "n": n,
        "pos_cols": tuple(pos_cols),
        "mom_cols": tuple(mom_cols),
        "t_min": float(t_np[0]),
        "t_max": float(t_np[-1]),
        "Tmax": float(args.Tmax),
        "peak_bytes_during_sweep": int(peak_bytes),
        "peak_bytes_above_baseline": int(peak_above_baseline),
        "peak_MB_during_sweep": peak_bytes / 1024 ** 2,
        "peak_MB_above_baseline": peak_above_baseline / 1024 ** 2,
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        },
        "fairness_settings": {
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": True,
            "cudnn_deterministic": True,
            "precision": "FP32 (TF32 disabled)",
        },
        "t_array": t_np,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sweep_{args.sweep_index:03d}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(sweep_result, f)

    print(f"[sweep {args.sweep_index}] Saved {out_path} "
          f"(total {sweep_total_s:.2f}s, "
          f"mean {sweep_times.mean()*1000:.2f}ms/obs, "
          f"peak {peak_above_baseline/1024**2:.2f}MB above baseline)")


if __name__ == "__main__":
    main()