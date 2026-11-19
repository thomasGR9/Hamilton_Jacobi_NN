"""
Single-sweep timing script for the HJN (PyTorch) method.

Designed to be invoked once per sweep by run_benchmark.sh. Loads the model,
does one in-process warm-up call, then runs ONE measured sweep through all
N_obs observation points, saving a per-sweep pickle.

The bash loop invokes this script 1+N times: one cache-warmup invocation
(--warmup-only, no pickle saved) plus N timed invocations (each saves a
sweep pickle). The aggregator stitches the N sweep pickles into one final
pickle in the same format as the original in-process timing function.

Usage:
    python time_one_sweep_pytorch.py \
        --config-py /path/to/hjn_config.py \
        --data-pkl /path/to/all_dataframes_low_samples.pkl \
        --test-id-key test_id_df_low_samples_multiple_periods \
        --test-data-key test_df_low_samples_multiple_periods \
        --checkpoint /path/to/best_model.pt \
        --Tmax 6.492624759674072 \
        --sweep-index 3 \
        --output-dir benchmark_outputs/hjn_task1_sweeps/

    # For warmup-only (no pickle saved):
    python time_one_sweep_pytorch.py ... --warmup-only

The --config-py file is a Python file that defines:
  - build_networks(device, checkpoint_path) -> (mapping_net, inverse_net)
  - get_data_from_trajectory_id (the data accessor function)
  - Batched_Autoregressive_Handoff_Inference_Protocol (the inference function)
See hjn_config_example.py for a template.
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
    required = ["build_networks", "get_data_from_trajectory_id",
                "Batched_Autoregressive_Handoff_Inference_Protocol"]
    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(f"Config file must define {name}")
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

    mapping_net, inverse_net = user.build_networks(
        device=device, checkpoint_path=args.checkpoint
    )
    mapping_net.eval()
    inverse_net.eval()

    with open(args.data_pkl, "rb") as f:
        dfs = pickle.load(f)
    test_id_df = dfs[args.test_id_key]
    test_df = dfs[args.test_data_key]

    trajectory_ids = test_id_df["trajectory_id"].values.astype(int)
    all_x, all_u = [], []
    ref_t = None
    for _, row_data in test_id_df.iterrows():
        traj_id = int(row_data["trajectory_id"])
        td = user.get_data_from_trajectory_id(test_id_df, test_df, trajectory_ids=traj_id)
        x_i = torch.as_tensor(td["x"].to_numpy(dtype=np.float32), device=device)
        u_i = torch.as_tensor(td["u"].to_numpy(dtype=np.float32), device=device)
        t_i = torch.as_tensor(td["t"].to_numpy(dtype=np.float32), device=device)
        if ref_t is None:
            ref_t = t_i
        else:
            if len(t_i) != len(ref_t) or not torch.allclose(t_i, ref_t, atol=1e-6):
                raise ValueError(f"Trajectory {traj_id} has different time values.")
        all_x.append(x_i)
        all_u.append(u_i)

    x_batch = torch.stack(all_x, dim=0)
    u_batch = torch.stack(all_u, dim=0)
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
        user.Batched_Autoregressive_Handoff_Inference_Protocol(
            mapping_net=mapping_net,
            inverse_net=inverse_net,
            x_obs=x_batch[:, obs_idx],
            u_obs=u_batch[:, obs_idx],
            t_obs=t_np[obs_idx],
            t_min=t_np[0],
            t_max=t_np[-1],
            Tmax=args.Tmax,
            query_times=query_times_per_obs[obs_idx],
            query_indices=query_indices_per_obs[obs_idx],
            device=device,
        )

    # In-process warmup: 1 single-obs call. Handles first-call effects
    # (CUDA context init, cuDNN autotune) symmetric with JAX side.
    print(f"[sweep {args.sweep_index}] In-process warmup (1 call)...")
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