"""
Aggregate per-sweep pickles from a multi-process benchmark run into one final
pickle in the same shape as the original in-process timing functions.

The output pickle's 'timing_stats' field is fully compatible with
plot_summary_statistics(..., timing_stats_batch=...) — the same plotting code
used for the original in-process runs works unchanged.

Usage:
    python aggregate_sweeps.py \
        --sweep-dir benchmark_outputs/hjn_task1_sweeps \
        --output benchmark_outputs/hjn_task1_final.pkl

After this, in your notebook:
    with open("benchmark_outputs/hjn_task1_final.pkl", "rb") as f:
        results = pickle.load(f)
    plot_summary_statistics(loss_results,
                            timing_stats_batch=results['timing_stats'])
"""

import argparse
import datetime
import pickle
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", required=True,
                        help="Directory containing sweep_NNN.pkl files.")
    parser.add_argument("--output", required=True,
                        help="Path for the aggregated output pickle.")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    sweep_files = sorted(sweep_dir.glob("sweep_*.pkl"))
    if not sweep_files:
        print(f"No sweep files found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(sweep_files)} sweep files in {sweep_dir}")

    sweeps = []
    for sf in sweep_files:
        with open(sf, "rb") as f:
            sweeps.append(pickle.load(f))
    print(f"Loaded {len(sweeps)} sweep pickles.")

    # Validate consistency
    ref = sweeps[0]
    for s in sweeps[1:]:
        for key in ("N_obs", "N_traj", "method_name", "task_name"):
            if s.get(key) != ref.get(key):
                print(f"WARNING: sweep {s.get('sweep_index')} has different "
                      f"{key} ({s.get(key)}) than reference ({ref.get(key)})",
                      file=sys.stderr)

    N_obs = ref["N_obs"]
    N_traj = ref["N_traj"]
    n_repeats = len(sweeps)

    # Stack: (n_repeats, N_obs)
    time_matrix = np.empty((n_repeats, N_obs))
    peak_MB_list = []
    sweep_indices = []
    sweep_totals = np.empty(n_repeats)
    timestamps = []

    for i, s in enumerate(sweeps):
        if len(s["sweep_times"]) != N_obs:
            raise ValueError(
                f"Sweep {s.get('sweep_index')} has {len(s['sweep_times'])} times, expected {N_obs}"
            )
        time_matrix[i, :] = s["sweep_times"]
        sweep_totals[i] = s["sweep_total_s"]
        sweep_indices.append(s.get("sweep_index", i))
        timestamps.append(s.get("timestamp"))
        peak = s.get("peak_MB_above_baseline")
        if peak is not None and peak >= 0:
            peak_MB_list.append(peak)

    # Stats: same as original in-process function
    per_obs_mean = np.mean(time_matrix, axis=0)
    per_obs_std = np.std(time_matrix, axis=0, ddof=1)

    mean_s = np.mean(per_obs_mean)
    std_s = np.std(per_obs_mean, ddof=1)
    median_s = np.median(per_obs_mean)
    min_s = np.min(per_obs_mean)
    max_s = np.max(per_obs_mean)
    measurement_noise_mean_ms = np.mean(per_obs_std) * 1000.0

    total_sweep_mean_s = np.mean(sweep_totals)
    total_sweep_std_s = np.std(sweep_totals, ddof=1)

    # Memory stats — aggregate from sweeps (each sweep tracked its own peak)
    if peak_MB_list:
        peak_MB_above_baseline_mean = float(np.mean(peak_MB_list))
        peak_MB_above_baseline_max = float(np.max(peak_MB_list))
        peak_MB_above_baseline_min = float(np.min(peak_MB_list))
    else:
        peak_MB_above_baseline_mean = None
        peak_MB_above_baseline_max = None
        peak_MB_above_baseline_min = None

    # timing_stats: drop-in compatible with plot_summary_statistics
    timing_stats = {
        "time_matrix": time_matrix,
        "per_obs_mean": per_obs_mean,
        "per_obs_std": per_obs_std,
        "mean_s": mean_s,
        "std_s": std_s,
        "measurement_noise_mean_ms": measurement_noise_mean_ms,
        "median_s": median_s,
        "min_s": min_s,
        "max_s": max_s,
        "total_sweep_mean_s": total_sweep_mean_s,
        "total_sweep_std_s": total_sweep_std_s,
        # Legacy fields for plot compatibility
        "n_warmup": 1,  # 1 warmup invocation
        "n_repeats": n_repeats,
        "N_obs": N_obs,
        "N_traj": N_traj,
        "device_name": ref["hardware"].get("gpu_name", "unknown"),
        "timing_method": "perf_counter_with_sync_multiprocess",
    }

    # Final results dict — same shape as in-process function output
    results = {
        "method_name": ref.get("method_name", "unknown"),
        "task_name": ref.get("task_name", "unknown"),
        "timestamp": datetime.datetime.now().isoformat(),
        "aggregation_info": {
            "n_sweep_pickles": n_repeats,
            "sweep_indices": sweep_indices,
            "first_sweep_timestamp": timestamps[0] if timestamps else None,
            "last_sweep_timestamp": timestamps[-1] if timestamps else None,
            "scheme": "multi-process: 1 warmup invocation + N timed invocations",
        },
        "hardware": ref.get("hardware", {}),
        "fairness_settings": ref.get("fairness_settings", {}),
        "protocol": {
            "n_warmup_invocations": 1,
            "n_timed_invocations": n_repeats,
            "in_process_warmup_calls_per_invocation": 1,
            "timing_method": "time.perf_counter() with synchronization",
            "scheme": "multi-process to avoid cross-sweep state accumulation",
        },
        "data": {
            "N_traj": N_traj,
            "N_obs": N_obs,
            "t_min": ref.get("t_min"),
            "t_max": ref.get("t_max"),
            "t_array": ref.get("t_array"),
            "Tmax": ref.get("Tmax"),
            "steps_per_dt": ref.get("steps_per_dt"),
        },
        "timing_stats": timing_stats,
        "memory_stats": {
            "peak_MB_above_baseline_mean": peak_MB_above_baseline_mean,
            "peak_MB_above_baseline_max": peak_MB_above_baseline_max,
            "peak_MB_above_baseline_min": peak_MB_above_baseline_min,
            "n_sweeps_with_memory_data": len(peak_MB_list),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    # Print summary
    print(f"\n{'=' * 78}")
    print(f"AGGREGATED RESULTS — {results['method_name']} / {results['task_name']}")
    print(f"{'=' * 78}")
    print(f"Aggregated over {n_repeats} sweeps "
          f"(sweep indices: {min(sweep_indices)}..{max(sweep_indices)})")
    print(f"\nPer observation point:")
    print(f"  mean ± std across {N_obs} obs choices: "
          f"{mean_s * 1000:.3f} ± {std_s * 1000:.3f} ms (STRUCTURAL)")
    print(f"  measurement noise (mean of per-obs std over repeats): "
          f"{measurement_noise_mean_ms:.3f} ms")
    print(f"  median: {median_s * 1000:.3f} ms")
    print(f"  range:  [{min_s * 1000:.3f}, {max_s * 1000:.3f}] ms")
    print(f"\nTotal sweep:")
    print(f"  {total_sweep_mean_s:.4f} ± {total_sweep_std_s:.4f} s "
          f"(mean ± std across {n_repeats} sweeps)")
    if peak_MB_above_baseline_mean is not None:
        print(f"\nPeak GPU memory above baseline (per-sweep tracked):")
        print(f"  mean: {peak_MB_above_baseline_mean:.2f} MB")
        print(f"  range: [{peak_MB_above_baseline_min:.2f}, {peak_MB_above_baseline_max:.2f}] MB")
    print(f"\nSaved aggregated pickle to: {output_path}")
    print(f"\nTo use in notebook:")
    print(f"  with open('{output_path}', 'rb') as f:")
    print(f"      results = pickle.load(f)")
    print(f"  plot_summary_statistics(loss_results, timing_stats_batch=results['timing_stats'])")


if __name__ == "__main__":
    main()