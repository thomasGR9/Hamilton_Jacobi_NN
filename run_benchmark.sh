#!/bin/bash
# Multi-process timing benchmark driver.
#
# Invokes a sweep-timing script N+1 times: one warmup invocation (populates
# JAX disk compile cache + cuDNN state), then N timed invocations (each saves
# a per-sweep pickle).
#
# Designed to be parameterized for either method via the SWEEP_SCRIPT variable
# and the SCRIPT_ARGS string. Edit the configuration block at the top.
#
# After this script completes, run aggregate_sweeps.py on the output dir to
# produce the final pickle.
#
# Usage:
#   ./run_benchmark.sh

set -e  # exit on any error

# =============================================================================
# CONFIGURATION — EDIT THIS BLOCK
# =============================================================================


CONDA_ENV="ultralytics-env"
SWEEP_SCRIPT="time_one_sweep_pytorch.py"
OUTPUT_DIR="benchmark_outputs/hjn_task2_sweeps_high_energy"
SCRIPT_ARGS=(
    --config-py /home/thomasdev/Desktop/Hamilton_Jacobi_NN/Hamilton_Jacobi_NN/hjn_config.py
    --data-pkl /home/thomasdev/Desktop/Hamilton_Jacobi_NN/Hamilton_Jacobi_NN/df_high_energy_low_samples_pendulum.pkl
    --test-id-key test_id_df_high_energy_low_samples_pendulum
    --test-data-key test_df_high_energy_low_samples_pendulum
    --checkpoint /home/thomasdev/Desktop/Hamilton_Jacobi_NN/Hamilton_Jacobi_NN//save_directory_noisy_full_pred_low_samples_pendulum_4/best_model.pt
    --Tmax 4.06
    --task-name task_1
    --method-name HJN
)


# Number of timed sweeps (analogous to n_repeats in the in-process function)
N_SWEEPS=20

# JAX-specific: persistent compile cache directory (only matters for JAX runs;
# harmless for PyTorch).
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax_compile_cache_benchmark"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# =============================================================================
# END CONFIGURATION
# =============================================================================

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Multi-process benchmark"
echo "  Sweep script: $SWEEP_SCRIPT"
echo "  Conda env:    $CONDA_ENV"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Sweeps:       1 warmup + $N_SWEEPS timed"
echo "  Start time:   $(date)"
echo "============================================================"

# Initialize conda for this shell
# (conda activate doesn't work in non-interactive shells without this)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ----- WARMUP INVOCATION -----
echo
echo "[$(date +%T)] Warmup invocation (populates compile cache, no pickle saved)..."
python "$SWEEP_SCRIPT" \
    "${SCRIPT_ARGS[@]}" \
    --sweep-index -1 \
    --output-dir "$OUTPUT_DIR" \
    --warmup-only
echo "[$(date +%T)] Warmup complete."

# ----- TIMED INVOCATIONS -----
for i in $(seq 1 $N_SWEEPS); do
    echo
    echo "[$(date +%T)] Timed sweep $i / $N_SWEEPS ..."
    python "$SWEEP_SCRIPT" \
        "${SCRIPT_ARGS[@]}" \
        --sweep-index "$i" \
        --output-dir "$OUTPUT_DIR"
done

echo
echo "============================================================"
echo "All sweeps complete. End time: $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "Next step: run aggregate_sweeps.py --sweep-dir $OUTPUT_DIR"
echo "============================================================"