#!/bin/bash
# ============================================================================
# In-Context Learning Experiments for ECG Age Prediction
#
# Usage:
#   bash run_experiments.sh                    # Run all baselines + context methods
#   bash run_experiments.sh --with-transformer # Also train/evaluate ICL transformer
#   bash run_experiments.sh --quick            # Quick run with fewer experiments
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ── Configuration ──
DATA_DIR="${REPO_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/results"

# Automatically find the best checkpoint (prefer LDS+FDS over vanilla)
CHECKPOINT_DIR="${REPO_DIR}/checkpoint_resnetWang"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_DIR="${REPO_DIR}/checkpoint_inception"
fi

find_best_checkpoint() {
    # Prefer LDS+FDS checkpoint, fall back to vanilla
    local ckpt=""
    ckpt=$(find "$CHECKPOINT_DIR" -name "ckpt.best.pth.tar" -path "*LDS_FDS*" 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        ckpt=$(find "$CHECKPOINT_DIR" -name "ckpt.best.pth.tar" 2>/dev/null | head -1)
    fi
    echo "$ckpt"
}

CHECKPOINT=$(find_best_checkpoint)
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in $CHECKPOINT_DIR"
    echo "Please train a model first using train.py, or set CHECKPOINT manually."
    exit 1
fi

echo "=============================================="
echo "In-Context Learning for ECG Age Prediction"
echo "=============================================="
echo "Data dir:    $DATA_DIR"
echo "Checkpoint:  $CHECKPOINT"
echo "Output dir:  $OUTPUT_DIR"
echo "=============================================="

# Determine backbone from checkpoint path
BACKBONE="resnet1d_wang_fds"
if echo "$CHECKPOINT" | grep -q "inception"; then
    BACKBONE="inception1d_fds"
fi
echo "Backbone:    $BACKBONE"

# ── Parse arguments ──
WITH_TRANSFORMER=false
WITH_PFN=false
QUICK=false
for arg in "$@"; do
    case $arg in
        --with-transformer) WITH_TRANSFORMER=true ;;
        --with-pfn)         WITH_PFN=true ;;
        --quick)            QUICK=true ;;
    esac
done

cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_DIR"

# ── Experiment 1: Baselines + Context Methods (k=15) ──
echo ""
echo "=== Experiment 1: Baselines + Context Selection (k=15) ==="
STRATEGIES="vanilla,subsample,smoter,inverse"
if $QUICK; then
    STRATEGIES="vanilla,inverse"
fi

TRANSFORMER_FLAG=""
TRANSFORMER_ARGS=""
if $WITH_TRANSFORMER; then
    TRANSFORMER_FLAG="--run_transformer"
    TRANSFORMER_ARGS="--transformer_epochs 30 --episodes_per_epoch 5000"
fi

PFN_FLAG=""
if $WITH_PFN; then
    PFN_FLAG="--run_pfn"
fi

for SEED in 0 1 2; do
    echo ""
    echo "--- Seed $SEED ---"
    python benchmark.py \
        --checkpoint "$CHECKPOINT" \
        --data_dir "$DATA_DIR" \
        --dataset 1000_timesteps \
        --backbone "$BACKBONE" \
        --output_dir "$OUTPUT_DIR" \
        --context_size 15 \
        --strategies "$STRATEGIES" \
        --seed "$SEED" \
        $TRANSFORMER_FLAG \
        $TRANSFORMER_ARGS \
        $PFN_FLAG

    if $QUICK; then
        break  # Only one seed for quick mode
    fi
done

# ── Experiment 2: Vary context size k ──
if ! $QUICK; then
    echo ""
    echo "=== Experiment 2: Varying Context Size k ==="
    for K in 5 10 15 25 50; do
        echo ""
        echo "--- k=$K ---"
        python benchmark.py \
            --checkpoint "$CHECKPOINT" \
            --data_dir "$DATA_DIR" \
            --dataset 1000_timesteps \
            --backbone "$BACKBONE" \
            --output_dir "$OUTPUT_DIR" \
            --context_size "$K" \
            --strategies "vanilla,inverse" \
            --no_baselines \
            --seed 0
    done
fi

# ── Aggregate results ──
echo ""
echo "=== Aggregating Results ==="
python -c "
import pandas as pd
import glob
import os

csv_files = glob.glob('${OUTPUT_DIR}/results_*.csv')
if csv_files:
    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('${OUTPUT_DIR}/all_results.csv', index=False)
    print(f'Combined {len(csv_files)} result files into all_results.csv')

    # Print summary
    summary = combined.groupby('method').agg({
        'overall_l1': ['mean', 'std'],
        'many_l1': ['mean', 'std'],
        'few_l1': ['mean', 'std'],
        'overall_gmean': ['mean', 'std'],
    }).round(3)
    print()
    print(summary.to_string())
else:
    print('No result files found.')
"

echo ""
echo "=============================================="
echo "Done! Results saved to: $OUTPUT_DIR"
echo "=============================================="
