#!/bin/bash
################################################################################
# IM-Context Reproduction Script for ECG Age Prediction
# 
# This script reproduces the TMLR 2024 paper "IM-Context: In-Context Learning 
# for Imbalanced Regression Tasks" using the PTB-XL ECG age prediction dataset
# with age-based stratification (many: <30, median: 30-60, few: >60).
#
# Usage:
#   bash run_im_context_reproduction.sh [quick|full|transform]
#     quick     - Run quick baseline experiments (fastest)
#     full      - Run all baseline + context strategies (recommended)
#     transform - Also train ICL transformer (slowest, expensive)
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
IN_CONTEXT_DIR="$REPO_DIR/in_context"

# Configuration
DATA_DIR="$REPO_DIR/data"
OUTPUT_DIR="$IN_CONTEXT_DIR/results"
LOG_FILE="$OUTPUT_DIR/logs/reproduction.log"

# Determine checkpoint
CHECKPOINT=$(find "$REPO_DIR/checkpoint_resnetWang" -name "ckpt.best.pth.tar" -path "*LDS_FDS*" 2>/dev/null | head -1)
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(find "$REPO_DIR/checkpoint_inception" -name "ckpt.best.pth.tar" 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found. Please train a model first." >&2
    exit 1
fi

# Create directories
mkdir -p "$OUTPUT_DIR/logs"

# Determine backbone from checkpoint
BACKBONE="resnet1d_wang_fds"
if echo "$CHECKPOINT" | grep -q "inception"; then
    BACKBONE="inception1d_fds"
fi

# Parse mode argument
MODE="${1:-full}"
TRANSFORMER_FLAG=""
case "$MODE" in
    quick)
        STRATEGIES="vanilla,inverse"
        echo "[QUICK MODE] Running only vanilla and inverse strategies"
        ;;
    full)
        STRATEGIES="vanilla,subsample,smoter,inverse"
        echo "[FULL MODE] Running all strategies (vanilla, subsample, smoter, inverse)"
        ;;
    transform)
        STRATEGIES="vanilla,subsample,smoter,inverse"
        TRANSFORMER_FLAG="--run_transformer --transformer_epochs 30 --episodes_per_epoch 5000"
        echo "[FULL+TRANSFORMER MODE] Running all strategies with ICL transformer training"
        ;;
    *)
        echo "Usage: $0 [quick|full|transform]" >&2
        exit 1
        ;;
esac

# Log header
{
    echo "==============================================================================="
    echo "IM-Context Reproduction: In-Context Learning for ECG Age Prediction"
    echo "==============================================================================="
    echo "Start time:      $(date)"
    echo "Repository:      $REPO_DIR"
    echo "Data directory:  $DATA_DIR"
    echo "Output dir:      $OUTPUT_DIR"
    echo "Checkpoint:      $CHECKPOINT"
    echo "Backbone:        $BACKBONE"
    echo "Mode:            $MODE"
    echo "Strategies:      $STRATEGIES"
    echo "==============================================================================="
} | tee -a "$LOG_FILE"

# Data summary
{
    echo ""
    echo "Data Summary:"
    python -c "
import pandas as pd
df = pd.read_csv('$DATA_DIR/1000_timesteps.csv')
total = len(df)
many = len(df[df['age'] < 30])
median = len(df[(df['age'] >= 30) & (df['age'] < 60)])
few = len(df[df['age'] >= 60])
print(f'  Total samples:     {total:,}')
print(f'  Age <30 (many):    {many:,} ({100*many/total:.1f}%)')
print(f'  Age 30-60 (median):{median:,} ({100*median/total:.1f}%)')
print(f'  Age >=60 (few):    {few:,} ({100*few/total:.1f}%)')
print(f'  Age range:         {df[\"age\"].min():.0f} - {df[\"age\"].max():.0f}')
"
} | tee -a "$LOG_FILE"

# Run experiments with different seeds for stability
echo "" | tee -a "$LOG_FILE"
echo "Running experiments..." | tee -a "$LOG_FILE"

for SEED in 0 1 2; do
    {
        echo ""
        echo "───────────────────────────────────────────────────────────────────────────────"
        echo "Experiment round with seed=$SEED"
        echo "───────────────────────────────────────────────────────────────────────────────"
    } | tee -a "$LOG_FILE"

    cd "$IN_CONTEXT_DIR"
    python benchmark.py \
        --checkpoint "$CHECKPOINT" \
        --data_dir "$DATA_DIR" \
        --dataset 1000_timesteps \
        --backbone "$BACKBONE" \
        --output_dir "$OUTPUT_DIR" \
        --context_size 15 \
        --strategies "$STRATEGIES" \
        --seed "$SEED" \
        --run_baselines \
        $TRANSFORMER_FLAG \
        2>&1 | tee -a "$LOG_FILE"
done

# Compile results
{
    echo ""
    echo "───────────────────────────────────────────────────────────────────────────────"
    echo "Compiling results..."
    echo "───────────────────────────────────────────────────────────────────────────────"
} | tee -a "$LOG_FILE"

python "$IN_CONTEXT_DIR/compile_results.py" \
    --results_dir "$OUTPUT_DIR" \
    --output_csv "$OUTPUT_DIR/FINAL_RESULTS.csv" \
    2>&1 | tee -a "$LOG_FILE"

# Print summary
{
    echo ""
    echo "═════════════════════════════════════════════════════════════════════════════"
    echo "IM-Context Reproduction Complete"
    echo "═════════════════════════════════════════════════════════════════════════════"
    echo "Results saved to:"
    echo "  - CSV summary:  $OUTPUT_DIR/FINAL_RESULTS.csv"
    echo "  - Full log:     $LOG_FILE"
    echo "  - Result CSVs:  $OUTPUT_DIR/seed_*/results.csv"
    echo "═════════════════════════════════════════════════════════════════════════════"
    echo "End time:        $(date)"
} | tee -a "$LOG_FILE"

echo ""
echo "✓ Reproduction script completed successfully!"
