# In-Context Learning for ECG Age Prediction

Applies the **IM-Context** method (TMLR 2024) to ECG age prediction using the PTB-XL dataset. This approach uses adaptive context selection strategies to improve predictions in tail regions (young/elderly patients) where data is scarce.

## Method Overview

1. **Feature Extraction**: Use a pretrained ResNet1D/Inception1D backbone to extract 128-dim embeddings from raw ECG signals
2. **Feature Normalization**: StandardScaler + PowerTransformer (concatenated for richer features)
3. **Context Selection**: For each test sample, select k training samples as "context" using:
   - **Vanilla**: k-nearest neighbors in feature space (cosine similarity)
   - **Subsample**: Undersample majority age region to balance with minority
   - **SMOTER**: Synthetic oversampling of rare ages + undersampling of common ages
   - **Inverse Distribution**: Inverse-frequency weighted sampling across age bins
4. **Prediction**: Use context for regression via:
   - **KNN Context**: Distance-weighted average of context labels
   - **Local Ridge**: Fit Ridge regression on context neighbors per test sample
   - **ICL Transformer**: GPT2-based transformer trained to predict from in-context examples
   - **Sklearn Baselines**: Standard models (KNN, GradientBoosting, XGBoost) on all training data
5. **Evaluation**: Shot-region metrics (many/median/few) for RMSE, MAE, G-Mean

## File Structure

```
in_context/
├── __init__.py               # Package init
├── feature_extractor.py      # Extract embeddings from pretrained backbones
├── sampling_strategies.py    # Context selection strategies (vanilla, subsample, SMOTER, inverse)
├── icl_models.py             # In-context transformer + baseline prediction models
├── metrics.py                # Shot-region evaluation metrics (many/median/few)
├── benchmark.py              # Main benchmark runner
├── run_experiments.sh        # Shell script to run all experiments
└── README.md                 # This file
```

## Quick Start

### 1. Run with baselines only (fast)

```bash
python benchmark.py \
    --checkpoint ../checkpoint_resnetWang/<experiment>/ckpt.best.pth.tar \
    --data_dir ../data \
    --context_size 15 \
    --strategies vanilla,subsample,smoter,inverse
```

### 2. Run with ICL Transformer

```bash
python benchmark.py \
    --checkpoint ../checkpoint_resnetWang/<experiment>/ckpt.best.pth.tar \
    --data_dir ../data \
    --context_size 15 \
    --strategies vanilla,inverse \
    --run_transformer \
    --transformer_epochs 30 \
    --device auto
```

### 3. Run full experiment suite

```bash
bash run_experiments.sh                    # Baselines + context methods
bash run_experiments.sh --with-transformer # + ICL transformer
bash run_experiments.sh --quick            # Quick test run
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to pretrained `ckpt.best.pth.tar` |
| `--data_dir` | `../data` | Data directory with CSV + numpy files |
| `--context_size` / `-k` | 15 | Number of context neighbors per test sample |
| `--strategies` | `vanilla,subsample,smoter,inverse` | Context selection strategies |
| `--run_baselines` | True | Run sklearn baselines (KNN, GB, Ridge, XGB) |
| `--run_transformer` | False | Train and evaluate ICL transformer |
| `--backbone` | `resnet1d_wang_fds` | Feature extractor model |
| `--seed` | 0 | Random seed |

## Expected Output

Results are saved as CSV with per-region metrics:

| method | overall_l1 | many_l1 | median_l1 | few_l1 | overall_gmean | ... |
|--------|-----------|---------|-----------|--------|---------------|-----|
| knn | 8.123 | 6.45 | 7.89 | 12.34 | 6.78 | ... |
| knn_vanilla | 7.856 | 6.12 | 7.65 | 11.98 | 6.45 | ... |
| knn_inverse | 7.432 | 6.34 | 7.21 | 10.56 | 6.12 | ... |

## Note

This module is self-contained and does **not** modify any files in the parent `reproduce_paper/` directory. It only reads:
- Pretrained model checkpoints
- Data CSV and numpy files
