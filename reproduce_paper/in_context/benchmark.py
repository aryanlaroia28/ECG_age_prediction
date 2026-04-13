"""
Main Benchmark Script: In-Context Learning for ECG Age Prediction.

Workflow:
  1. Extract features from pretrained backbone (ResNet1D / Inception1D)
  2. Normalize features (StandardScaler + PowerTransformer)
  3. Apply context selection strategies (vanilla, subsample, SMOTER, inverse)
  4. Predict with in-context models (transformer, KNN, sklearn baselines)
  5. Evaluate with shot-region metrics (many / median / few)
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Local imports
from feature_extractor import extract_and_cache_features
from sampling_strategies import build_context, select_context_vanilla
from icl_models import (
    ICLTransformerModel,
    train_icl_transformer,
    predict_icl_transformer,
    predict_knn_context,
    predict_sklearn_baseline,
    predict_sklearn_with_context,
    load_pfn_model,
    predict_pfn,
    PFN_LABEL_GAIN,
)
from metrics import shot_metrics, print_shot_results, results_to_row


def normalize_features(X_train, X_test, X_val=None):
    """
    Normalize with StandardScaler + PowerTransformer, then concatenate.
    This doubles feature dimensions but captures non-linear relationships.
    """
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    X_val_std = std_scaler.transform(X_val) if X_val is not None else None

    pt_scaler = PowerTransformer(method='yeo-johnson')
    X_train_pt = pt_scaler.fit_transform(X_train)
    X_test_pt = pt_scaler.transform(X_test)
    X_val_pt = pt_scaler.transform(X_val) if X_val is not None else None

    X_train_combined = np.concatenate([X_train_std, X_train_pt], axis=1)
    X_test_combined = np.concatenate([X_test_std, X_test_pt], axis=1)
    X_val_combined = np.concatenate([X_val_std, X_val_pt], axis=1) if X_val is not None else None

    # Also normalize labels for transformer training
    y_scaler = StandardScaler()

    return X_train_combined, X_test_combined, X_val_combined, y_scaler


def run_benchmark(args):
    """Main benchmark runner."""
    print("=" * 100)
    print("In-Context Learning for ECG Age Prediction")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    device = torch.device(args.device if args.device != 'auto' else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # ── Step 1: Extract features from pretrained backbone ──
    print("\n[Step 1] Extracting features from pretrained backbone...")
    data = extract_and_cache_features(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        model_name=args.backbone,
        cache_dir=os.path.join(args.output_dir, 'cached_features'),
        device=str(device),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    X_train_raw = data['train']['features']
    y_train = data['train']['labels']
    X_val_raw = data['val']['features']
    y_val = data['val']['labels']
    X_test_raw = data['test']['features']
    y_test = data['test']['labels']

    print(f"  Train: {X_train_raw.shape} | Val: {X_val_raw.shape} | Test: {X_test_raw.shape}")
    print(f"  Feature dim: {X_train_raw.shape[1]}")
    print(f"  Age range: train [{y_train.min():.0f}-{y_train.max():.0f}], "
          f"test [{y_test.min():.0f}-{y_test.max():.0f}]")

    # ── Step 2: Normalize features ──
    print("\n[Step 2] Normalizing features (StandardScaler + PowerTransformer)...")
    X_train, X_test, X_val, y_scaler = normalize_features(X_train_raw, X_test_raw, X_val_raw)
    print(f"  Combined feature dim: {X_train.shape[1]}")

    # Normalize labels for transformer
    y_scaler.fit(y_train.reshape(-1, 1))
    y_train_norm = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_norm = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_norm = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # ── Step 3-5: Run methods ──
    results = []
    k = args.context_size

    # Define methods to run
    methods = []

    # Sklearn baselines (no context selection — use all training data)
    if args.run_baselines:
        for baseline in ['knn', 'gradient_boosting', 'ridge', 'xgboost']:
            methods.append(('sklearn', baseline, 'none'))

    # Context-based methods
    strategies = args.strategies.split(',')
    for strategy in strategies:
        strategy = strategy.strip()
        if not strategy:
            continue
        # KNN with context
        methods.append(('knn_context', 'knn', strategy))
        # Local Ridge with context
        methods.append(('local_ridge', 'ridge', strategy))

    # Transformer methods
    if args.run_transformer:
        for strategy in strategies:
            strategy = strategy.strip()
            if not strategy:
                continue
            methods.append(('transformer', 'gpt2', strategy))

    # PFN methods
    if args.run_pfn:
        for strategy in strategies:
            strategy = strategy.strip()
            if not strategy:
                continue
            methods.append(('pfn', 'pfn_bo', strategy))
        # PFN with all context
        methods.append(('pfn', 'pfn_bo', 'all'))

    print(f"\n[Step 3-5] Running {len(methods)} method configurations...")
    print("-" * 100)

    for method_idx, (method_type, model_name, strategy) in enumerate(methods, 1):
        method_label = f"{model_name}_{strategy}" if strategy != 'none' else model_name
        print(f"\n[{method_idx}/{len(methods)}] {method_label}")
        print(f"  Type: {method_type} | Strategy: {strategy}")

        try:
            if method_type == 'sklearn':
                # Standard sklearn baseline — no context selection
                preds_norm = predict_sklearn_baseline(
                    model_name, X_train, X_test, y_train_norm,
                    random_state=args.seed, n_neighbors=k,
                )
                preds = y_scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()

            elif method_type in ('knn_context', 'local_ridge'):
                # Context-based prediction
                ctx_X, ctx_y, indices = build_context(
                    X_train, y_train_norm, X_test,
                    strategy=strategy, k=k, random_state=args.seed,
                )
                if indices is None:
                    # 'all' strategy: use all data with vanilla KNN
                    indices = select_context_vanilla(ctx_X, X_test, k=k)

                if method_type == 'knn_context':
                    preds_norm = predict_knn_context(
                        ctx_X, X_test, ctx_y, indices, weights='distance'
                    )
                else:
                    preds_norm = predict_sklearn_with_context(
                        'ridge', ctx_X, X_test, ctx_y, indices
                    )
                preds = y_scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()

            elif method_type == 'transformer':
                # Build context
                ctx_X, ctx_y, indices = build_context(
                    X_train, y_train_norm, X_test,
                    strategy=strategy, k=k, random_state=args.seed,
                )
                if indices is None:
                    indices = select_context_vanilla(ctx_X, X_test, k=k)

                # Also prepare validation context for early stopping
                _, _, val_indices = build_context(
                    ctx_X, ctx_y, X_val,
                    strategy='vanilla', k=k, random_state=args.seed,
                )

                n_dims = ctx_X.shape[1]

                # Build and train transformer
                model = ICLTransformerModel(
                    n_dims=n_dims,
                    n_positions=k + 2,
                    n_embd=args.n_embd,
                    n_layer=args.n_layer,
                    n_head=args.n_head,
                )
                print(f"  Training ICL Transformer (n_dims={n_dims}, "
                      f"n_embd={args.n_embd}, n_layer={args.n_layer})...")

                model, losses = train_icl_transformer(
                    model, ctx_X, ctx_y, n_dims=n_dims, k=k,
                    epochs=args.transformer_epochs, lr=args.transformer_lr,
                    batch_size=args.transformer_batch_size,
                    num_episodes_per_epoch=args.episodes_per_epoch,
                    device=str(device),
                    val_X=X_val, val_y=y_val_norm, val_indices=val_indices,
                )

                # Save trained transformer
                save_path = os.path.join(
                    args.output_dir, 'models',
                    f"icl_transformer_{strategy}_k{k}_seed{args.seed}.pt"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"  Saved model to {save_path}")

                # Predict
                preds_norm = predict_icl_transformer(
                    model, ctx_X, X_test, ctx_y, y_test_norm, indices,
                    n_dims=n_dims, device=str(device), batch_size=args.batch_size,
                )
                preds = y_scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()

            elif method_type == 'pfn':
                # PFN (Prior-Fitted Network) in-context prediction
                pfn_model, pfn_n_dims = load_pfn_model(
                    args.pfn_model_path, device=str(device)
                )
                pfn_in_con_dim = min(pfn_n_dims, X_train.shape[1])

                if strategy == 'all':
                    # Use entire training set as context
                    pfn_k = min(X_train.shape[0], 10000)
                    if X_train.shape[0] > 10000:
                        print(f"  WARNING: Training set too large ({X_train.shape[0]}), "
                              f"skipping 'all' strategy.")
                        continue
                    ctx_X = X_train
                    ctx_y = y_train_norm
                    indices = select_context_vanilla(ctx_X, X_test, k=pfn_k)
                elif strategy == 'inverse':
                    # Original paper: concatenate vanilla kNN + inverse-sampled kNN
                    # Both index into the full X_train
                    ctx_X = X_train
                    ctx_y = y_train_norm
                    vanilla_indices = select_context_vanilla(X_train, X_test, k=k)

                    from sampling_strategies import context_inverse_distribution
                    selected = context_inverse_distribution(
                        y_train_norm, random_state=args.seed,
                    )
                    inv_knn_indices = select_context_vanilla(
                        X_train[selected], X_test,
                        k=min(k, len(selected) - 1),
                    )
                    # Map back to original X_train indices
                    inv_knn_indices = np.array(selected)[inv_knn_indices]
                    indices = np.concatenate([vanilla_indices, inv_knn_indices], axis=1)
                    pfn_k = indices.shape[1]
                else:
                    ctx_X, ctx_y, indices = build_context(
                        X_train, y_train_norm, X_test,
                        strategy=strategy, k=k, random_state=args.seed,
                    )
                    pfn_k = k
                    if indices is None:
                        indices = select_context_vanilla(ctx_X, X_test, k=pfn_k)

                print(f"  PFN context: {ctx_X.shape[0]} samples, "
                      f"k={pfn_k}, in_con_dim={pfn_in_con_dim}")

                preds_norm = predict_pfn(
                    pfn_model, ctx_X, X_test, ctx_y, y_test_norm, indices,
                    model_n_dims=pfn_n_dims, in_con_dim=pfn_in_con_dim,
                    in_con_size=pfn_k, label_gain=PFN_LABEL_GAIN,
                    device=str(device), batch_size=args.pfn_batch_size,
                )
                preds = y_scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()

            else:
                raise ValueError(f"Unknown method type: {method_type}")

            # Evaluate
            shot_dict = shot_metrics(preds, y_test)
            print_shot_results(shot_dict, prefix="  ")
            results.append(results_to_row(method_label, shot_dict, seed=args.seed,
                                          extra={'strategy': strategy, 'model': model_name,
                                                 'k': k}))

            # Incremental save after each method
            csv_path = os.path.join(args.output_dir, f"results_k{k}_seed{args.seed}.csv")
            pd.DataFrame(results).to_csv(csv_path, index=False)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Summary ──
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        display_cols = ['method', 'overall_mse', 'overall_l1', 'overall_gmean',
                        'many_l1', 'median_l1', 'few_l1',
                        'many_gmean', 'median_gmean', 'few_gmean']
        existing_cols = [c for c in display_cols if c in results_df.columns]
        print(results_df[existing_cols].to_string(index=False, float_format='%.3f'))

        # Final save
        csv_path = os.path.join(args.output_dir, f"results_k{k}_seed{args.seed}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
    else:
        print("No results collected.")

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="In-Context Learning Benchmark for ECG Age Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data & model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained backbone checkpoint (ckpt.best.pth.tar)')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to data directory with CSV + numpy files')
    parser.add_argument('--dataset', type=str, default='1000_timesteps',
                        choices=['1000_timesteps', '5000_timesteps'])
    parser.add_argument('--backbone', type=str, default='resnet1d_wang_fds',
                        choices=['resnet1d_wang_fds', 'inception1d_fds'])
    parser.add_argument('--output_dir', type=str, default='./in_context/results',
                        help='Output directory for results')

    # Context selection
    parser.add_argument('--context_size', '-k', type=int, default=15,
                        help='Number of context samples per test sample')
    parser.add_argument('--strategies', type=str, default='vanilla,subsample,smoter,inverse',
                        help='Comma-separated context strategies')

    # What to run
    parser.add_argument('--run_baselines', action='store_true', default=True,
                        help='Run sklearn baselines (KNN, GB, Ridge, XGBoost)')
    parser.add_argument('--no_baselines', dest='run_baselines', action='store_false')
    parser.add_argument('--run_transformer', action='store_true', default=False,
                        help='Train and evaluate ICL transformer')
    parser.add_argument('--run_pfn', action='store_true', default=False,
                        help='Run PFN (Prior-Fitted Network) in-context prediction')
    parser.add_argument('--pfn_model_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'models', 'pfn_hebo.pt'),
                        help='Path to pre-trained PFN model (pfn_hebo.pt)')
    parser.add_argument('--pfn_batch_size', type=int, default=1,
                        help='Batch size for PFN inference (1 recommended)')

    # Transformer hyperparams
    parser.add_argument('--n_embd', type=int, default=128, help='Transformer embedding dim')
    parser.add_argument('--n_layer', type=int, default=6, help='Transformer layers')
    parser.add_argument('--n_head', type=int, default=4, help='Transformer attention heads')
    parser.add_argument('--transformer_epochs', type=int, default=30,
                        help='Transformer training epochs')
    parser.add_argument('--transformer_lr', type=float, default=1e-4,
                        help='Transformer learning rate')
    parser.add_argument('--transformer_batch_size', type=int, default=64)
    parser.add_argument('--episodes_per_epoch', type=int, default=5000,
                        help='Random episodes per training epoch')

    # General
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for feature extraction')
    parser.add_argument('--workers', type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Save config
    config_path = os.path.join(args.output_dir, f"config_k{args.context_size}_seed{args.seed}.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    run_benchmark(args)
