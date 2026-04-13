#!/usr/bin/env python3
"""
Compute and plot error vs age (5-year bins) for each saved run checkpoint.
Saves CSV with per-bin MAE and a PNG bar plot to `sanity_plots/`.
"""
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'sanity_plots')
os.makedirs(OUT, exist_ok=True)

# checkpoint dirs to scan (same convention as sanity_check.py)
CHECKPOINT_DIRS = [
    'checkpoint_resnetWang',
    'checkpoint_inception',
    'checkpoint_conr',
    'checkpoint_conr_ranksim',
    'checkpoint_ranksim',
]

def load_csv(dataset_csv_path):
    df = pd.read_csv(dataset_csv_path)
    df_test = df[df['split'] == 'test'].reset_index(drop=True)
    return df_test

def make_bins():
    # bins edges 0,5,10,...,100 -> labels like '0-5','5-10',..., '95-100'
    edges = list(range(0, 101, 5))
    labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)]
    return edges, labels

def try_load_model(run_name, ckpt_path, device):
    # Lazy import models to avoid breaking if not present
    from resnet1d_wang_fds import resnet1d_wang_fds
    from inception1d_fds import inception1d_fds

    # choose backbone from run name heuristic
    backbone = 'inception' if 'inception' in run_name.lower() else 'resnet'
    use_fds = '_fds_' in run_name.lower()
    # default FDS params (will work for most reproduced runs)
    fds_kwargs = dict(
        fds=use_fds,
        bucket_num=100,
        bucket_start=3,
        start_update=0,
        start_smooth=1,
        kernel='gaussian',
        ks=9,
        sigma=1.0,
        momentum=0.9,
    )

    if backbone == 'inception':
        model = inception1d_fds(input_channels=12, **fds_kwargs)
    else:
        model = resnet1d_wang_fds(input_channels=12, **fds_kwargs)

    # Wrap with DataParallel if checkpoint has module. prefixes
    state = torch.load(ckpt_path, map_location='cpu')
    state_dict = state.get('state_dict', state)
    try:
        # remove possible module prefixes if model not wrapped
        model.load_state_dict(state_dict, strict=False)
    except Exception:
        # try stripping 'module.' from keys
        new_state = {}
        for k, v in state_dict.items():
            nk = k.replace('module.', '')
            new_state[nk] = v
        try:
            model.load_state_dict(new_state, strict=False)
        except Exception as e:
            print(f"Warning: loading checkpoint for {run_name} failed: {e}")
            return None

    model.to(device)
    model.eval()
    return model

def run_inference(model, test_loader, device):
    preds, labels = [], []
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.extend(outputs.detach().cpu().squeeze().numpy().tolist())
            labels.extend(targets.detach().cpu().squeeze().numpy().tolist())
    return np.array(preds).astype(float), np.array(labels).astype(float)

def compute_bin_mae(preds, labels, edges, labels_text):
    ages = labels
    abs_err = np.abs(preds - labels)
    df = pd.DataFrame({'age': ages, 'abs_err': abs_err})
    df['age_bin'] = pd.cut(df['age'], bins=edges, labels=labels_text, include_lowest=True, right=False)
    summary = df.groupby('age_bin').agg(count=('abs_err','count'), mae=('abs_err','mean')).reset_index()
    # ensure all bins present
    summary = summary.set_index('age_bin').reindex(labels_text).reset_index()
    return summary

def plot_bin_mae(summary, run_name, out_dir):
    fig, ax = plt.subplots(figsize=(12,4))
    x = summary['age_bin'].astype(str)
    y = summary['mae'].fillna(0)
    counts = summary['count'].fillna(0).astype(int)
    ax.bar(x, y, color='C0', alpha=0.8)
    ax.set_xlabel('Age bin')
    ax.set_ylabel('MAE (years)')
    ax.set_title(f'Error vs Age (MAE) — {run_name}')
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha='right', fontsize=8)
    for i, c in enumerate(counts):
        ax.text(i, y.iloc[i] + 0.02, str(c), ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'error_vs_age_{run_name}.png'), dpi=150)
    plt.close()

def main():
    dataset_csv = os.path.join(DATA_DIR, '1000_timesteps.csv')
    if not os.path.isfile(dataset_csv):
        print(f"Dataset CSV not found: {dataset_csv}")
        return
    df_test = load_csv(dataset_csv)

    from datasets import PTBXLDataset
    edges, labels_text = make_bins()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for ckpt_dir_name in CHECKPOINT_DIRS:
        ckpt_dir = os.path.join(BASE, ckpt_dir_name)
        if not os.path.isdir(ckpt_dir):
            continue
        for run in sorted(os.listdir(ckpt_dir)):
            run_dir = os.path.join(ckpt_dir, run)
            # prefer best checkpoint
            ckpt_best = os.path.join(run_dir, 'ckpt.best.pth.tar')
            ckpt = ckpt_best if os.path.isfile(ckpt_best) else os.path.join(run_dir, 'ckpt.pth.tar')
            if not os.path.isfile(ckpt):
                continue

            print(f"Processing {run}...")

            # create test dataset/loader using df_test rows that belong to this run's dataset
            # (train.py stores dataset csv in data folder; here we reuse the global test split)
            test_dataset = PTBXLDataset(df=df_test, data_dir=DATA_DIR, split='test')
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

            model = try_load_model(run, ckpt, device)
            if model is None:
                print(f"Skipping {run} due to model load failure")
                continue

            preds, labels = run_inference(model, test_loader, device)
            summary = compute_bin_mae(preds, labels, edges, labels_text)

            # save CSV and plot
            out_csv = os.path.join(OUT, f'error_vs_age_{run}.csv')
            summary.to_csv(out_csv, index=False)
            plot_bin_mae(summary, run, OUT)
            print(f"Saved {out_csv} and plot for {run}")


if __name__ == '__main__':
    main()
