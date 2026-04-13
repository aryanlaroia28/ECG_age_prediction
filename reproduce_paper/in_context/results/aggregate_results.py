#!/usr/bin/env python3
"""
Aggregate per-seed result CSVs into mean and std summary.

Usage: run after all `results_k15_seed*.csv` are present.
"""
import glob
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PATTERN = os.path.join(BASE, 'results_k15_seed*.csv')
OUT = os.path.join(BASE, 'results_k15_mean_std.csv')

def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        print('No per-seed results found.')
        return

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['__source'] = os.path.basename(f)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True, sort=False)

    # Group by method and compute mean/std for numeric columns
    group_cols = ['method']
    numeric_cols = [c for c in all_df.columns if all_df[c].dtype.kind in 'fi' and c != 'seed']
    agg_mean = all_df.groupby(group_cols)[numeric_cols].mean()
    agg_std = all_df.groupby(group_cols)[numeric_cols].std()

    # Flatten columns
    mean_df = agg_mean.reset_index()
    std_df = agg_std.reset_index()
    # Prefix std columns
    std_df = std_df.rename(columns={c: f"{c}_std" for c in std_df.columns if c not in group_cols})

    summary = pd.merge(mean_df, std_df, on=group_cols, how='left')
    summary.to_csv(OUT, index=False)
    print(f'Aggregated {len(files)} files -> {OUT}')

if __name__ == '__main__':
    main()
