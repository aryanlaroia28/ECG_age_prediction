#!/usr/bin/env python
"""
Compile results from ICL experiments across multiple seeds.

Aggregates results from multiple runs (seeds), computes statistics (mean, std),
and formats output showing performance on many/median/few age regions.
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile IM-Context results across multiple seeds"
    )
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing seed_*/results.csv files')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Output CSV path for compiled results')
    return parser.parse_args()


def load_seed_results(results_dir):
    """Load results.csv from all seed_* subdirectories."""
    results = []
    seed_dirs = sorted(glob.glob(os.path.join(results_dir, 'seed_*')))
    
    for seed_dir in seed_dirs:
        result_file = os.path.join(seed_dir, 'results.csv')
        if os.path.exists(result_file):
            seed_num = os.path.basename(seed_dir).replace('seed_', '')
            df = pd.read_csv(result_file)
            df['seed'] = seed_num
            results.append(df)
            print(f"✓ Loaded seed {seed_num}: {len(df)} method configurations")
        else:
            print(f"⚠ Missing results file: {result_file}")
    
    if not results:
        raise FileNotFoundError(f"No results.csv found in {results_dir}")
    
    combined = pd.concat(results, ignore_index=True)
    return combined


def compile_statistics(df_combined):
    """
    Compute mean and std across seeds for each method.
    
    Returns:
        DataFrame with columns: method, metrics (mean), metrics_std
    """
    # Group by method (everything except seed and run_id)
    metric_cols = [c for c in df_combined.columns 
                   if c not in ['seed', 'run_id', 'method']]
    
    grouped = df_combined.groupby('method')
    
    summary_rows = []
    for method, group in grouped:
        row = {'method': method}
        
        # Compute mean and std for each metric
        for col in metric_cols:
            if col in group.columns:
                col_data = group[col].dropna()
                if len(col_data) > 0:
                    row[f'{col}_mean'] = col_data.mean()
                    row[f'{col}_std'] = col_data.std() if len(col_data) > 1 else 0.0
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def format_results_by_region(df_combined):
    """
    Format results with separate sections for each age region (many/median/few).
    """
    metrics_by_region = {}
    
    # Identify metrics for each region
    regions = ['overall', 'many', 'median', 'few']
    
    for region in regions:
        region_metrics = {}
        region_cols = [c for c in df_combined.columns 
                       if region in c.lower() and c != 'method']
        
        for col in region_cols:
            col_data = df_combined[col].dropna()
            if len(col_data) > 0:
                region_metrics[col] = {
                    'mean': col_data.mean(),
                    'std': col_data.std() if len(col_data) > 1 else 0.0,
                    'min': col_data.min(),
                    'max': col_data.max(),
                }
        
        metrics_by_region[region] = region_metrics
    
    return metrics_by_region


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Compiling IM-Context Results")
    print("=" * 80)
    
    # Load all seed results
    print(f"\nLoading results from: {args.results_dir}")
    df_combined = load_seed_results(args.results_dir)
    print(f"\n✓ Total results loaded: {len(df_combined)} rows")
    print(f"  Unique methods: {df_combined['method'].nunique()}")
    print(f"  Seeds: {sorted(df_combined['seed'].unique())}")
    
    # Compile statistics
    print("\nCompiling statistics across seeds...")
    df_summary = compile_statistics(df_combined)
    print(f"✓ Summary compiled: {len(df_summary)} methods")
    
    # Save summary
    print(f"\nSaving results to: {args.output_csv}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # Save combined results (all seeds)
    df_combined_output = args.output_csv.replace('.csv', '_all_seeds.csv')
    df_combined.to_csv(df_combined_output, index=False)
    print(f"✓ All seeds saved: {df_combined_output}")
    
    # Save summary (mean/std)
    df_summary.to_csv(args.output_csv, index=False)
    print(f"✓ Summary saved: {args.output_csv}")
    
    # Print region-wise statistics
    print("\n" + "=" * 80)
    print("Regional Performance Summary")
    print("=" * 80)
    
    metrics_by_region = format_results_by_region(df_combined)
    for region in ['overall', 'many', 'median', 'few']:
        if region in metrics_by_region:
            print(f"\n{region.upper()}:")
            for metric, stats in metrics_by_region[region].items():
                print(f"  {metric:30} {stats['mean']:8.4f} ± {stats['std']:6.4f} "
                      f"[{stats['min']:8.4f}, {stats['max']:8.4f}]")
    
    # Print method ranking by overall L1
    print("\n" + "=" * 80)
    print("Method Ranking (by overall L1/MAE)")
    print("=" * 80)
    
    if 'overall_l1_mean' in df_summary.columns:
        df_ranked = df_summary[['method', 'overall_l1_mean', 'overall_l1_std']].copy()
        df_ranked = df_ranked.sort_values('overall_l1_mean')
        df_ranked = df_ranked.reset_index(drop=True)
        df_ranked.index = df_ranked.index + 1
        print("\nRank  Method                          L1/MAE (±std)")
        print("-" * 60)
        for idx, row in df_ranked.iterrows():
            print(f"{idx:3}   {row['method']:30} {row['overall_l1_mean']:8.4f} ± {row['overall_l1_std']:6.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Compilation complete!")
    print(f"Output files:")
    print(f"  Summary (mean/std):  {args.output_csv}")
    print(f"  All seeds data:      {df_combined_output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
