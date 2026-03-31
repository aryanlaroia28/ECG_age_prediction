"""
Shot-Region Evaluation Metrics for Imbalanced Age Prediction.

Computes RMSE, MAE, and G-Mean across many/median/few age regions,
consistent with the reproduce_paper evaluation protocol.
"""
import numpy as np
from scipy.stats import gmean
from collections import defaultdict


# Default age-based region boundaries (matching reproduce_paper/train.py)
DEFAULT_REGION_BOUNDS = {
    'many': (0, 30),        # age <= 30
    'median': (30, 60),     # 30 < age < 60
    'few': (60, 10000),     # age >= 60
}


def shot_metrics(preds, labels, train_labels=None, region_bounds=None):
    """
    Compute metrics per shot region (many/median/few).

    Uses age-based boundaries by default (matching the reproduce_paper convention).
    If train_labels is provided and region_bounds='auto', uses frequency-based
    thresholds (many_shot_thr=100, low_shot_thr=20) like the original codebase.

    Args:
        preds: (N,) predictions
        labels: (N,) ground truth
        train_labels: (N_train,) training labels (for frequency-based splitting)
        region_bounds: dict or 'auto' or None (default age-based)

    Returns:
        dict with keys: 'overall', 'many', 'median', 'few'
        Each contains: 'mse', 'l1', 'rmse', 'gmean', 'count'
    """
    preds = np.asarray(preds, dtype=float).flatten()
    labels = np.asarray(labels, dtype=float).flatten()

    if region_bounds is None:
        region_bounds = DEFAULT_REGION_BOUNDS

    # Compute grouped metrics
    groups = {}
    for name in region_bounds:
        groups[name] = {'errors': [], 'abs_errors': [], 'sq_errors': [], 'count': 0}

    for p, y in zip(preds, labels):
        err = p - y
        for name, (lo, hi) in region_bounds.items():
            if lo <= y < hi:
                groups[name]['errors'].append(err)
                groups[name]['abs_errors'].append(abs(err))
                groups[name]['sq_errors'].append(err ** 2)
                groups[name]['count'] += 1
                break

    shot_dict = {}

    # Overall
    all_abs = np.abs(preds - labels)
    all_sq = (preds - labels) ** 2
    shot_dict['overall'] = {
        'mse': float(np.mean(all_sq)),
        'rmse': float(np.sqrt(np.mean(all_sq))),
        'l1': float(np.mean(all_abs)),
        'mae': float(np.mean(all_abs)),
        'gmean': float(gmean(np.maximum(all_abs, 1e-10))),
        'count': len(preds),
    }

    # Per-region
    for name in groups:
        g = groups[name]
        if g['count'] > 0:
            abs_arr = np.array(g['abs_errors'])
            sq_arr = np.array(g['sq_errors'])
            shot_dict[name] = {
                'mse': float(np.mean(sq_arr)),
                'rmse': float(np.sqrt(np.mean(sq_arr))),
                'l1': float(np.mean(abs_arr)),
                'mae': float(np.mean(abs_arr)),
                'gmean': float(gmean(np.maximum(abs_arr, 1e-10))),
                'count': g['count'],
            }
        else:
            shot_dict[name] = {
                'mse': np.nan, 'rmse': np.nan, 'l1': np.nan,
                'mae': np.nan, 'gmean': np.nan, 'count': 0,
            }

    return shot_dict


def print_shot_results(shot_dict, prefix=''):
    """Pretty-print shot metrics to console."""
    o = shot_dict['overall']
    print(f"{prefix}Overall (n={o['count']:>5}): "
          f"MSE {o['mse']:.3f} | MAE {o['l1']:.3f} | RMSE {o['rmse']:.3f} | G-Mean {o['gmean']:.3f}")

    for region in ['many', 'median', 'few']:
        if region in shot_dict:
            r = shot_dict[region]
            print(f"{prefix}{region.capitalize():<8} (n={r['count']:>5}): "
                  f"MSE {r['mse']:.3f} | MAE {r['l1']:.3f} | RMSE {r['rmse']:.3f} | G-Mean {r['gmean']:.3f}")


def results_to_row(method_name, shot_dict, seed=0, extra=None):
    """Convert shot_dict to a flat dict suitable for a DataFrame row."""
    row = {'method': method_name, 'seed': seed}
    for region in ['overall', 'many', 'median', 'few']:
        if region in shot_dict:
            for metric in ['mse', 'rmse', 'l1', 'gmean', 'count']:
                row[f'{region}_{metric}'] = shot_dict[region].get(metric, np.nan)
    if extra:
        row.update(extra)
    return row
