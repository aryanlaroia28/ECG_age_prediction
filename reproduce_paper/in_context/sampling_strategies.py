"""
Adaptive Context Selection Strategies for In-Context Learning.

Implements strategies from IM-Context (TMLR 2024):
  - Vanilla: k-nearest neighbors in feature space
  - Subsample: Random undersampling of majority region
  - SMOTER: Synthetic oversampling of minority + undersampling of majority
  - Inverse Distribution: Inverse-frequency weighted sampling
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors, KernelDensity
from scipy.spatial.distance import pdist, squareform


# ──────────────────────────────────────────────────────────────────────
# Core: k-NN context selection
# ──────────────────────────────────────────────────────────────────────

def select_context_vanilla(feature_train, feature_test, k=15, metric='cosine'):
    """
    Vanilla context selection: k-nearest neighbors in feature space.

    Returns:
        indices: (n_test, k) array of training indices per test sample
    """
    neigh = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    neigh.fit(feature_train)
    _, indices = neigh.kneighbors(feature_test)
    return indices


# ──────────────────────────────────────────────────────────────────────
# Relevance-based domain splitting
# ──────────────────────────────────────────────────────────────────────

def pdf_relevance(y, bandwidth=1.0):
    """Compute relevance scores: rare samples get higher relevance."""
    y = np.asarray(y).flatten().reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(y)
    pdf_vals = np.exp(kde.score_samples(y))
    relevance = 1 - (pdf_vals - pdf_vals.min()) / (pdf_vals.max() - pdf_vals.min() + 1e-10)
    return relevance.flatten()


def split_domains(X, y, relevance, threshold=0.5):
    """Split data into normal (majority) and rare (minority) domains."""
    X, y = np.asarray(X), np.asarray(y).flatten()
    relevance = np.asarray(relevance).flatten()

    rare_idx = np.where(relevance >= threshold)[0]
    norm_idx = np.where(relevance < threshold)[0]

    return (X[norm_idx], y[norm_idx], norm_idx,
            X[rare_idx], y[rare_idx], rare_idx)


# ──────────────────────────────────────────────────────────────────────
# Strategy 1: Random Undersampling
# ──────────────────────────────────────────────────────────────────────

def context_subsample(X_train, y_train, under='balance', relevance_threshold=0.5,
                      random_state=0):
    """
    Subsample the majority domain to balance with minority domain.

    Returns:
        selected_indices: indices into original X_train
    """
    rng = np.random.RandomState(random_state)
    relevance = pdf_relevance(y_train)
    X_norm, y_norm, norm_idx, X_rare, y_rare, rare_idx = split_domains(
        X_train, y_train, relevance, relevance_threshold
    )

    if len(rare_idx) == 0 or len(rare_idx) >= len(norm_idx):
        return np.arange(len(y_train))

    if under == 'balance':
        new_norm_size = len(rare_idx)
    elif under == 'average':
        new_norm_size = int((len(rare_idx) + len(rare_idx) ** 2 / len(norm_idx)) / 2)
        new_norm_size = max(new_norm_size, 1)
    else:
        new_norm_size = int(len(norm_idx) * 0.5)

    chosen_norm = rng.choice(norm_idx, size=min(new_norm_size, len(norm_idx)), replace=False)
    selected = np.concatenate([chosen_norm, rare_idx])
    return selected


# ──────────────────────────────────────────────────────────────────────
# Strategy 2: SMOTER (Synthetic Minority Over-sampling for Regression)
# ──────────────────────────────────────────────────────────────────────

def _get_neighbors(X, k):
    """Return indices of k nearest neighbors for each sample."""
    dist_mat = squareform(pdist(X))
    order = [np.argsort(row) for row in dist_mat]
    neighbor_indices = np.array([row[1:k + 1] for row in order])
    return neighbor_indices


def _smoter_interpolate(X, y, k, size, random_state=None):
    """Generate synthetic samples by interpolating between neighbors."""
    rng = np.random.RandomState(random_state)
    neighbor_indices = _get_neighbors(X, k)
    sample_indices = rng.choice(len(y), size=size, replace=True)

    X_new, y_new = [], []
    for i in sample_indices:
        X_case, y_case = X[i], y[i]
        neighbor = rng.choice(neighbor_indices[i])
        X_neighbor, y_neighbor = X[neighbor], y[neighbor]

        rand = rng.rand() * np.ones_like(X_case)
        diff = (X_case - X_neighbor) * rand
        X_synth = X_neighbor + diff

        d1 = np.linalg.norm(X_synth - X_case)
        d2 = np.linalg.norm(X_synth - X_neighbor)
        y_synth = (d2 * y_case + d1 * y_neighbor) / (d1 + d2 + 1e-10)

        X_new.append(X_synth)
        y_new.append(y_synth)

    return np.array(X_new), np.array(y_new)


def context_smoter(X_train, y_train, k=5, over='balance', relevance_threshold=0.5,
                   random_state=0):
    """
    SMOTER: Oversample rare domain + undersample normal domain.

    Returns:
        X_new: augmented training features
        y_new: augmented training labels
        (these replace X_train, y_train for context building)
    """
    relevance = pdf_relevance(y_train)
    X_norm, y_norm, _, X_rare, y_rare, _ = split_domains(
        X_train, y_train, relevance, relevance_threshold
    )

    if len(y_rare) < 2:
        return X_train, y_train

    norm_size, rare_size = len(y_norm), len(y_rare)

    if over == 'balance':
        new_rare_size = new_norm_size = int((norm_size + rare_size) / 2)
    elif over == 'extreme':
        new_rare_size, new_norm_size = norm_size, rare_size
    elif over == 'average':
        new_rare_size = int(((norm_size + rare_size) / 2 + norm_size) / 2)
        new_norm_size = int(((norm_size + rare_size) / 2 + rare_size) / 2)
    else:
        new_rare_size = new_norm_size = int((norm_size + rare_size) / 2)

    # Oversample rare domain (split by median for low/high rare)
    y_median = np.median(y_train)
    low_mask = y_rare < y_median
    high_mask = ~low_mask

    X_rare_parts, y_rare_parts = [], []
    for mask in [low_mask, high_mask]:
        if mask.sum() >= 2:
            target_size = int(mask.sum() / rare_size * new_rare_size)
            target_size = max(target_size, mask.sum() + 1)
            X_aug, y_aug = _smoter_interpolate(
                X_rare[mask], y_rare[mask], k=min(k, mask.sum() - 1),
                size=target_size, random_state=random_state
            )
            X_rare_parts.append(X_aug)
            y_rare_parts.append(y_aug)
        elif mask.sum() > 0:
            X_rare_parts.append(X_rare[mask])
            y_rare_parts.append(y_rare[mask])

    if X_rare_parts:
        X_rare_new = np.concatenate(X_rare_parts, axis=0)
        y_rare_new = np.concatenate(y_rare_parts, axis=0)
    else:
        X_rare_new, y_rare_new = X_rare, y_rare

    # Undersample normal domain
    rng = np.random.RandomState(random_state)
    new_norm_size = min(new_norm_size, len(y_norm))
    norm_chosen = rng.choice(len(y_norm), size=new_norm_size, replace=False)
    X_norm_new = X_norm[norm_chosen]
    y_norm_new = y_norm[norm_chosen]

    X_new = np.concatenate([X_rare_new, X_norm_new], axis=0)
    y_new = np.concatenate([y_rare_new, y_norm_new], axis=0)

    return X_new, y_new


# ──────────────────────────────────────────────────────────────────────
# Strategy 3: Inverse Distribution Sampling
# ──────────────────────────────────────────────────────────────────────

def context_inverse_distribution(y_train, target_samples=None, n_bins=50,
                                 random_state=0):
    """
    Sample training indices with inverse-frequency weighting per bin.

    Returns:
        selected_indices: indices into original y_train
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y_train).flatten()

    if target_samples is None:
        target_samples = len(y)

    bins = np.linspace(y.min(), y.max(), n_bins + 1)
    bin_assignments = np.digitize(y, bins)

    bin_counts = np.bincount(bin_assignments, minlength=n_bins + 2)
    # Skip bin 0 (below min): bins range from 1 to n_bins+1
    inverse_weights = 1.0 / np.maximum(bin_counts[1:n_bins + 1], 1)
    inverse_weights *= target_samples / inverse_weights.sum()

    selected_indices = []
    for i in range(1, n_bins + 1):
        in_bin = np.where(bin_assignments == i)[0]
        if len(in_bin) == 0:
            continue
        rng.shuffle(in_bin)
        n_select = int(np.round(inverse_weights[i - 1]))
        # Allow replacement if bin is small
        if n_select > len(in_bin):
            chosen = rng.choice(in_bin, size=n_select, replace=True)
        else:
            chosen = in_bin[:n_select]
        selected_indices.extend(chosen)

    return np.array(selected_indices)


# ──────────────────────────────────────────────────────────────────────
# Unified context builder
# ──────────────────────────────────────────────────────────────────────

def build_context(feature_train, y_train, feature_test, strategy='vanilla',
                  k=15, metric='cosine', random_state=0, **kwargs):
    """
    Build context indices for each test sample using the specified strategy.

    Args:
        feature_train: (N_train, D) training features
        y_train: (N_train,) training labels
        feature_test: (N_test, D) test features
        strategy: 'vanilla' | 'subsample' | 'smoter' | 'inverse' | 'all'
        k: Number of context neighbors
        metric: Distance metric for kNN
        random_state: Random seed

    Returns:
        context_X_train: possibly augmented training features
        context_y_train: possibly augmented training labels
        indices: (N_test, k) context indices per test sample, or None for 'all'
    """
    y_flat = np.asarray(y_train).flatten()

    if strategy == 'all':
        # Use entire training set as context (no selection)
        return feature_train, y_flat, None

    if strategy == 'vanilla':
        indices = select_context_vanilla(feature_train, feature_test, k=k, metric=metric)
        return feature_train, y_flat, indices

    if strategy == 'subsample':
        selected = context_subsample(feature_train, y_flat,
                                     random_state=random_state, **kwargs)
        X_sub = feature_train[selected]
        y_sub = y_flat[selected]
        indices = select_context_vanilla(X_sub, feature_test, k=min(k, len(y_sub) - 1),
                                         metric=metric)
        return X_sub, y_sub, indices

    if strategy == 'smoter':
        X_aug, y_aug = context_smoter(feature_train, y_flat,
                                      random_state=random_state, **kwargs)
        indices = select_context_vanilla(X_aug, feature_test, k=min(k, len(y_aug) - 1),
                                         metric=metric)
        return X_aug, y_aug, indices

    if strategy == 'inverse':
        selected = context_inverse_distribution(y_flat, random_state=random_state,
                                                **kwargs)
        X_inv = feature_train[selected]
        y_inv = y_flat[selected]
        indices = select_context_vanilla(X_inv, feature_test, k=min(k, len(y_inv) - 1),
                                         metric=metric)
        return X_inv, y_inv, indices

    raise ValueError(f"Unknown strategy: {strategy}")
