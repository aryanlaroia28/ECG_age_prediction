"""
In-Context Learning Models for ECG Age Prediction.

Includes:
  - GPT2-based transformer for in-context regression
  - PFN (Prior-Fitted Network) for in-context regression
  - Traditional baselines (KNN, GradientBoosting, etc.) with context-aware evaluation
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
# GPT2-based In-Context Regression Transformer
# ──────────────────────────────────────────────────────────────────────

class ICLTransformerModel(nn.Module):
    """
    Transformer (GPT2-based) that performs in-context regression.

    Input format: interleaved sequence [x1, y1, x2, y2, ..., xk, yk, x_test, 0]
    Output: prediction at the x_test position.
    """
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super().__init__()
        from transformers import GPT2Model, GPT2Config

        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleave x's and y's: [x1, y1, x2, y2, ...]"""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (ys_b.view(bsize, points, 1),
             torch.zeros(bsize, points, dim - 1, device=ys_b.device)),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain out-of-range indices")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]


# ──────────────────────────────────────────────────────────────────────
# Dataset for batching in-context test samples
# ──────────────────────────────────────────────────────────────────────

class ICLTestDataset(Dataset):
    """
    Packages context + test sample for the transformer.

    For each test sample, constructs:
      - x_context: [x_ctx_1, ..., x_ctx_k, x_test]
      - y_context: [y_ctx_1, ..., y_ctx_k, 0]
    """
    def __init__(self, X_train, X_test, y_train, y_test, indices, n_dims):
        """
        Args:
            X_train: (N_train, D) context pool features
            X_test: (N_test, D) test features
            y_train: (N_train,) context pool labels
            y_test: (N_test,) test labels (for reference)
            indices: (N_test, k) kNN indices into X_train per test sample
            n_dims: model input dimension (may pad/truncate features to this)
        """
        self.X_train = torch.as_tensor(X_train, dtype=torch.float32)
        self.X_test = torch.as_tensor(X_test, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_train, dtype=torch.float32).flatten()
        self.y_test = torch.as_tensor(y_test, dtype=torch.float32).flatten()
        self.indices = indices
        self.n_dims = n_dims
        self.k = indices.shape[1]

    def __len__(self):
        return len(self.X_test)

    def __getitem__(self, idx):
        ctx_idx = self.indices[idx]

        # Context features: (k, D)
        x_ctx = self.X_train[ctx_idx]
        # Append test sample: (k+1, D)
        x_all = torch.cat([x_ctx, self.X_test[idx].unsqueeze(0)], dim=0)

        # Context labels: (k+1,) with 0 for test position
        y_ctx = self.y_train[ctx_idx]
        y_all = torch.cat([y_ctx, torch.zeros(1)], dim=0)

        # Pad/truncate features to n_dims
        D = x_all.shape[1]
        if D < self.n_dims:
            pad = torch.zeros(x_all.shape[0], self.n_dims - D)
            x_all = torch.cat([x_all, pad], dim=1)
        elif D > self.n_dims:
            x_all = x_all[:, :self.n_dims]

        return x_all, y_all


# ──────────────────────────────────────────────────────────────────────
# Prediction with transformer
# ──────────────────────────────────────────────────────────────────────

def predict_icl_transformer(model, X_train, X_test, y_train, y_test, indices,
                            n_dims, device='cpu', batch_size=64):
    """
    Run in-context transformer prediction for all test samples.

    Returns:
        predictions: (N_test,) numpy array
    """
    dataset = ICLTestDataset(X_train, X_test, y_train, y_test, indices, n_dims)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)
    model.eval()

    predictions = []
    k = indices.shape[1]

    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, desc="ICL Transformer inference"):
            x_batch = x_batch.to(device)  # (B, k+1, n_dims)
            y_batch = y_batch.to(device)  # (B, k+1)

            pred = model(x_batch, y_batch)  # (B, k+1)
            # Take prediction at the test position (last position)
            pred_test = pred[:, -1].cpu().numpy()
            predictions.append(pred_test)

    return np.concatenate(predictions, axis=0)


# ──────────────────────────────────────────────────────────────────────
# PFN (Prior-Fitted Network) for In-Context Regression
# ──────────────────────────────────────────────────────────────────────

# Default label gain for PFN (from original IM-Context paper)
PFN_LABEL_GAIN = 0.1


def load_pfn_model(model_path, device='cpu'):
    """
    Load a pre-trained PFN model from a .pt file.

    Args:
        model_path: Path to pfn_hebo.pt
        device: torch device

    Returns:
        model: PFN TransformerModel
        model_n_dims: The model's expected feature dimension
    """
    # Ensure pfns4bo is importable (it's packaged alongside this file)
    pfns4bo_parent = os.path.dirname(os.path.abspath(__file__))
    if pfns4bo_parent not in sys.path:
        sys.path.insert(0, pfns4bo_parent)

    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    # Extract model's native feature dimension from encoder
    model_n_dims = model.encoder[1].base_encoder.weight.shape[1]

    print(f"  Loaded PFN model from {model_path}")
    print(f"  PFN model_n_dims: {model_n_dims}")
    print(f"  PFN criterion: {type(model.criterion).__name__}")

    return model, model_n_dims


class PFNTestDataset(Dataset):
    """
    Batching dataset for PFN inference (matches original BatchingTestSample).

    For each test sample, constructs context + test and chunks features
    into blocks of size in_con_dim to fit the PFN's fixed input dimension.
    Features larger than in_con_dim are split into multiple chunks and
    predictions are averaged across chunks.
    """
    def __init__(self, X_train, X_test, y_train, y_test, indices,
                 model_n_dims, in_con_dim, in_con_size, label_gain=PFN_LABEL_GAIN):
        self.X_train = torch.as_tensor(X_train, dtype=torch.float32).unsqueeze(0)
        self.X_test = torch.as_tensor(X_test, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_train, dtype=torch.float32)
        if self.y_train.dim() == 1:
            self.y_train = self.y_train.unsqueeze(-1)
        self.y_train = self.y_train.unsqueeze(0)  # (1, N_train, 1)
        self.y_test = torch.as_tensor(y_test, dtype=torch.float32)

        self.indices = indices
        self.feature_size = X_train.shape[-1]
        self.model_n_dims = model_n_dims
        self.in_con_size = in_con_size
        self.in_con_dim = in_con_dim
        self.label_gain = label_gain

        # Compute the correct padded dimension for chunking
        if self.feature_size <= in_con_dim:
            self.correct_dim = in_con_dim
        else:
            self.correct_dim = (self.feature_size // in_con_dim + 1) * in_con_dim

    def __len__(self):
        return len(self.X_test)

    def __getitem__(self, idx):
        # Build context: (1, in_con_size+1, feature_size)
        x_context = torch.cat(
            (self.X_train[:, self.indices[idx], :],
             self.X_test[idx, :].unsqueeze(0).unsqueeze(0)),
            dim=1,
        )

        # Pad to correct_dim for even chunking
        batched_x = torch.zeros(1, self.in_con_size + 1, self.correct_dim)
        batched_x[:, :, :self.feature_size] = x_context

        # Chunk: permute to (1, correct_dim, seq_len) -> reshape to (n_chunks, in_con_dim, seq_len), permute back
        batched_x = batched_x.permute(0, 2, 1).reshape(
            -1, self.in_con_dim, self.in_con_size + 1
        ).permute(0, 2, 1)

        # Pad chunk dim to model_n_dims
        plac_hold = torch.zeros(batched_x.shape[0], batched_x.shape[1], self.model_n_dims)
        plac_hold[:, :, :self.in_con_dim] = batched_x
        batched_x = plac_hold

        # Labels: (n_chunks, in_con_size+1, 1) with 0 for test position
        batched_y = torch.cat(
            (self.label_gain * self.y_train[:, self.indices[idx], :],
             torch.zeros((1, 1, 1))),
            dim=1,
        ).repeat(batched_x.shape[0], 1, 1)

        return batched_x, batched_y


def predict_pfn(model, X_train, X_test, y_train, y_test, indices,
                model_n_dims, in_con_dim=None, in_con_size=15,
                label_gain=PFN_LABEL_GAIN, device='cpu', batch_size=1):
    """
    Run PFN in-context prediction for all test samples.

    The PFN uses a BarDistribution criterion to output a discretized probability
    distribution, from which we extract the mean prediction.

    Args:
        model: Pre-trained PFN TransformerModel
        X_train: (N_train, D) context pool features
        X_test: (N_test, D) test features
        y_train: (N_train,) context pool labels (normalized)
        y_test: (N_test,) test labels (normalized, for reference)
        indices: (N_test, k) context indices per test sample
        model_n_dims: PFN model's native feature dimension
        in_con_dim: Feature chunk size (default: min(model_n_dims, feature_dim))
        in_con_size: Number of context samples per test point
        label_gain: Scaling factor for labels (default: 0.1)
        device: torch device
        batch_size: Batch size for inference (default: 1 as in original)

    Returns:
        predictions: (N_test,) numpy array of predictions (in original label scale)
    """
    if in_con_dim is None:
        in_con_dim = min(model_n_dims, X_train.shape[1])

    dataset = PFNTestDataset(
        X_train, X_test, y_train, y_test, indices,
        model_n_dims=model_n_dims, in_con_dim=in_con_dim,
        in_con_size=in_con_size, label_gain=label_gain,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="PFN inference"):
            batch_sz, n_chunks, seq_len, dim = x.shape

            # Flatten batch and chunks: (batch*chunks, seq_len, dim)
            x = x.view(-1, in_con_size + 1, model_n_dims).float().to(device)
            # PFN expects (seq_len, batch, dim) format
            x = x.permute(1, 0, 2)

            y = y.view(-1, in_con_size + 1, 1).float().to(device)
            y = y.permute(1, 0, 2)

            # PFN forward: outputs logits over BarDistribution bins
            pred_logits = model((None, x, y), single_eval_pos=in_con_size)

            # Extract mean prediction from the BarDistribution
            pred_mean = model.criterion.mean(pred_logits.softmax(-1).log_())

            # Reshape: (batch_sz, n_chunks) -> average across chunks
            pred_mean = pred_mean.view(batch_sz, n_chunks)
            pred_mean = pred_mean.mean(dim=1).detach().cpu().numpy()

            # Undo label gain
            pred_mean = pred_mean / label_gain

            predictions.append(pred_mean)

    return np.concatenate(predictions, axis=0)


# ──────────────────────────────────────────────────────────────────────
# Train the ICL Transformer on the dataset
# ──────────────────────────────────────────────────────────────────────

class ICLTrainDataset(Dataset):
    """
    Training dataset that creates random in-context learning episodes.

    Each episode: pick a random subset of k training samples as context,
    pick one additional sample as the prediction target.
    """
    def __init__(self, X_train, y_train, k, n_dims, num_episodes=10000):
        self.X = torch.as_tensor(X_train, dtype=torch.float32)
        self.y = torch.as_tensor(y_train, dtype=torch.float32).flatten()
        self.k = k
        self.n_dims = n_dims
        self.num_episodes = num_episodes
        self.n_train = len(self.y)

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        # Random context + 1 target
        all_idx = np.random.choice(self.n_train, size=self.k + 1, replace=False)
        ctx_idx = all_idx[:self.k]
        tgt_idx = all_idx[self.k]

        # Build sequence: [context..., target]
        x_ctx = self.X[ctx_idx]
        x_tgt = self.X[tgt_idx].unsqueeze(0)
        x_all = torch.cat([x_ctx, x_tgt], dim=0)  # (k+1, D)

        y_ctx = self.y[ctx_idx]
        y_tgt = self.y[tgt_idx]
        y_all = torch.cat([y_ctx, torch.zeros(1)], dim=0)  # mask target label

        # Pad/truncate
        D = x_all.shape[1]
        if D < self.n_dims:
            pad = torch.zeros(x_all.shape[0], self.n_dims - D)
            x_all = torch.cat([x_all, pad], dim=1)
        elif D > self.n_dims:
            x_all = x_all[:, :self.n_dims]

        return x_all, y_all, y_tgt


def train_icl_transformer(model, X_train, y_train, n_dims, k=15,
                          epochs=50, lr=1e-4, batch_size=64,
                          num_episodes_per_epoch=5000, device='cpu',
                          val_X=None, val_y=None, val_indices=None):
    """
    Train the in-context transformer on random episodes from the training data.

    Args:
        model: ICLTransformerModel
        X_train, y_train: Training data
        n_dims: Feature dimension for the model
        k: Context size
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Training batch size
        num_episodes_per_epoch: Random episodes per epoch
        device: torch device
        val_X, val_y, val_indices: Optional validation set for early stopping

    Returns:
        model: Trained model
        train_losses: List of per-epoch training losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_losses = []
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        dataset = ICLTrainDataset(X_train, y_train, k=k, n_dims=n_dims,
                                  num_episodes=num_episodes_per_epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch, y_target in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_target = y_target.to(device)

            pred = model(x_batch, y_batch)  # (B, k+1)
            pred_at_target = pred[:, -1]     # prediction at target position
            loss = criterion(pred_at_target, y_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        scheduler.step()

        # Validation
        val_msg = ""
        if val_X is not None and val_indices is not None:
            val_preds = predict_icl_transformer(
                model, X_train, val_X, y_train, val_y, val_indices,
                n_dims=n_dims, device=device, batch_size=batch_size
            )
            val_loss = np.mean((val_preds - val_y.flatten()) ** 2)
            val_msg = f" | Val MSE: {val_loss:.4f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train MSE: {avg_loss:.4f}{val_msg}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        print(f"  Restored best model (val MSE: {best_val_loss:.4f})")

    return model, train_losses


# ──────────────────────────────────────────────────────────────────────
# Traditional baselines with context-aware prediction
# ──────────────────────────────────────────────────────────────────────

def predict_knn_context(X_train, X_test, y_train, indices, weights='distance'):
    """
    KNN prediction: for each test sample, average labels of its context neighbors.
    """
    y_flat = np.asarray(y_train).flatten()
    predictions = []
    for i in range(len(X_test)):
        ctx_labels = y_flat[indices[i]]
        if weights == 'distance':
            ctx_features = X_train[indices[i]]
            dists = np.linalg.norm(ctx_features - X_test[i], axis=1)
            dists = np.maximum(dists, 1e-10)
            w = 1.0 / dists
            pred = np.average(ctx_labels, weights=w)
        else:
            pred = np.mean(ctx_labels)
        predictions.append(pred)
    return np.array(predictions)


def predict_sklearn_baseline(model_name, X_train, X_test, y_train, **kwargs):
    """
    Train and predict with a standard sklearn model (no context selection, uses all data).
    """
    y_flat = np.asarray(y_train).flatten()

    if model_name == 'knn':
        from sklearn.neighbors import KNeighborsRegressor
        k = kwargs.get('n_neighbors', 15)
        model = KNeighborsRegressor(n_neighbors=k, metric='cosine', n_jobs=-1)
    elif model_name == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=kwargs.get('n_estimators', 200),
            random_state=kwargs.get('random_state', 0),
            n_iter_no_change=5,
        )
    elif model_name == 'decision_tree':
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(
            max_depth=kwargs.get('max_depth', None),
            random_state=kwargs.get('random_state', 0),
        )
    elif model_name == 'mlp':
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(
            hidden_layer_sizes=kwargs.get('hidden_layers', (128, 64)),
            random_state=kwargs.get('random_state', 0),
            max_iter=500,
            n_iter_no_change=10,
        )
    elif model_name == 'xgboost':
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=kwargs.get('n_estimators', 200),
            random_state=kwargs.get('random_state', 0),
            n_jobs=-1,
        )
    elif model_name == 'lasso':
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=kwargs.get('alpha', 0.01))
    elif model_name == 'ridge':
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=kwargs.get('alpha', 1.0))
    else:
        raise ValueError(f"Unknown baseline: {model_name}")

    model.fit(X_train, y_flat)
    predictions = model.predict(X_test)
    return predictions


def predict_sklearn_with_context(model_name, X_train, X_test, y_train, indices,
                                 **kwargs):
    """
    For each test sample, fit a local sklearn model on its context neighbors only.
    This is the in-context analogue for traditional models.
    """
    y_flat = np.asarray(y_train).flatten()
    predictions = []

    for i in range(len(X_test)):
        ctx_X = X_train[indices[i]]
        ctx_y = y_flat[indices[i]]

        # For very small contexts, fallback to simple mean
        if len(ctx_y) < 3:
            predictions.append(np.mean(ctx_y))
            continue

        if model_name == 'ridge':
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=kwargs.get('alpha', 1.0))
        elif model_name == 'lasso':
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=kwargs.get('alpha', 0.01))
        elif model_name == 'knn':
            from sklearn.neighbors import KNeighborsRegressor
            k = min(kwargs.get('n_neighbors', 5), len(ctx_y) - 1)
            model = KNeighborsRegressor(n_neighbors=max(k, 1))
        else:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)

        model.fit(ctx_X, ctx_y)
        pred = model.predict(X_test[i:i + 1])
        predictions.append(pred[0])

    return np.array(predictions)
