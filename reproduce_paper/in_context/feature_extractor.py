"""
Feature Extractor for ECG signals using pretrained ResNet1D / Inception1D backbones.

Extracts fixed-dimensional embeddings from raw ECG signals, which are then
used as tabular features for the in-context learning pipeline.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path so we can import the original models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resnet1d_wang_fds import resnet1d_wang_fds
from datasets import PTBXLDataset


def load_pretrained_backbone(checkpoint_path, model_name='resnet1d_wang_fds',
                             input_channels=12, device='cpu'):
    """
    Load a pretrained model and return it in eval mode.

    Args:
        checkpoint_path: Path to ckpt.best.pth.tar
        model_name: 'resnet1d_wang_fds' or 'inception1d_fds'
        input_channels: Number of ECG channels (12 for standard)
        device: torch device

    Returns:
        model: Loaded model in eval mode (on CPU for feature extraction)
    """
    if model_name == 'resnet1d_wang_fds':
        model = resnet1d_wang_fds(input_channels=input_channels)
    elif model_name == 'inception1d_fds':
        from inception1d_fds import inception1d_fds
        model = inception1d_fds(input_channels=input_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load checkpoint (handle DataParallel wrapping)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']

    # Strip 'module.' prefix if saved with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Loaded pretrained {model_name} from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}, Best loss: {checkpoint.get('best_loss', 'N/A')}")

    return model


class FeatureExtractorHook:
    """
    Hooks into the model's global average pooling layer to capture
    intermediate embeddings (128-dim for ResNet1D).
    """
    def __init__(self, model, model_name='resnet1d_wang_fds'):
        self.model = model
        self.features = None

        # Register forward hook on the layer before the final linear head
        if model_name == 'resnet1d_wang_fds':
            self._hook = model.global_pool.register_forward_hook(self._hook_fn)
        elif model_name == 'inception1d_fds':
            self._hook = model.gap.register_forward_hook(self._hook_fn)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _hook_fn(self, module, input, output):
        self.features = output.squeeze(-1).detach()

    def remove(self):
        self._hook.remove()


def extract_features(model, dataloader, device='cpu', model_name='resnet1d_wang_fds'):
    """
    Extract backbone features for all samples in the dataloader.

    Args:
        model: Pretrained model in eval mode
        dataloader: DataLoader yielding (ecg_tensor, label, weight) tuples
        device: torch device
        model_name: Model architecture name

    Returns:
        features: np.ndarray (N, feature_dim)
        labels: np.ndarray (N,)
    """
    hook = FeatureExtractorHook(model, model_name)
    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, targets, _ in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            _ = model(inputs)  # forward pass triggers the hook
            all_features.append(hook.features.cpu().numpy())
            all_labels.append(targets.numpy().flatten())

    hook.remove()

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def extract_and_cache_features(checkpoint_path, data_dir, dataset_name='1000_timesteps',
                               model_name='resnet1d_wang_fds', cache_dir=None,
                               device='cpu', batch_size=256, num_workers=4):
    """
    Extract features from a pretrained model and cache them to disk.

    Args:
        checkpoint_path: Path to best checkpoint
        data_dir: Path to data directory with CSV + numpy files
        dataset_name: '1000_timesteps' or '5000_timesteps'
        model_name: Model architecture
        cache_dir: Where to save cached features (default: in_context/cached_features/)
        device: torch device
        batch_size: Batch size for extraction
        num_workers: DataLoader workers

    Returns:
        dict with keys 'train', 'val', 'test', each containing
        {'features': np.ndarray, 'labels': np.ndarray}
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cached_features')
    os.makedirs(cache_dir, exist_ok=True)

    # Derive a cache key from the checkpoint
    ckpt_basename = os.path.basename(os.path.dirname(checkpoint_path))
    cache_file = os.path.join(cache_dir, f"{ckpt_basename}_{dataset_name}_features.npz")

    # Return cached if exists
    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        data = np.load(cache_file)
        return {
            'train': {'features': data['train_features'], 'labels': data['train_labels']},
            'val': {'features': data['val_features'], 'labels': data['val_labels']},
            'test': {'features': data['test_features'], 'labels': data['test_labels']},
        }

    # Load model
    model = load_pretrained_backbone(checkpoint_path, model_name, device=device)

    # Load data splits
    csv_path = os.path.join(data_dir, f"{dataset_name}.csv")
    df = pd.read_csv(csv_path)

    result = {}
    arrays_to_save = {}

    for split in ['train', 'val', 'test']:
        df_split = df[df['split'] == split]
        dataset = PTBXLDataset(data_dir=data_dir, df=df_split, split=split)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

        features, labels = extract_features(model, loader, device=device,
                                            model_name=model_name)
        result[split] = {'features': features, 'labels': labels}
        arrays_to_save[f'{split}_features'] = features
        arrays_to_save[f'{split}_labels'] = labels

        print(f"  {split}: {features.shape[0]} samples, {features.shape[1]}-dim features")

    # Cache to disk
    np.savez(cache_file, **arrays_to_save)
    print(f"Cached features to {cache_file}")

    return result


if __name__ == '__main__':
    """Quick test of feature extraction."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to ckpt.best.pth.tar')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='1000_timesteps')
    parser.add_argument('--model', type=str, default='resnet1d_wang_fds')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    data = extract_and_cache_features(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        model_name=args.model,
        device=args.device,
    )

    for split in ['train', 'val', 'test']:
        f = data[split]['features']
        l = data[split]['labels']
        print(f"{split}: features {f.shape}, labels {l.shape}, "
              f"age range [{l.min():.0f}, {l.max():.0f}]")
