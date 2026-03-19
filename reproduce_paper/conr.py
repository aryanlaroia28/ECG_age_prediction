# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2021-present, Yuzhe Yang
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
########################################################################################
# Code is based on the LDS and FDS (https://arxiv.org/pdf/2102.09554.pdf) implementation
# from https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir 
# by Yuzhe Yang et al.
########################################################################################
import torch
import torch.nn.functional as F
import pickle


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, beta=1., weights=None):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


# ConR loss function
def ConR(features, targets, preds, w=1, weights=1, t=0.2, e=0.01):
    """
    Contrastive Regression (ConR) loss function with numerical stability.
    
    Args:
        features: (batch_size, feature_dim) feature embeddings
        targets: (batch_size,) target labels (ground truth)
        preds: (batch_size,) predicted values
        w: window parameter for positive pairs (default: 1)
        weights: sample weights (default: 1)
        t: temperature parameter for contrastive loss (default: 0.2, overridden to 0.07)
        e: exponential coefficient for label distance weighting (default: 0.01)
    
    Returns:
        ConR loss
    """
    t = 0.07
    eps = 1e-8

    # Ensure targets and preds are 1D (batch_size,)
    targets = targets.flatten()
    preds = preds.flatten()

    q = torch.nn.functional.normalize(features, dim=1)
    k = torch.nn.functional.normalize(features, dim=1)

    l_k = targets[None, :]
    l_q = targets

    p_k = preds[None, :]
    p_q = preds
    
    # Handle weights - ensure proper shape
    if isinstance(weights, (int, float)):
        weights_tensor = torch.ones_like(targets, dtype=torch.float32)
    elif weights.dim() == 1:
        weights_tensor = weights.float()
    else:
        weights_tensor = weights.squeeze().float()

    # Proper reshape for distance calculation
    l_q_expanded = l_q.unsqueeze(-1)
    p_q_expanded = p_q.unsqueeze(-1)
    
    l_dist = torch.abs(l_q_expanded - l_k)
    p_dist = torch.abs(p_q_expanded - p_k)

    pos_i = l_dist <= w
    neg_i = ((l_dist > w) & (p_dist <= w))

    # Remove self-pairs
    for i in range(pos_i.shape[0]):
        pos_i[i, i] = False
    
    prod = torch.einsum("nc,kc->nk", [q, k]) / t
    pos = prod * pos_i.float()
    neg = prod * neg_i.float()
    
    # Clamp exponential arguments to prevent overflow/underflow
    l_dist_clamped = torch.clamp(l_dist * e, min=-10, max=10)
    pushing_w = weights_tensor.unsqueeze(-1) * torch.exp(l_dist_clamped)
    
    # Stable computation of negative exponentials
    neg_clamped = torch.clamp(neg, min=-100, max=100)
    neg_exp_dot = (pushing_w * torch.exp(neg_clamped) * neg_i.float()).sum(1)
    
    # For each query sample, if there is no negative pair, zero-out the loss
    has_neg = (neg_i.float()).sum(1) > 0
    
    # Number of positive pairs per sample
    num_pos = (pos_i.float()).sum(1)
    num_pos = torch.clamp(num_pos, min=1.0)  # Prevent division by zero
    
    # Compute loss with numerical stability
    pos_clamped = torch.clamp(pos, min=-100, max=100)
    pos_exp = torch.exp(pos_clamped)
    
    # Numerator and denominator
    numerator = pos_exp
    denominator = (pos_exp.sum(1, keepdim=True) + neg_exp_dot.unsqueeze(-1))
    
    # Clamp to prevent log(0)
    denominator = torch.clamp(denominator, min=eps)
    ratio = torch.clamp(numerator / denominator, min=eps, max=1.0)
    
    # Log and weighted sum
    loss_per_sample = (-torch.log(ratio) * pos_i.float()).sum(1) / num_pos
    
    # Zero out losses where no negative pairs exist
    loss_per_sample = loss_per_sample * has_neg.float()
    
    # Final weighted mean
    loss = (weights_tensor * loss_per_sample).mean()
    
    # Safeguard against NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=features.device, dtype=features.dtype, requires_grad=True)
    
    return loss
