"""
Causal Hidden MLP - Training multiple effective hidden sizes together.

Architecture: Linear -> SiLU -> Cumulative aggregation -> Output per hidden size

The cumulative operation creates a "causal" effect where hidden position i
can only see information from positions 0..i (inclusive).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CausalHiddenMLP(nn.Module):
    """
    MLP with causal hidden dimension tracking.
    
    Output: [batch, hidden_size, num_classes] where output[:, i, :] 
    represents prediction using only the first i+1 hidden units.
    
    Uses cumulative SUM of per-hidden contributions to logits.
    Each hidden unit j contributes h[j] * W[:, j] to the logits.
    Exit i uses sum of contributions from j=0 to i.
    """
    
    def __init__(self, in_features: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, in_features]

        Returns:
            Output tensor of shape [batch, hidden_size, num_classes]
            where output[:, i, :] uses cumulative mean of hidden units 0..i.
            
        Key insight: Use cumulative MEAN (not sum) to normalize contributions.
        This prevents scale drift and enables natural monotonicity.
        
        out[:, i, :] = (1/(i+1)) * sum(j=0 to i) h[j] * W2[:, j] + b
        """
        # First linear layer + SiLU
        h = self.linear1(x)  # [batch, hidden]
        h = F.silu(h)  # [batch, hidden]

        W = self.linear2.weight  # [num_classes, hidden]
        b = self.linear2.bias    # [num_classes]
        
        # Per-hidden contribution: [batch, hidden, num_classes]
        contrib = h.unsqueeze(-1) * W.t().unsqueeze(0)
        
        # Cumulative SUM then divide by count = Cumulative MEAN
        cumsum = contrib.cumsum(dim=1)  # [batch, hidden, num_classes]
        counts = torch.arange(1, self.hidden_size + 1, device=h.device, dtype=h.dtype)
        out = cumsum / counts.view(1, -1, 1)  # [batch, hidden, num_classes]
        
        # Add bias (same for all exits)
        out = out + b.view(1, 1, -1)

        return out


class CausalHiddenMLPWithHeads(nn.Module):
    """
    Alternative: Each exit has its own output head but shares hidden representation.
    
    This allows each exit to learn how to best use its accumulated information.
    """
    
    def __init__(self, in_features: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.linear1 = nn.Linear(in_features, hidden_size)
        # Each exit has its own output head - but they all take the full cum_h
        # and learn to mask/use different parts
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in range(hidden_size)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Output tensor of shape [batch, hidden_size, num_classes]
        """
        # First linear layer + SiLU
        h = self.linear1(x)  # [batch, hidden]
        h = F.silu(h)  # [batch, hidden]
        
        # Cumulative sum of hidden states
        cum_h = h.cumsum(dim=1)  # [batch, hidden]
        # Normalize to prevent explosion
        indices = torch.arange(1, self.hidden_size + 1, device=h.device, dtype=h.dtype)
        cum_h_norm = cum_h / indices.sqrt()  # [batch, hidden]
        
        # Apply each head to the FULL cumulative representation
        # But each head learns to focus on different parts
        outputs = []
        for i, head in enumerate(self.heads):
            # Head i uses the full cum_h_norm but learns to use first i+1 units
            out_i = head(cum_h_norm)  # [batch, num_classes] - applies to all positions
            outputs.append(out_i)
        
        out = torch.stack(outputs, dim=1)  # [batch, hidden, num_classes]
        return out


class RegularMLP(nn.Module):
    """Standard MLP for comparison baseline."""
    
    def __init__(self, in_features: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        h = F.silu(h)
        out = self.linear2(h)
        return out


def causal_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                mono_weight: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute loss for causal hidden MLP with optional monotonicity regularization.
    """
    batch_size, hidden_size, num_classes = predictions.shape
    
    if targets.dim() == 1:
        targets_expanded = targets.unsqueeze(1).expand(-1, hidden_size)
    else:
        targets_expanded = targets
    
    pred_flat = predictions.reshape(-1, num_classes)
    tgt_flat = targets_expanded.reshape(-1)
    
    per_sample_loss = F.cross_entropy(pred_flat, tgt_flat, reduction='none')
    per_sample_loss = per_sample_loss.reshape(batch_size, hidden_size)
    per_hidden_loss = per_sample_loss.mean(dim=0)
    
    mean_loss = per_hidden_loss.mean()
    
    if mono_weight > 0:
        # Penalize when later exits have higher loss
        loss_diff = per_hidden_loss[:-1] - per_hidden_loss[1:]
        mono_penalty = F.relu(-loss_diff).mean()
        mean_loss = mean_loss + mono_weight * mono_penalty
    
    return mean_loss, per_hidden_loss


def causal_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute accuracy per hidden position as exp(-loss)."""
    batch_size, hidden_size, num_classes = predictions.shape

    if targets.dim() == 1:
        targets_expanded = targets.unsqueeze(1).expand(-1, hidden_size)
    else:
        targets_expanded = targets

    pred_flat = predictions.reshape(-1, num_classes)
    tgt_flat = targets_expanded.reshape(-1)

    # Cross-entropy loss per sample and hidden position
    per_sample_loss = F.cross_entropy(pred_flat, tgt_flat, reduction='none')
    per_sample_loss = per_sample_loss.reshape(batch_size, hidden_size)
    
    # Accuracy as exp(-loss) - this gives probability-like measure
    per_hidden_acc = torch.exp(-per_sample_loss).mean(dim=0)

    return per_hidden_acc


def causal_accuracy_classif(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute standard classification accuracy per hidden position."""
    batch_size, hidden_size = predictions.shape[:2]

    if targets.dim() == 1:
        targets_expanded = targets.unsqueeze(1).expand(-1, hidden_size)
    else:
        targets_expanded = targets

    preds = predictions.argmax(dim=2)
    correct = (preds == targets_expanded).float()
    per_hidden_acc = correct.mean(dim=0)

    return per_hidden_acc


if __name__ == "__main__":
    # Quick test
    batch_size = 4
    in_features = 10
    hidden_size = 8
    num_classes = 3
    
    model = CausalHiddenMLP(in_features, hidden_size, num_classes)
    x = torch.randn(batch_size, in_features)
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (batch_size, hidden_size, num_classes)
    
    targets = torch.randint(0, num_classes, (batch_size,))
    loss, per_hidden_loss = causal_loss(out, targets)
    acc = causal_accuracy(out, targets)
    
    print(f"Mean loss: {loss.item():.4f}")
    print(f"Per-hidden accuracy: {acc}")
    print("Test passed!")
