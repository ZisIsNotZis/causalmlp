"""
Training script for Causal Hidden MLP experiments.

Compares:
1. Causal Hidden MLP - outputs predictions at each hidden size
2. Regular MLP - standard baseline

Dataset: Synthetic classification (fast experimentation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

from causal_mlp import CausalHiddenMLP, CausalHiddenMLPWithHeads, RegularMLP, causal_loss, causal_accuracy


def get_synthetic_loaders(batch_size: int = 128, n_samples: int = 10000, 
                          in_features: int = 64, num_classes: int = 10):
    """Generate synthetic classification dataset."""
    torch.manual_seed(42)
    
    # Generate random features and labels
    X = torch.randn(n_samples, in_features)
    # Create a learnable pattern: some features are informative
    true_weights = torch.randn(in_features, num_classes) * 0.5
    # Make first 20% of features more informative
    true_weights[:in_features//5, :] *= 3
    y = (X @ true_weights).argmax(dim=1)
    
    # Split train/test
    n_train = int(n_samples * 0.9)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, in_features, num_classes


def train_causal_mlp(model: CausalHiddenMLP, train_loader: DataLoader,
                     optimizer: optim.Optimizer, device: torch.device, epoch: int,
                     scaler: torch.cuda.amp.GradScaler = None) -> tuple:
    """Train causal MLP for one epoch."""
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss, _ = causal_loss(output, target, mono_weight=0.1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss, _ = causal_loss(output, target, mono_weight=0.1)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate_causal_mlp(model: CausalHiddenMLP, test_loader: DataLoader,
                        device: torch.device) -> tuple:
    """Evaluate causal MLP and return per-hidden-size metrics."""
    model.eval()
    total_loss = 0.0
    all_accuracies = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            with torch.cuda.amp.autocast():
                output = model(data)
            loss, per_hidden_loss = causal_loss(output.float(), target, mono_weight=0.0)
            acc = causal_accuracy(output.float(), target)

            total_loss += loss.item()
            all_accuracies.append(acc.cpu())

    avg_loss = total_loss / len(test_loader)
    avg_per_hidden_acc = torch.stack(all_accuracies).mean(dim=0)  # [hidden_size]

    return avg_loss, avg_per_hidden_acc.numpy()


def train_regular_mlp(model: RegularMLP, train_loader: DataLoader,
                      optimizer: optim.Optimizer, device: torch.device, epoch: int) -> float:
    """Train regular MLP for one epoch."""
    model.train()
    total_loss = 0.0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate_regular_mlp(model: RegularMLP, test_loader: DataLoader, 
                         device: torch.device) -> tuple:
    """Evaluate regular MLP."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(test_loader), correct / total


def run_experiment(hidden_size: int = 128, num_epochs: int = 20, 
                   learning_rate: float = 1e-3, batch_size: int = 128,
                   device: str = None, model_variant: str = 'cumsum'):
    """Run full experiment comparing causal vs regular MLP."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    print(f"Using device: {device}")
    print(f"Model variant: {model_variant}")
    print(f"Hidden size: {hidden_size}, Epochs: {num_epochs}, LR: {learning_rate}")
    print("-" * 60)
    
    # Load synthetic data
    train_loader, test_loader, in_features, num_classes = get_synthetic_loaders(
        batch_size=batch_size, in_features=64, num_classes=10)
    
    print(f"Input features: {in_features}, Classes: {num_classes}")
    print("-" * 60)
    
    # Initialize models
    if model_variant == 'cumsum':
        causal_model = CausalHiddenMLP(in_features, hidden_size, num_classes).to(device)
    else:
        causal_model = CausalHiddenMLPWithHeads(in_features, hidden_size, num_classes).to(device)
    
    regular_model = RegularMLP(in_features, hidden_size, num_classes).to(device)
    
    # Count parameters
    causal_params = sum(p.numel() for p in causal_model.parameters())
    regular_params = sum(p.numel() for p in regular_model.parameters())
    print(f"Causal MLP parameters: {causal_params:,}")
    print(f"Regular MLP parameters: {regular_params:,}")
    print("-" * 60)
    
    # Optimizers
    causal_optimizer = optim.Adam(causal_model.parameters(), lr=learning_rate)
    regular_optimizer = optim.Adam(regular_model.parameters(), lr=learning_rate)
    
    # Mixed precision scaler for CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Training tracking
    causal_train_losses = []
    causal_test_losses = []
    causal_test_accuracies = []  # List of per-hidden-size accuracies
    
    regular_train_losses = []
    regular_test_losses = []
    regular_test_accuracies = []
    
    best_causal_acc = 0
    best_regular_acc = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()

        # Train causal model
        causal_train_loss = train_causal_mlp(causal_model, train_loader,
                                             causal_optimizer, device, epoch, scaler)
        
        # Train regular model (for comparison only, not jointly)
        regular_train_loss = train_regular_mlp(regular_model, train_loader,
                                               regular_optimizer, device, epoch)
        
        # Evaluate
        causal_test_loss, causal_per_hidden_acc = evaluate_causal_mlp(
            causal_model, test_loader, device)
        regular_test_loss, regular_test_acc = evaluate_regular_mlp(
            regular_model, test_loader, device)
        
        # Track metrics
        causal_train_losses.append(causal_train_loss)
        causal_test_losses.append(causal_test_loss)
        causal_test_accuracies.append(causal_per_hidden_acc)
        
        regular_train_losses.append(regular_train_loss)
        regular_test_losses.append(regular_test_loss)
        regular_test_accuracies.append(regular_test_acc)
        
        best_causal_acc = max(best_causal_acc, causal_per_hidden_acc.max())
        best_regular_acc = max(best_regular_acc, regular_test_acc)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Causal Loss: {causal_train_loss:.4f} | "
              f"Causal Acc (max): {causal_per_hidden_acc.max():.4f} | "
              f"Regular Loss: {regular_train_loss:.4f} | "
              f"Regular Acc: {regular_test_acc:.4f} | "
              f"Time: {elapsed:.1f}s")
    
    print("-" * 60)
    print("Training complete!")
    print(f"Best Causal MLP accuracy (any hidden size): {best_causal_acc:.4f}")
    print(f"Best Regular MLP accuracy: {best_regular_acc:.4f}")
    
    # Final per-hidden-size analysis
    final_causal_acc = causal_test_accuracies[-1]
    print("\nFinal per-hidden-size accuracy (Causal MLP):")
    print(f"  Min: {final_causal_acc.min():.4f} (hidden=1)")
    print(f"  Max: {final_causal_acc.max():.4f} (hidden={hidden_size})")
    
    # Check if accuracy is monotonically increasing
    is_increasing = all(final_causal_acc[i] <= final_causal_acc[i+1] 
                        for i in range(len(final_causal_acc)-1))
    print(f"  Monotonically increasing: {is_increasing}")
    
    # Compute performance gap
    final_regular_acc = regular_test_accuracies[-1]
    causal_final_max = final_causal_acc.max()
    performance_gap = final_regular_acc - causal_final_max
    print(f"\nPerformance comparison (final epoch):")
    print(f"  Regular MLP: {final_regular_acc:.4f}")
    print(f"  Causal MLP (best hidden): {causal_final_max:.4f}")
    print(f"  Gap (Regular - Causal): {performance_gap:.4f}")
    
    # Plot results
    plot_results(
        causal_train_losses, causal_test_losses, causal_test_accuracies,
        regular_train_losses, regular_test_losses, regular_test_accuracies,
        hidden_size
    )
    
    return {
        'causal_per_hidden_acc': final_causal_acc,
        'regular_acc': final_regular_acc,
        'causal_train_losses': causal_train_losses,
        'causal_test_losses': causal_test_losses,
        'causal_test_accuracies': causal_test_accuracies,
        'regular_train_losses': regular_train_losses,
        'regular_test_losses': regular_test_losses,
        'regular_test_accuracies': regular_test_accuracies,
        'is_monotonic': is_increasing,
        'performance_gap': performance_gap
    }


def plot_results(causal_train_losses, causal_test_losses, causal_test_accuracies,
                 regular_train_losses, regular_test_losses, regular_test_accuracies,
                 hidden_size: int):
    """Plot training results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training losses
    ax = axes[0, 0]
    ax.plot(causal_train_losses, label='Causal MLP Train', alpha=0.7)
    ax.plot(regular_train_losses, label='Regular MLP Train', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Test losses
    ax = axes[0, 1]
    ax.plot(causal_test_losses, label='Causal MLP', alpha=0.7)
    ax.plot(regular_test_losses, label='Regular MLP', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Test accuracy comparison
    ax = axes[1, 0]
    causal_max_acc = [acc.max() for acc in causal_test_accuracies]
    causal_min_acc = [acc.min() for acc in causal_test_accuracies]
    ax.plot(causal_max_acc, label='Causal MLP (max hidden)', alpha=0.7)
    ax.plot(causal_min_acc, label='Causal MLP (min hidden)', alpha=0.7, linestyle='--')
    ax.plot(regular_test_accuracies, label='Regular MLP', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Per-hidden-size accuracy (final epoch)
    ax = axes[1, 1]
    final_acc = causal_test_accuracies[-1]
    hidden_indices = np.arange(1, hidden_size + 1)
    ax.plot(hidden_indices, final_acc, marker='o', markersize=3)
    ax.set_xlabel('Hidden Size (effective)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Hidden-Size Accuracy (Final Epoch)')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(hidden_indices, final_acc, 1)
    p = np.poly1d(z)
    ax.plot(hidden_indices, p(hidden_indices), "r--", alpha=0.5, label=f'Trend (slope={z[0]:.6f})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("\nResults saved to training_results.png")
    
    # Also save per-hidden-size accuracy plot separately
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.plot(hidden_indices, final_acc, 'b-o', markersize=4, linewidth=1.5)
    ax2.set_xlabel('Effective Hidden Size')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title(f'Causal Hidden MLP: Accuracy vs Effective Hidden Size\n(Hidden Size = {hidden_size})')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, hidden_size)
    ax2.set_ylim(max(0, final_acc.min() - 0.05), min(1, final_acc.max() + 0.05))
    
    # Add annotations
    ax2.annotate(f'Min: {final_acc.min():.4f}', xy=(1, final_acc[0]), 
                 xytext=(5, 10), textcoords='offset points', fontsize=9)
    ax2.annotate(f'Max: {final_acc.max():.4f}', xy=(hidden_size, final_acc[-1]), 
                 xytext=(-60, 10), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('per_hidden_accuracy.png', dpi=150)
    print("Per-hidden accuracy saved to per_hidden_accuracy.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Causal Hidden MLP')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--model', type=str, default='cumsum', choices=['cumsum', 'heads'],
                        help='Model variant: cumsum (shared weights) or heads (separate heads)')
    
    args = parser.parse_args()
    
    results = run_experiment(
        hidden_size=args.hidden_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        model_variant=args.model
    )
    
    # Save results
    import json
    results_serializable = {
        'causal_per_hidden_acc': results['causal_per_hidden_acc'].tolist(),
        'regular_acc': float(results['regular_acc']),
        'is_monotonic': bool(results['is_monotonic']),
        'performance_gap': float(results['performance_gap']),
        'final_causal_max_acc': float(results['causal_per_hidden_acc'].max()),
    }
    
    with open('results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print("\nResults saved to results.json")
