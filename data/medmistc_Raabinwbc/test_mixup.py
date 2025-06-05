import torch
import matplotlib.pyplot as plt
import numpy as np
from load_blood_data import load_bloodmnist
from torchvision.utils import make_grid
import os


def denormalize(tensor):
    """Convert normalized tensor back to image format"""
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return tensor * std + mean


def plot_images(images, labels, title, nrow=4):
    """Plot a grid of images with their labels"""
    plt.figure(figsize=(15, 15))
    grid = make_grid(images, nrow=nrow, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title, fontsize=15)
    plt.axis('off')
    plt.show()


def test_majority_minority_mixup():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data with mixup
    print("\nLoading data with mixup...")
    train_loader, test_loader, mixup_fn, mixup_criterion = load_bloodmnist(
        batch_size=16,
        use_mixup=True,
        augment=True,
        mixup_alpha=0.2,
        minority_threshold=0.2  # Classes with count < 0.5 * mean_count are minority
    )

    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nOriginal batch shape: {images.shape}")
    print(f"Original labels: {labels}")

    # Display original images
    plot_images(denormalize(images), labels,
                "Original Images (Before Mixup)")

    # Apply mixup
    mixed_images, labels_a, labels_b, lam = mixup_fn(images, labels)
    print(f"\nMixup lambda: {lam}")
    print(f"Labels A (minority): {labels_a}")
    print(f"Labels B (majority): {labels_b}")

    # Display mixed images
    plot_images(denormalize(mixed_images), labels_a,
                f"Mixed Images (lambda={lam:.2f})")

    # Test different alpha values
    print("\nTesting different alpha values...")
    alphas = [0.1, 0.5, 1.0]
    plt.figure(figsize=(15, 10))

    # Plot original images
    plt.subplot(2, 2, 1)
    grid_orig = make_grid(denormalize(images), nrow=4, normalize=True)
    plt.imshow(grid_orig.permute(1, 2, 0).cpu().numpy())
    plt.title("Original Images", fontsize=12)
    plt.axis('off')

    # Plot mixed images for different alphas
    for i, alpha in enumerate(alphas, 2):
        plt.subplot(2, 2, i)
        mixed_images, labels_a, labels_b, lam = mixup_fn(
            images, labels, alpha=alpha)
        grid_mixed = make_grid(denormalize(
            mixed_images), nrow=4, normalize=True)
        plt.imshow(grid_mixed.permute(1, 2, 0).cpu().numpy())
        plt.title(
            f"Mixed Images (alpha={alpha}, lambda={lam:.2f})", fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Test class distribution
    print("\nTesting class distribution...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())

    class_counts = np.bincount(all_labels)
    print("Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count} samples")


if __name__ == "__main__":
    test_majority_minority_mixup()
