import torch
from load_blood_data import load_bloodmnist
import matplotlib.pyplot as plt
import numpy as np


def visualize_batch(images, labels, title, n_cols=4):
    """Visualize a batch of images with their labels"""
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten()

    for idx, (img, label) in enumerate(zip(images, labels)):
        # Convert tensor to numpy array and transpose to (H, W, C)
        img = img.numpy().transpose(1, 2, 0)
        # Denormalize
        img = (img * 0.5 + 0.5).clip(0, 1)

        axes[idx].imshow(img)
        axes[idx].set_title(f'Label: {label}')
        axes[idx].axis('off')

    # Hide empty subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def test_bloodmnist():
    """Test different augmentation methods on the same batch"""
    print("\n=== Testing BloodMNIST Dataset ===")

    # Test parameters
    img_size = (64, 64)
    batch_size = 16
    seed = 42

    # Load dataset without any augmentation
    print("\n1. Testing without augmentation:")
    train_loader, _ = load_bloodmnist(
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        augment=False,
        use_mixup=False
    )

    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    visualize_batch(images, labels, "Without Augmentation")

    # Load dataset with augmentation
    print("\n2. Testing with augmentation:")
    train_loader, _ = load_bloodmnist(
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        augment=True,
        use_mixup=False
    )

    # Get a batch
    aug_images, aug_labels = next(iter(train_loader))
    print(f"Augmented batch shape: {aug_images.shape}")
    print(f"Augmented labels: {aug_labels}")
    visualize_batch(aug_images, aug_labels, "With Augmentation")

    # Load dataset with mixup
    print("\n3. Testing with mixup:")
    train_loader, _, mixup_fn = load_bloodmnist(
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        augment=False,
        use_mixup=True
    )

    # Get a batch and apply mixup
    mixup_images, mixup_labels = next(iter(train_loader))
    mixed_images, y_a, y_b, lam = mixup_fn(mixup_images, mixup_labels)
    print(f"Mixed batch shape: {mixed_images.shape}")
    print(f"Mixup lambda: {lam}")
    print(f"Original labels: {mixup_labels}")
    print(f"Mixed labels (y_a): {y_a}")
    print(f"Mixed labels (y_b): {y_b}")
    visualize_batch(mixed_images, y_a, f"With Mixup (λ={lam:.2f})")


if __name__ == "__main__":
    test_bloodmnist()
