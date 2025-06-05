import torch
import matplotlib.pyplot as plt
import numpy as np
from load_blood_data import load_bloodmnist, load_wbc
from torchvision import transforms


def denormalize(tensor):
    """Convert normalized tensor back to image format"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return tensor * std + mean


def plot_images(images, labels, title, num_images=4):
    """Plot a grid of images with their labels"""
    plt.figure(figsize=(15, 4))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i + 1)
        # Handle both single images and batches
        if images.dim() == 4:  # If it's a batch
            img = denormalize(images[i])
        else:  # If it's a single image
            img = denormalize(images)
        # Convert to numpy and adjust dimensions
        img = img.squeeze().permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(
            f'Label: {labels[i].item() if torch.is_tensor(labels) else labels[i]}')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def test_majority_minority_mixup():
    print("Loading data with mixup...")
    # Load data with mixup enabled
    train_loader, test_loader, mixup_fn = load_bloodmnist(
        batch_size=16,
        use_mixup=True,
        augment=True,
        mixup_alpha=0.2,
        minority_threshold=0.2
    )

    # Get a batch of data
    images, labels = next(iter(train_loader))
    print(f"\nOriginal batch shape: {images.shape}")
    print(f"Original labels: {labels}")

    # Display original images
    plot_images(images, labels, "Original Images")

    # Apply mixup
    mixed_images, mixed_labels = mixup_fn(images, labels)
    print(f"\nMixed batch shape: {mixed_images.shape}")
    print(
        f"Mixed labels shape: {mixed_labels[0].shape}, {mixed_labels[1].shape}")

    # Display mixed images
    plot_images(mixed_images, mixed_labels[0], "Mixed Images (Label 1)")
    plot_images(mixed_images, mixed_labels[1], "Mixed Images (Label 2)")

    # Test different alpha values
    print("\nTesting different alpha values...")
    alphas = [0.1, 0.5, 0.9]
    for alpha in alphas:
        print(f"\nAlpha = {alpha}")
        mixed_images, mixed_labels = mixup_fn(images, labels, alpha=alpha)
        plot_images(
            mixed_images, mixed_labels[0], f"Mixed Images (Alpha={alpha})")

    # Test class distribution
    print("\nTesting class distribution...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())

    unique, counts = np.unique(all_labels, return_counts=True)
    print("Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples")


def test_wbc_mixup():
    print("\nTesting WBC dataset mixup...")
    # Load WBC data with mixup enabled
    train_loader, test_loader, mixup_fn = load_wbc(
        train_root="path/to/wbc/train",
        test_root="path/to/wbc/test",
        batch_size=16,
        use_mixup=True,
        augment=True,
        mixup_alpha=0.2,
        minority_threshold=0.2
    )

    # Get a batch of data
    images, labels = next(iter(train_loader))
    print(f"\nOriginal WBC batch shape: {images.shape}")
    print(f"Original WBC labels: {labels}")

    # Display original images
    plot_images(images, labels, "Original WBC Images")

    # Apply mixup
    mixed_images, mixed_labels = mixup_fn(images, labels)
    print(f"\nMixed WBC batch shape: {mixed_images.shape}")
    print(
        f"Mixed WBC labels shape: {mixed_labels[0].shape}, {mixed_labels[1].shape}")

    # Display mixed images
    plot_images(mixed_images, mixed_labels[0], "Mixed WBC Images (Label 1)")
    plot_images(mixed_images, mixed_labels[1], "Mixed WBC Images (Label 2)")


if __name__ == "__main__":
    # Test BloodMNIST dataset
    test_majority_minority_mixup()

    # Uncomment to test WBC dataset
    # test_wbc_mixup()
