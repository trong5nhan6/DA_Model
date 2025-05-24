import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision import transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, batch_size=64, seed=42, shuffle=True, num_workers=4, pin_memory=True):
    """
    Create a DataLoader with specified parameters
    Args:
        dataset: PyTorch dataset
        batch_size: Number of samples per batch
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle the data
        num_workers: Number of subprocesses for data loading
        pin_memory: Whether to pin memory in CPU
    Returns:
        DataLoader object
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )


def get_subset(dataset, ratio, seed=42):
    """
    Get a random subset of the dataset
    Args:
        dataset: PyTorch dataset
        ratio: Ratio of samples to keep (0-1)
        seed: Random seed for reproducibility
    Returns:
        Subset of the dataset or None if ratio <= 0
    """
    if ratio <= 0:
        return None
    if ratio >= 1:
        return dataset
    random.seed(seed)
    indices = random.sample(range(len(dataset)), int(len(dataset) * ratio))
    return Subset(dataset, indices)


def load_mnist_and_mnistm_from_folder(
    mnistm_root,
    batch_size=64,
    seed=42,
    num_workers=4,
    mnist_ratio=1.0,
    mnistm_ratio=1.0,
    pin_memory=True
):
    """
    Load MNIST and MNIST-M datasets for domain adaptation
    Args:
        mnistm_root: Root directory containing MNIST-M dataset
        batch_size: Number of samples per batch
        seed: Random seed for reproducibility
        num_workers: Number of subprocesses for data loading
        mnist_ratio: Ratio of MNIST samples to use
        mnistm_ratio: Ratio of MNIST-M samples to use
        pin_memory: Whether to pin memory in CPU
    Returns:
        Tuple of (mnist_loader, mnist_test_loader, mnistm_loader, mnistm_test_loader)
    """
    # Define transformations for MNIST (source domain)
    transform_mnist = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(3),  # Convert to RGB
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define transformations for MNIST-M (target domain)
    transform_mnistm = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load source domain: MNIST dataset
    mnist_train = MNIST(root='./data', train=True,
                        download=True, transform=transform_mnist)
    mnist_test = MNIST(root='./data', train=False,
                       download=True, transform=transform_mnist)
    mnist_train = get_subset(mnist_train, mnist_ratio, seed)
    mnist_test = get_subset(mnist_test, mnist_ratio, seed)

    # Load target domain: MNIST-M dataset
    mnistm_train = ImageFolder(
        root=f"{mnistm_root}/training", transform=transform_mnistm)
    mnistm_test = ImageFolder(
        root=f"{mnistm_root}/testing", transform=transform_mnistm)
    mnistm_train = get_subset(mnistm_train, mnistm_ratio, seed)
    mnistm_test = get_subset(mnistm_test, mnistm_ratio, seed)

    # Create dataloaders for training and testing
    mnist_loader = make_loader(mnist_train, batch_size, seed, shuffle=True,
                               num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True) if mnist_train else None
    mnist_test_loader = make_loader(mnist_test, batch_size, seed, shuffle=False,
                                    num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True) if mnist_test else None
    mnistm_loader = make_loader(mnistm_train, batch_size, seed, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True) if mnistm_train else None
    mnistm_test_loader = make_loader(mnistm_test, batch_size, seed, shuffle=False,
                                     num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True) if mnistm_test else None

    return mnist_loader, mnist_test_loader, mnistm_loader, mnistm_test_loader


if __name__ == "__main__":
    set_seed(123)

    source_loader, source_test_loader, target_loader, target_test_loader = load_mnist_and_mnistm_from_folder(
        mnistm_root="./mnist_m",
        batch_size=64,
        seed=123,
        pin_memory=True
    )

    # In thử 1 batch để kiểm tra dữ liệu
    xs, ys = next(iter(source_loader))
    print("Source batch shape:", xs.shape)

    xt, yt = next(iter(target_loader))
    print("Target batch shape:", xt.shape)
