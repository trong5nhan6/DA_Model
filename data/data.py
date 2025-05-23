from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)                        # Python RNG
    np.random.seed(seed)                     # NumPy RNG
    torch.manual_seed(seed)                  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)         # PyTorch GPU

    # Đảm bảo tính tái lập với cuDNN (nên dùng cho debug hoặc demo)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, batch_size=64, seed=42, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


def make_loader(dataset, batch_size=64, seed=42, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


def load_mnist_and_mnistm_from_folder(mnistm_root, batch_size=64, seed=42):
    transform_mnist = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3 channels
    ])

    transform_mnistm = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB normalize
    ])

    # Source domain: MNIST
    mnist_train = MNIST(root='./data', train=True,
                        download=True, transform=transform_mnist)
    mnist_test = MNIST(root='./data', train=False,
                       download=True, transform=transform_mnist)

    # Target domain: MNIST-M organized as ImageFolder
    mnistm_train = ImageFolder(
        root=f"{mnistm_root}/training", transform=transform_mnistm)
    mnistm_test = ImageFolder(
        root=f"{mnistm_root}/testing", transform=transform_mnistm)

    # Dataloaders
    mnist_loader = make_loader(mnist_train, batch_size=batch_size, seed=seed)
    mnist_test_loader = make_loader(
        mnist_test, batch_size=batch_size, seed=seed, shuffle=False)
    mnistm_loader = make_loader(mnistm_train, batch_size=batch_size, seed=seed)
    mnistm_test_loader = make_loader(
        mnistm_test, batch_size=batch_size, seed=seed, shuffle=False)

    return mnist_loader, mnist_test_loader, mnistm_loader, mnistm_test_loader


if __name__ == "__main__":
    set_seed(123)  # Bước đầu tiên và quan trọng nhất!

    source_loader, source_test_loader, target_loader, target_test_loader = load_mnist_and_mnistm_from_folder(
        mnistm_root="./mnist_m", batch_size=64, seed=123
    )

    # Sau đó khởi tạo mô hình và train như bình thường
    print(source_loader.shape)
    print(source_test_loader.shape)
    print(target_loader.shape)
    print(target_test_loader.shape)
