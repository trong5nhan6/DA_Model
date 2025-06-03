import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from medmnist import BloodMNIST

# ======== Nhãn được giữ lại ========
ORIGINAL_TO_NEW = {
    0: 1,  # basophil
    1: 2,  # eosinophil
    4: 3,  # lymphocyte
    5: 4,  # monocyte
    6: 5   # neutrophil
}

# ======== Tiện ích chung ========


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, batch_size=64, seed=42, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )


def get_subset(dataset, ratio, seed=42):
    if ratio <= 0:
        return None
    if ratio >= 1:
        return dataset
    random.seed(seed)
    indices = random.sample(range(len(dataset)), int(len(dataset) * ratio))
    return Subset(dataset, indices)

# ======== Bộ dataset đã lọc ========


class FilteredBloodMNIST(Dataset):
    def __init__(self, split="train", transform=None, download=True):
        self.dataset = BloodMNIST(
            split=split, transform=transform, download=download)
        self.transform = transform

        self.filtered_indices = [
            i for i, (_, label) in enumerate(self.dataset)
            if int(label) in ORIGINAL_TO_NEW
        ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        orig_idx = self.filtered_indices[idx]
        img, label = self.dataset[orig_idx]
        new_label = ORIGINAL_TO_NEW[int(label)]
        return img, new_label

# ======== Hàm load chính ========


def load_bloodmnist(
    img_size=(64, 64),
    batch_size=64,
    seed=42,
    num_workers=4,
    train_ratio=1.0,
    test_ratio=1.0,
    pin_memory=True,
    augment=False,
    download=True
):
    mean = [0.5] * 3
    std = [0.5] * 3

    if augment:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_ds = FilteredBloodMNIST(
        split="train", transform=transform, download=download)
    test_ds = FilteredBloodMNIST(
        split="test", transform=transform, download=download)

    train_ds = get_subset(train_ds, train_ratio, seed)
    test_ds = get_subset(test_ds, test_ratio, seed)

    train_loader = make_loader(train_ds, batch_size, seed, shuffle=True,
                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader = make_loader(test_ds, batch_size, seed, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader
