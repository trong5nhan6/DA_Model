import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from medmnist import BloodMNIST

# ======== label mapping ========
ORIGINAL_TO_NEW = {
    0: 1,  # basophil
    1: 2,  # eosinophil
    4: 3,  # lymphocyte
    5: 4,  # monocyte
    6: 5   # neutrophil
}

WBC_LABELS = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
WBC_LABEL2ID = {label: idx + 1 for idx, label in enumerate(WBC_LABELS)}
folder_to_labelname = {
    "Basophil": "basophil",
    "Eosinophil": "eosinophil",
    "Lymphocyte": "lymphocyte",
    "Monocyte": "monocyte",
    "Neutrophil": "neutrophil"
}


# ======== Utils ========
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


# ======== Dataset BloodMNIST đã lọc ========
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


# ======== Dataset WBC Folder ========
class WBCFolderDataset(Dataset):
    def __init__(self, data_root, transform=None, seed=42):
        self.data = []
        self.transform = transform
        for folder, label_name in folder_to_labelname.items():
            label_id = WBC_LABEL2ID[label_name]
            folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fpath = os.path.join(folder_path, fname)
                    self.data.append((fpath, label_id))

        random.seed(seed)
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ======== Loader cho BloodMNIST ========
def load_bloodmnist(
    img_size=(28, 28),
    batch_size=32,
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
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(
            15) if augment else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_ds = FilteredBloodMNIST(
        split="train", transform=transform, download=download)
    test_ds = FilteredBloodMNIST(
        split="test", transform=transform, download=download)

    train_ds = get_subset(train_ds, train_ratio, seed)
    test_ds = get_subset(test_ds, test_ratio, seed)

    return (
        make_loader(train_ds, batch_size, seed, True, num_workers, pin_memory),
        make_loader(test_ds, batch_size, seed, False, num_workers, pin_memory)
    )


# ======== Loader cho WBC Folder ========
def load_wbc(
    train_root,
    test_root,
    img_size=(64, 64),
    batch_size=64,
    seed=42,
    num_workers=4,
    train_ratio=1.0,
    test_ratio=1.0,
    pin_memory=True,
    augment=False
):
    mean = [0.5] * 3
    std = [0.5] * 3
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(
            15) if augment else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_ds = WBCFolderDataset(train_root, transform=transform, seed=seed)
    test_ds = WBCFolderDataset(test_root, transform=transform, seed=seed)

    train_ds = get_subset(train_ds, train_ratio, seed)  
    test_ds = get_subset(test_ds, test_ratio, seed)

    return (
        make_loader(train_ds, batch_size, seed, True, num_workers, pin_memory),
        make_loader(test_ds, batch_size, seed, False, num_workers, pin_memory)
    )
