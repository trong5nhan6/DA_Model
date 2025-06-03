import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


WBC_LABELS = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
WBC_LABEL2ID = {label: idx + 1 for idx, label in enumerate(WBC_LABELS)}

folder_to_labelname = {
    "Basophil": "basophil",
    "Eosinophil": "eosinophil",
    "Lymphocyte": "lymphocyte",
    "Monocyte": "monocyte",
    "Neutrophil": "neutrophil"
}


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


class WBCFolderDataset(Dataset):
    def __init__(self, data_root, transform=None):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_wbc_from_folders(
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
    if augment:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    train_ds = WBCFolderDataset(train_root, transform=transform)
    test_ds = WBCFolderDataset(test_root, transform=transform)

    train_ds = get_subset(train_ds, train_ratio, seed)
    test_ds = get_subset(test_ds, test_ratio, seed)

    train_loader = make_loader(train_ds, batch_size, seed, shuffle=True,
                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader = make_loader(test_ds, batch_size, seed, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader
