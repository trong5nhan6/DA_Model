import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from medmnist import BloodMNIST
from collections import Counter

# ======== label mapping ========
ORIGINAL_TO_NEW = {
    0: 0,  # basophil (original 0 -> new 0)
    1: 1,  # eosinophil
    4: 2,  # lymphocyte
    5: 3,  # monocyte
    6: 4   # neutrophil
}

WBC_LABELS = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
WBC_LABEL2ID = {label: idx for idx, label in enumerate(WBC_LABELS)}
folder_to_labelname = {
    "Basophil": "basophil",
    "Eosinophil": "eosinophil",
    "Lymphocyte": "lymphocyte",
    "Monocyte": "monocyte",
    "Neutrophil": "neutrophil"
}


# ======== Mixup Data Augmentation ========
def get_strong_augmentation(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])


def get_weak_augmentation(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])


def get_standard_transform(img_size, augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Simple mixup loss function
    Args:
        criterion: Base loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixup lambda
    Returns:
        loss: Combined loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ======== Class-wise Augmentation ========
class ClassWiseAugDataset(Dataset):
    def __init__(self, dataset, minority_classes, strong_aug, weak_aug):
        self.dataset = dataset
        self.minority_classes = minority_classes
        self.strong_aug = strong_aug
        self.weak_aug = weak_aug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Convert tensor to PIL Image if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # If it's a single image
                image = transforms.ToPILImage()(image)
            else:  # If it's a batch
                image = transforms.ToPILImage()(image.squeeze(0))

        # Apply augmentation based on class
        if label in self.minority_classes:
            aug = self.strong_aug
        else:
            aug = self.weak_aug

        return aug(image), label


def minority_majority_mixup(images, labels, minority_classes, majority_classes, alpha=0.4):
    """
    Mixup between minority and majority classes
    Args:
        images: Batch of images
        labels: Batch of labels
        minority_classes: List of minority class indices
        majority_classes: List of majority class indices
        alpha: Mixup alpha parameter
    Returns:
        mixed_images: Mixed images
        y_a: Labels of minority samples
        y_b: Labels of majority samples
        lam: Mixup lambda
    """
    # Convert to sets for faster lookup
    minority_classes = set(minority_classes)
    majority_classes = set(majority_classes)

    # Get indices for minority and majority samples
    minor_idx = [i for i, y in enumerate(labels) if y in minority_classes]
    major_idx = [i for i, y in enumerate(labels) if y in majority_classes]

    n = min(len(minor_idx), len(major_idx))
    if n == 0:
        return images, labels, labels, 1.0

    # Randomly select n pairs
    selected_minor = random.sample(minor_idx, n)
    selected_major = random.sample(major_idx, n)

    # Generate mixup lambda
    lam = np.random.beta(alpha, alpha)

    # Mixup images
    mixed = lam * images[selected_minor] + (1 - lam) * images[selected_major]
    y_a = labels[selected_minor]
    y_b = labels[selected_major]

    return mixed, y_a, y_b, lam


def get_class_distribution(dataset):
    """Get class distribution from dataset"""
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    return class_counts


def identify_minority_classes(dataset, threshold=0.1):
    """
    Identify minority classes based on class distribution
    Args:
        dataset: Dataset to analyze
        threshold: Classes with count less than threshold * total_samples are considered minority
                  (threshold is a percentage, e.g., 0.1 means 10% of total data)
    Returns:
        minority_classes: List of minority class indices
        majority_classes: List of majority class indices
    """
    class_counts = get_class_distribution(dataset)
    total_samples = sum(class_counts.values())
    threshold_count = total_samples * threshold

    minority_classes = [
        cls for cls, count in class_counts.items() if count < threshold_count]
    majority_classes = [
        cls for cls, count in class_counts.items() if count >= threshold_count]

    return minority_classes, majority_classes


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
    download=True,
    use_mixup=False,
    mixup_alpha=0.2,
    device=None,
    minority_threshold=0.2  # New parameter for minority class identification
):
    if use_mixup:
        # Define strong and weak augmentations for mixup
        strong_transform = get_standard_transform(img_size, False)
        weak_transform = get_standard_transform(img_size, False)

    # Use standard transforms for both train and test
    train_transform = get_standard_transform(img_size, augment)
    test_transform = get_standard_transform(img_size, False)

    # Load datasets
    train_ds = FilteredBloodMNIST(
        split="train", transform=train_transform, download=download)
    test_ds = FilteredBloodMNIST(
        split="test", transform=test_transform, download=download)

    # Apply subset if needed
    train_ds = get_subset(train_ds, train_ratio, seed)
    test_ds = get_subset(test_ds, test_ratio, seed)

    # Create test loader (same for both cases)
    test_loader = make_loader(
        dataset=test_ds,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )

    if use_mixup:
        # Identify minority and majority classes
        minority_classes, majority_classes = identify_minority_classes(
            train_ds, minority_threshold)
        print(f"Minority classes: {minority_classes}")
        print(f"Majority classes: {majority_classes}")

        # Create class-wise augmented dataset
        train_ds = ClassWiseAugDataset(
            train_ds,
            minority_classes=minority_classes,
            strong_aug=strong_transform,
            weak_aug=weak_transform
        )

        # Create data loader
        train_loader = make_loader(
            dataset=train_ds,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
        )

        # Create mixup function with minority/majority classes
        def mixup_fn(x, y, device=None):
            return minority_majority_mixup(
                x, y,
                minority_classes=minority_classes,
                majority_classes=majority_classes,
                alpha=mixup_alpha
            )

        return train_loader, test_loader, mixup_fn
    else:
        train_loader = make_loader(
            dataset=train_ds,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
        )

        return train_loader, test_loader


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
    augment=False,
    use_mixup=False,
    mixup_alpha=0.2,
    device=None,
    minority_threshold=0.2  # New parameter for minority class identification
):
    if use_mixup:
        # Define strong and weak augmentations for mixup
        strong_transform = get_standard_transform(img_size, False)
        weak_transform = get_standard_transform(img_size, False)

    # Use standard transforms for both train and test
    train_transform = get_standard_transform(img_size, augment)
    test_transform = get_standard_transform(img_size, False)

    # Load datasets
    train_ds = WBCFolderDataset(
        train_root, transform=train_transform, seed=seed)
    test_ds = WBCFolderDataset(test_root, transform=test_transform, seed=seed)

    # Apply subset if needed
    train_ds = get_subset(train_ds, train_ratio, seed)
    test_ds = get_subset(test_ds, test_ratio, seed)

    # Create test loader (same for both cases)
    test_loader = make_loader(
        dataset=test_ds,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )

    if use_mixup:
        # Identify minority and majority classes
        minority_classes, majority_classes = identify_minority_classes(
            train_ds, minority_threshold)
        print(f"Minority classes: {minority_classes}")
        print(f"Majority classes: {majority_classes}")

        # Create class-wise augmented dataset
        train_ds = ClassWiseAugDataset(
            train_ds,
            minority_classes=minority_classes,
            strong_aug=strong_transform,
            weak_aug=weak_transform
        )

        # Create data loader
        train_loader = make_loader(
            dataset=train_ds,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
        )

        # Create mixup function with minority/majority classes
        def mixup_fn(x, y, device=None):
            return minority_majority_mixup(
                x, y,
                minority_classes=minority_classes,
                majority_classes=majority_classes,
                alpha=mixup_alpha
            )

        return train_loader, test_loader, mixup_fn
    else:
        train_loader = make_loader(
            dataset=train_ds,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
        )

        return train_loader, test_loader
