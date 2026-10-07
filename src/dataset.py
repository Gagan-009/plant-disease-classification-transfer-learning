import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(is_train: bool = True, image_size: int = 224):
    """
    Returns data transforms for train and validation/test splits.
    Uses official ImageNet normalization as expected by pretrained models.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class PlantDiseaseDataset(Dataset):
    """
    PyTorch Dataset backed by split CSV files (train.csv, val.csv, test.csv).
    Handles relative path normalization across working directories.
    """
    def __init__(self, csv_file: str, base_dir: str = ".", transform=None, class_to_idx: dict = None):
        self.df = pd.read_csv(csv_file)
        self.base_dir = os.path.abspath(base_dir)
        self.transform = transform

        # Build or reuse class_to_idx mapping
        all_classes = sorted(self.df["class"].unique())
        if class_to_idx is None:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_classes)}
        else:
            self.class_to_idx = class_to_idx

        self.classes = [k for k, v in sorted(self.class_to_idx.items(), key=lambda item: item[1])]

        # Resolve filepaths robustly
        self.filepaths = []
        for raw_p in self.df["filepath"]:
            norm_p = raw_p.replace("\\", "/").lstrip("./").lstrip("../")
            candidates = [
                os.path.join(self.base_dir, norm_p),
                os.path.join(self.base_dir, "data_temp_pv", "raw", "color", os.path.basename(os.path.dirname(raw_p)), os.path.basename(raw_p)),
                raw_p
            ]
            resolved = None
            for cand in candidates:
                if os.path.exists(cand):
                    resolved = cand
                    break
            if resolved is None:
                # Default to base_dir + normalized path
                resolved = os.path.join(self.base_dir, norm_p)
            self.filepaths.append(resolved)

        self.labels = [self.class_to_idx[cls_name] for cls_name in self.df["class"]]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_stratified_subset(csv_file: str, samples_per_class: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Creates a manageable stratified subset dataframe with equal samples per class
    for rapid smoke testing and demos without data leakage.
    """
    df = pd.read_csv(csv_file)
    records = []
    for cls_name, group in df.groupby("class"):
        records.append(group.sample(n=min(len(group), samples_per_class), random_state=seed))
    subset = pd.concat(records, ignore_index=True)
    return subset


def get_dataloader(
    csv_file: str,
    base_dir: str = ".",
    batch_size: int = 32,
    is_train: bool = True,
    samples_per_class: int = None,
    class_to_idx: dict = None,
    seed: int = 42,
    num_workers: int = 0
):
    """
    Factory function to build DataLoader from CSV split file.
    """
    transform = get_transforms(is_train=is_train)

    if samples_per_class is not None:
        subset_df = create_stratified_subset(csv_file, samples_per_class=samples_per_class, seed=seed)
        # Temporary in-memory CSV
        temp_csv_path = f"data/splits/_temp_subset_{'train' if is_train else 'eval'}.csv"
        os.makedirs(os.path.dirname(temp_csv_path), exist_ok=True)
        subset_df.to_csv(temp_csv_path, index=False)
        target_csv = temp_csv_path
    else:
        target_csv = csv_file

    dataset = PlantDiseaseDataset(
        csv_file=target_csv,
        base_dir=base_dir,
        transform=transform,
        class_to_idx=class_to_idx
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=False
    )
    return loader, dataset
