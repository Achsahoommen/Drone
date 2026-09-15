import os
import cv2
import json
import random
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from utils.config import (
    CONFIG, CLASSES, CLASS_MAP, AUG_CFG, TRAIN_CFG,
    get_project_root, setup_logging
)

logger = setup_logging("preprocessing")


# ── Augmentation Pipelines ─────────────────────────────────────

def get_train_augmentations() -> A.Compose:
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=AUG_CFG["horizontal_flip_prob"]),
        A.VerticalFlip(p=AUG_CFG["vertical_flip_prob"]),
        A.RandomRotate90(p=0.3),
        A.Affine(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=AUG_CFG["rotation_limit"],
            border_mode=cv2.BORDER_REFLECT,
            p=0.5,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=AUG_CFG["brightness_limit"],
                contrast_limit=AUG_CFG["contrast_limit"],
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0,
            ),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=AUG_CFG["blur_limit"], p=1.0),
        ], p=0.3),
        A.CoarseDropout(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_augmentations() -> A.Compose:
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ── Dataset ─────────────────────────────────────────────────────

class SpinachDataset(Dataset):
    """
    Expects folder structure:
        data/raw/healthy/*.jpg
        data/raw/mild/*.jpg
        data/raw/moderate/*.jpg
        data/raw/severe/*.jpg
    """

    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[A.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    @classmethod
    def from_directory(cls, root_dir: str, transform=None) -> "SpinachDataset":
        root = Path(root_dir)
        all_paths, all_labels = [], []

        for cls_name in CLASSES:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                logger.warning(f"Directory not found: {cls_dir}")
                continue
            for img_file in sorted(cls_dir.iterdir()):
                if img_file.suffix.lower() in cls.VALID_EXT:
                    all_paths.append(str(img_file))
                    all_labels.append(CLASS_MAP[cls_name])

        logger.info(
            f"Loaded {len(all_paths)} images: "
            + ", ".join(
                f"{CLASSES[i]}={all_labels.count(i)}" for i in range(len(CLASSES))
            )
        )
        return cls(all_paths, all_labels, transform)


# ── Dataset Splitting ───────────────────────────────────────────

def split_dataset(
    root_dir: str,
    output_dir: str,
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    if train_ratio is None:
        train_ratio = TRAIN_CFG["train_split"]
    if val_ratio is None:
        val_ratio = TRAIN_CFG["val_split"]
    if test_ratio is None:
        test_ratio = TRAIN_CFG["test_split"]

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    random.seed(seed)
    output = Path(output_dir)

    for split in ["train", "val", "test"]:
        for cls_name in CLASSES:
            (output / split / cls_name).mkdir(parents=True, exist_ok=True)

    stats = {"train": {}, "val": {}, "test": {}}

    for cls_name in CLASSES:
        src = Path(root_dir) / cls_name
        if not src.exists():
            continue

        images = sorted([
            f for f in src.iterdir()
            if f.suffix.lower() in SpinachDataset.VALID_EXT
        ])
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            for f in files:
                shutil.copy2(str(f), str(output / split_name / cls_name / f.name))
            stats[split_name][cls_name] = len(files)

    logger.info("Dataset split complete -> {}".format(output))
    for split, counts in stats.items():
        total = sum(counts.values())
        logger.info(f"  {split}: {total} images")
        for cls, cnt in counts.items():
            logger.info(f"    {cls}: {cnt}")

    return stats


# ── DataLoader Factory ──────────────────────────────────────────

def create_dataloaders(
    data_dir: str,
    batch_size: int = None,
    num_workers: int = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if batch_size is None:
        batch_size = TRAIN_CFG["batch_size"]
    if num_workers is None:
        num_workers = TRAIN_CFG.get("num_workers", 4)

    data_path = Path(data_dir)
    datasets = {}
    loaders = {}

    for split, aug_fn in [
        ("train", get_train_augmentations),
        ("val", get_val_augmentations),
        ("test", get_val_augmentations),
    ]:
        ds = SpinachDataset.from_directory(str(data_path / split), transform=aug_fn())
        datasets[split] = ds

        if split == "train" and len(ds) > 0:
            labels = np.array(ds.labels)
            class_counts = np.bincount(labels, minlength=len(CLASSES))
            class_weights = 1.0 / (class_counts + 1e-6)
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(ds),
                replacement=True,
            )
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split != "test"),
                num_workers=num_workers,
                pin_memory=True,
            )

    return loaders["train"], loaders["val"], loaders["test"]


# ── Data Augmentation Demo ──────────────────────────────────────

def visualize_augmentations(
    image_path: str,
    output_path: str,
    num_samples: int = 8,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_train_augmentations()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Data Augmentation Examples", fontsize=16)

    for idx, ax in enumerate(axes.flat):
        if idx == 0:
            ax.imshow(image)
            ax.set_title("Original")
        else:
            augmented = transform(image=image)
            aug_image = augmented["image"].permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            aug_image = aug_image * std + mean
            aug_image = np.clip(aug_image, 0, 1)
            ax.imshow(aug_image)
            ax.set_title(f"Augmented {idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Augmentation visualization saved to {output_path}")


if __name__ == "__main__":
    root = get_project_root()
    raw_dir = str(root / "data" / "raw")
    processed_dir = str(root / "data" / "processed")

    split_dataset(raw_dir, processed_dir)
