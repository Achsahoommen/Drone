import os
import sys
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = ["healthy", "mild", "moderate", "severe"]
NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
UNFREEZE_EPOCH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Data:   {DATA_DIR}")


class LeafDataset(torch.utils.data.Dataset):
    EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root_dir, split="train"):
        from sklearn.model_selection import train_test_split

        self.samples = []
        self.labels = []
        self.transform = get_transforms(split)

        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = Path(root_dir) / cls_name
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in self.EXT:
                    self.samples.append(str(f))
                    self.labels.append(cls_idx)

        all_paths = np.array(self.samples)
        all_labels = np.array(self.labels)

        train_p, test_p, train_l, test_l = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        val_p, test_p, val_l, test_l = train_test_split(
            test_p, test_l, test_size=0.5, random_state=42, stratify=test_l
        )

        if split == "train":
            self.samples, self.labels = list(train_p), list(train_l)
        elif split == "val":
            self.samples, self.labels = list(val_p), list(val_l)
        else:
            self.samples, self.labels = list(test_p), list(test_l)

        self.class_counts = np.bincount(self.labels, minlength=NUM_CLASSES)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.samples[idx]).convert("RGB")
        label = self.labels[idx]
        return self.transform(img), label


def get_transforms(split):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def build_model(frozen=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = not frozen
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, NUM_CLASSES),
    )
    return model.to(DEVICE)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    print("\nLoading dataset...")
    train_ds = LeafDataset(DATA_DIR, "train")
    val_ds = LeafDataset(DATA_DIR, "val")
    test_ds = LeafDataset(DATA_DIR, "test")

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Classes: {dict(zip(CLASSES, train_ds.class_counts))}")

    weights = 1.0 / (train_ds.class_counts + 1e-6)
    sample_weights = [float(w) for w in weights[train_ds.labels]]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(frozen=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0.0
    best_model: dict[str, torch.Tensor] = {}
    no_improve = 0

    print(f"\nPhase 1: Train classifier only ({UNFREEZE_EPOCH} epochs)")
    print(f"  Trainable params: {count_trainable(model):,}\n")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UNFREEZE_EPOCH, eta_min=1e-5)

    start = time.time()

    for epoch in range(1, UNFREEZE_EPOCH + 1):
        ep_start = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({
                "epoch": epoch, "model_state_dict": best_model,
                "val_acc": val_acc, "class_names": CLASSES,
            }, str(CHECKPOINT_DIR / "best_model.pth"))
        else:
            no_improve += 1

        tag = " *** BEST ***" if is_best else ""
        ep_time = time.time() - ep_start
        print(f"  Epoch {epoch:2d}/{UNFREEZE_EPOCH} | "
              f"Train: {train_loss:.4f} {train_acc:.1f}% | "
              f"Val: {val_loss:.4f} {val_acc:.1f}% | {ep_time:.0f}s{tag}")

    print(f"\n--- Phase 1 done. Best val: {best_acc:.1f}% ---\n")

    print(f"Phase 2: Fine-tune full model ({EPOCHS - UNFREEZE_EPOCH} epochs)")
    model = build_model(frozen=False)
    model.load_state_dict(best_model)
    print(f"  Trainable params: {count_trainable(model):,}\n")

    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    remaining = EPOCHS - UNFREEZE_EPOCH
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)
    no_improve = 0

    for epoch in range(1, remaining + 1):
        ep_start = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({
                "epoch": epoch + UNFREEZE_EPOCH, "model_state_dict": best_model,
                "val_acc": val_acc, "class_names": CLASSES,
            }, str(CHECKPOINT_DIR / "best_model.pth"))
        else:
            no_improve += 1

        tag = " *** BEST ***" if is_best else ""
        ep_time = time.time() - ep_start
        print(f"  Epoch {epoch:2d}/{remaining} | "
              f"Train: {train_loss:.4f} {train_acc:.1f}% | "
              f"Val: {val_loss:.4f} {val_acc:.1f}% | {ep_time:.0f}s{tag}")

        if no_improve >= 8:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.0f}s | Best val accuracy: {best_acc:.1f}%")

    print("\nEvaluating on test set...")
    model.load_state_dict(best_model)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
    print(f"\n{report}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return best_acc


if __name__ == "__main__":
    train()
