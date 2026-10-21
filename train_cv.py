"""5-fold stratified cross-validation for the spinach severity classifier.

- Holds out a stratified 10% test set (never trained on).
- Runs StratifiedKFold (5 folds) on the remaining 90% with 2-phase
  MobileNetV2 transfer learning (same recipe as train.py).
- Saves one model per fold: checkpoints/fold_{i}.pth
- Evaluates the SOFTMAX-AVERAGED ENSEMBLE on the held-out test set.
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms

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
FOLDS = 5
TEST_SIZE = 0.1
EPOCHS = 12
UNFREEZE_EPOCH = 5
PATIENCE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

print(f"Device: {DEVICE}")
print(f"Data:   {DATA_DIR}")


class LeafDataset(Dataset):
    def __init__(self, samples, labels, split="train"):
        self.samples = list(samples)
        self.labels = [int(x) for x in labels]
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


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


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def train_one_fold(fold, train_idx, val_idx, samples, labels, args):
    print(f"\n{'=' * 60}")
    print(f"  FOLD {fold + 1}/{FOLDS}  |  train {len(train_idx)} | val {len(val_idx)}")
    print(f"{'=' * 60}")

    tr_s, tr_l = samples[train_idx], labels[train_idx]
    va_s, va_l = samples[val_idx], labels[val_idx]

    train_ds = LeafDataset(tr_s, tr_l, "train")
    val_ds = LeafDataset(va_s, va_l, "val")

    counts = np.bincount(tr_l, minlength=NUM_CLASSES)
    weights = 1.0 / (counts + 1e-6)
    sampler = WeightedRandomSampler(list(weights[tr_l]), len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = build_model(frozen=True)
    best_acc, best_state, no_improve = 0.0, None, 0
    fold_path = CHECKPOINT_DIR / f"fold_{fold}.pth"

    unfreeze = args.unfreeze_epoch
    remaining = args.epochs - unfreeze

    print(f"\nPhase 1: classifier only ({unfreeze} epochs)")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(unfreeze, 1), eta_min=1e-5)

    for epoch in range(1, unfreeze + 1):
        ep_start = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step()
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({"fold": fold, "model_state_dict": best_state, "phase": 1,
                        "val_acc": val_acc, "class_names": CLASSES}, str(fold_path))
        else:
            no_improve += 1
        tag = " *** BEST ***" if is_best else ""
        print(f"  [F{fold + 1}] Epoch {epoch}/{unfreeze} | Train {tr_loss:.4f} {tr_acc:.1f}% | "
              f"Val {val_loss:.4f} {val_acc:.1f}% | {time.time() - ep_start:.0f}s{tag}")

    print(f"Phase 1 done. Best val: {best_acc:.1f}%")

    print(f"\nPhase 2: fine-tune full model ({remaining} epochs)")
    model = build_model(frozen=False)
    model.load_state_dict(best_state)
    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(remaining, 1), eta_min=1e-6)
    no_improve = 0

    for epoch in range(1, remaining + 1):
        ep_start = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step()
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({"fold": fold, "model_state_dict": best_state, "phase": 2,
                        "val_acc": val_acc, "class_names": CLASSES}, str(fold_path))
        else:
            no_improve += 1
        tag = " *** BEST ***" if is_best else ""
        print(f"  [F{fold + 1}] Epoch {epoch}/{remaining} | Train {tr_loss:.4f} {tr_acc:.1f}% | "
              f"Val {val_loss:.4f} {val_acc:.1f}% | {time.time() - ep_start:.0f}s{tag}")
        if no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(DEVICE))
            preds.extend(outputs.argmax(1).cpu().numpy())
            truths.extend(labels.numpy())

    print(f"Fold {fold + 1} final val accuracy: {best_acc:.2f}%")
    return best_state, best_acc, np.array(truths), np.array(preds)


def evaluate_ensemble(states, samples, labels, args):
    print(f"\n{'=' * 60}")
    print("  ENSEMBLE EVALUATION ON HELD-OUT TEST SET")
    print(f"{'=' * 60}")

    nets = []
    for state in states:
        net = build_model(frozen=False)
        net.load_state_dict(state)
        net.eval()
        nets.append(net)

    test_ds = LeafDataset(samples, labels, "val")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(DEVICE)
            probs = torch.stack([F_softmax(net, images) for net in nets]).mean(0)
            all_probs.append(probs.cpu())
            all_labels.extend(batch_labels.numpy())

    probs = torch.cat(all_probs).numpy()
    preds = probs.argmax(1)
    acc = 100.0 * (preds == np.array(all_labels)).mean()

    report = classification_report(all_labels, preds, target_names=CLASSES, digits=4)
    cm = confusion_matrix(all_labels, preds)
    print(f"\nEnsemble test accuracy: {acc:.2f}%\n")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    out = OUTPUT_DIR / "cv_report.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Ensemble test accuracy: {acc:.2f}%\n\n{report}\nConfusion Matrix:\n{np.array2string(cm)}\n")
    print(f"Report saved to {out}")
    return acc


def F_softmax(net, images):
    return torch.softmax(net(images), dim=1)


def main():
    parser = argparse.ArgumentParser(description="5-fold CV + ensemble training")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--unfreeze-epoch", type=int, default=UNFREEZE_EPOCH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--force", action="store_true",
                        help="retrain folds even if fold_*.pth already exists")
    args = parser.parse_args()

    samples, labels = [], []
    for ci, cls in enumerate(CLASSES):
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            raise SystemExit(f"Missing class folder: {cls_dir}")
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in EXT:
                samples.append(str(f))
                labels.append(ci)
    samples = np.array(samples)
    labels = np.array(labels)
    print(f"Total images: {len(samples)} | counts: {dict(zip(CLASSES, np.bincount(labels)))}")

    rest_s, test_s, rest_l, test_l = train_test_split(
        samples, labels, test_size=TEST_SIZE, random_state=42, stratify=labels)
    print(f"Held-out test set: {len(test_s)} images ({TEST_SIZE:.0%})")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    states, fold_accs = [], []

    start = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(rest_s, rest_l)):
        fold_path = CHECKPOINT_DIR / f"fold_{fold}.pth"
        if fold_path.exists() and not args.force:
            ckpt = torch.load(str(fold_path), map_location="cpu", weights_only=False)
            # Only reuse a fold if it finished BOTH phases (phase marker == 2).
            if "model_state_dict" in ckpt and ckpt.get("phase") == 2:
                states.append(ckpt["model_state_dict"])
                fold_accs.append(float(ckpt.get("val_acc", 0.0)))
                print(f"\nFold {fold + 1}: reusing completed {fold_path.name} "
                      f"(val {ckpt.get('val_acc', 0):.1f}%) — pass --force to retrain")
                continue
            if ckpt.get("phase") != 2:
                print(f"\nFold {fold + 1}: {fold_path.name} is a partial Phase-1 "
                      f"checkpoint — retraining from scratch")
        state, val_acc, _, _ = train_one_fold(fold, train_idx, val_idx, rest_s, rest_l, args)
        states.append(state)
        fold_accs.append(val_acc)

    fold_accs = np.array(fold_accs)
    print(f"\n{'=' * 60}")
    print(f"CV val accuracy per fold: {[f'{a:.2f}%' for a in fold_accs]}")
    print(f"Mean: {fold_accs.mean():.2f}%  |  Std: {fold_accs.std():.2f}%")
    print(f"Total training time: {(time.time() - start) / 60:.1f} min")

    evaluate_ensemble(states, test_s, test_l, args)
    print("\nSaved fold models:")
    for p in sorted(CHECKPOINT_DIR.glob("fold_*.pth")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
