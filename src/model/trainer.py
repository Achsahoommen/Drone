import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional

from utils.config import (
    CONFIG, TRAIN_CFG, INFERENCE_CFG, NUM_CLASSES, CLASSES,
    get_project_root, get_device, setup_logging
)
from model.network import build_model, count_parameters
from preprocessing.dataset import create_dataloaders

logger = setup_logging("trainer")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


class Trainer:
    def __init__(self, model=None, device=None, data_dir=None):
        self.root = get_project_root()
        self.device = device or get_device()
        self.model = model or build_model().to(self.device)
        self.data_dir = data_dir or str(self.root / "data" / "processed")
        self.checkpoint_dir = self.root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAIN_CFG["learning_rate"],
            weight_decay=TRAIN_CFG["weight_decay"],
        )

        self.scaler = GradScaler(enabled=TRAIN_CFG.get("mixed_precision", True))
        self.use_amp = TRAIN_CFG.get("mixed_precision", True)
        self.epochs = TRAIN_CFG["epochs"]
        self.early_stopper = EarlyStopping(
            patience=TRAIN_CFG["early_stopping_patience"]
        )

        self.best_val_acc = 0.0
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

        self.writer = None
        if CONFIG["logging"].get("tensorboard", True):
            self.writer = SummaryWriter(str(self.root / "logs" / "tensorboard"))

        info = count_parameters(self.model)
        logger.info(
            "Trainer initialized | Device: {} | Params: {} | Model: {}".format(
                self.device, info["total_m"], CONFIG["model"]["architecture"]
            )
        )

    def _get_scheduler(self):
        sched_type = TRAIN_CFG.get("scheduler", "cosine")
        if sched_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-6
            )
        elif sched_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=15, gamma=0.1
            )
        elif sched_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=5
            )
        return None

    def train_one_epoch(self, dataloader, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    "  Epoch {} [{}/{}] Loss: {:.4f} | Acc: {:.1f}%".format(
                        epoch, batch_idx + 1, len(dataloader),
                        loss.item(), 100.0 * correct / total
                    )
                )

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = 100.0 * correct / max(total, 1)
        return {"loss": epoch_loss, "accuracy": epoch_acc}

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = 100.0 * correct / max(total, 1)
        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "predictions": all_preds,
            "labels": all_labels,
        }

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "config": CONFIG,
        }
        path = self.checkpoint_dir / "checkpoint_epoch{}_acc{:.1f}.pth".format(
            epoch, val_acc
        )
        torch.save(state, str(path))

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, str(best_path))
            logger.info("New best model saved: {} (acc={:.2f}%)".format(
                best_path, val_acc
            ))

    def train(self):
        logger.info("=" * 60)
        logger.info("Starting training pipeline")
        logger.info("=" * 60)

        train_loader, val_loader, test_loader = create_dataloaders(self.data_dir)
        scheduler = self._get_scheduler()
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - epoch_start

            logger.info(
                "Epoch {}/{} ({:.1f}s) | LR: {:.2e} | "
                "Train Loss: {:.4f} Acc: {:.1f}% | "
                "Val Loss: {:.4f} Acc: {:.1f}%".format(
                    epoch, self.epochs, elapsed, current_lr,
                    train_metrics["loss"], train_metrics["accuracy"],
                    val_metrics["loss"], val_metrics["accuracy"],
                )
            )

            is_best = val_metrics["accuracy"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["accuracy"]
                self.save_checkpoint(epoch, val_metrics["accuracy"], is_best=True)

            if self.writer:
                self.writer.add_scalars("loss", {
                    "train": train_metrics["loss"],
                    "val": val_metrics["loss"],
                }, epoch)
                self.writer.add_scalars("accuracy", {
                    "train": train_metrics["accuracy"],
                    "val": val_metrics["accuracy"],
                }, epoch)
                self.writer.add_scalar("lr", current_lr, epoch)

            if self.early_stopper(val_metrics["accuracy"]):
                logger.info("Early stopping at epoch {}".format(epoch))
                break

        total_time = time.time() - start_time
        logger.info("Training complete in {:.1f}s".format(total_time))
        logger.info("Best validation accuracy: {:.2f}%".format(self.best_val_acc))

        logger.info("Evaluating on test set...")
        test_metrics = self.validate(test_loader)
        logger.info("Test Loss: {:.4f} | Test Accuracy: {:.1f}%".format(
            test_metrics["loss"], test_metrics["accuracy"]
        ))

        if self.writer:
            self.writer.close()

        return {
            "history": self.history,
            "best_val_acc": self.best_val_acc,
            "test_metrics": test_metrics,
        }


if __name__ == "__main__":
    trainer = Trainer()
    results = trainer.train()
