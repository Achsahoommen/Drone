import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_recall_fscore_support,
)

from utils.config import CLASSES, NUM_CLASSES, get_project_root, setup_logging

logger = setup_logging("evaluation")


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: str = None,
    title: str = "Confusion Matrix",
):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved: {}".format(save_path))
    plt.close()
    return cm


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training history saved: {}".format(save_path))
    plt.close()


def plot_per_class_metrics(
    y_true: List[int],
    y_pred: List[int],
    save_path: str = None,
):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES))
    )

    x = np.arange(NUM_CLASSES)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label="Precision", color="steelblue")
    ax.bar(x, recall, width, label="Recall", color="coral")
    ax.bar(x + width, f1, width, label="F1-Score", color="seagreen")

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Per-class metrics saved: {}".format(save_path))
    plt.close()


def generate_evaluation_report(
    y_true: List[int],
    y_pred: List[int],
    output_dir: str = None,
) -> Dict:
    if output_dir is None:
        output_dir = str(get_project_root() / "outputs")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    report_text = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        digits=4,
    )
    logger.info("\n" + report_text)

    report_path = Path(output_dir) / "classification_report.txt"
    with open(str(report_path), "w") as f:
        f.write("Spinach Disease Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("Overall Accuracy: {:.2f}%\n".format(acc * 100))
        f.write("F1 (Macro):       {:.4f}\n".format(f1_macro))
        f.write("F1 (Weighted):    {:.4f}\n\n".format(f1_weighted))
        f.write(report_text)

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=str(Path(output_dir) / "confusion_matrix.png"),
    )
    plot_per_class_metrics(
        y_true, y_pred,
        save_path=str(Path(output_dir) / "per_class_metrics.png"),
    )

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    y_true = np.random.randint(0, NUM_CLASSES, n).tolist()
    y_pred = y_true.copy()
    for i in range(n):
        if np.random.random() < 0.15:
            y_pred[i] = np.random.randint(0, NUM_CLASSES)

    result = generate_evaluation_report(y_true, y_pred)
    print("Accuracy: {:.2f}%".format(result["accuracy"] * 100))
