from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_model(y_true, y_pred, class_names: list[str]):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, digits=4
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_metrics(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in metrics.items() if k != "classification_report"}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(
        metrics["classification_report"], encoding="utf-8"
    )


def plot_confusion(cm: list[list[int]], class_names: list[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_epoch_curve(history: list[dict], out_path: Path, title: str) -> None:
    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.plot(epochs, val_f1, label="Val Macro F1")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def to_class_names(label_encoder) -> list[str]:
    return [str(c) for c in label_encoder.classes_]
