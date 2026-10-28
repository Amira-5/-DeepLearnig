import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

LOGS = ROOT / "outputs" / "logs"
FIGURES = ROOT / "outputs" / "figures"


def load_json(file_path: Path) -> dict:
    if not file_path.exists():
        print(f"Notice: Log file not found at {file_path}")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_model_comparison():
    FIGURES.mkdir(parents=True, exist_ok=True)

    scratch_hist = load_json(LOGS / "cls_scratch_history.json")
    resnet_hist = load_json(LOGS / "cls_resnet18_history.json")
    scratch_metrics = load_json(LOGS / "cls_scratch_test_metrics.json")
    resnet_metrics = load_json(LOGS / "cls_resnet18_test_metrics.json")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Comparison: Custom CNN Baseline vs. Pretrained ResNet-18", fontsize=16, fontweight="bold")

    # --- Panel 1: Loss Trajectory ---
    if "train_loss" in scratch_hist:
        epochs = range(1, len(scratch_hist["train_loss"]) + 1)
        axes[0].plot(epochs, scratch_hist.get("val_loss", []), "r--", label="Baseline Val Loss", linewidth=2)
        axes[0].plot(epochs, scratch_hist["train_loss"], "r-", label="Baseline Train Loss", alpha=0.4)

    if "train_loss" in resnet_hist:
        epochs = range(1, len(resnet_hist["train_loss"]) + 1)
        axes[0].plot(epochs, resnet_hist.get("val_loss", []), "b--", label="ResNet-18 Val Loss", linewidth=2)
        axes[0].plot(epochs, resnet_hist["train_loss"], "b-", label="ResNet-18 Train Loss", alpha=0.4)

    axes[0].set_title("Training & Validation Loss Trajectory")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- Panel 2: Validation Accuracy ---
    if "val_acc" in scratch_hist:
        epochs = range(1, len(scratch_hist["val_acc"]) + 1)
        axes[1].plot(epochs, scratch_hist["val_acc"], "r-o", label="Baseline Val Acc", linewidth=2)

    if "val_acc" in resnet_hist:
        epochs = range(1, len(resnet_hist["val_acc"]) + 1)
        axes[1].plot(epochs, resnet_hist["val_acc"], "b-o", label="ResNet-18 Val Acc", linewidth=2)

    axes[1].set_title("Validation Accuracy Convergence")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # --- Panel 3: Test Set Evaluation Metrics ---
    metrics_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Macro F1"]

    scratch_scores = [scratch_metrics.get(m, 0.0) for m in metrics_keys]
    resnet_scores = [resnet_metrics.get(m, 0.0) for m in metrics_keys]

    x = range(len(metric_labels))
    width = 0.35

    rects1 = axes[2].bar([i - width / 2 for i in x], scratch_scores, width, label="Baseline CNN", color="#d9534f")
    rects2 = axes[2].bar([i + width / 2 for i in x], resnet_scores, width, label="ResNet-18", color="#0275d8")

    axes[2].set_title("Test Set Evaluation Metrics")
    axes[2].set_ylabel("Score")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(metric_labels)
    axes[2].set_ylim(0, 1.0)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.6)

    for rect in rects1 + rects2:
        height = rect.get_height()
        if height > 0:
            axes[2].annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = FIGURES / "baseline_vs_resnet18_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nModel comparison figure saved to:\n  {out_path}\n")


if __name__ == "__main__":
    plot_model_comparison()