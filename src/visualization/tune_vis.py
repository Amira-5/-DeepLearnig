import sys
from pathlib import Path

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

LOGS = ROOT / "outputs" / "logs"
FIGURES = ROOT / "outputs" / "figures"


def visualize_tuning():
    FIGURES.mkdir(parents=True, exist_ok=True)
    results_path = LOGS / "hyperparameter_tuning_results.csv"

    if not results_path.exists():
        print(f"Error: Could not find results file at {results_path}")
        return

    df = pd.read_csv(results_path)

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hyperparameter Tuning Analysis: Grid Search vs. Random Search", fontsize=16, fontweight="bold")

    # 1. Search Convergence (Trial vs Cumulative Best Val Acc)
    for method, group in df.groupby("method"):
        group = group.sort_values("trial")
        cum_max = group["val_acc"].cummax()
        axes[0, 0].plot(
            group["trial"],
            cum_max,
            marker="o",
            linewidth=2,
            label=f"{method} (Cumulative Best)",
        )
        axes[0, 0].scatter(
            group["trial"], group["val_acc"], alpha=0.35, label=f"{method} (Raw Trial)"
        )

    axes[0, 0].set_title("Search Convergence Speed")
    axes[0, 0].set_xlabel("Trial Number")
    axes[0, 0].set_ylabel("Validation Accuracy")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend(fontsize=9)

    # 2. Learning Rate vs Validation Accuracy
    for method, group in df.groupby("method"):
        axes[0, 1].scatter(
            group["lr"],
            group["val_acc"],
            s=90,
            alpha=0.8,
            label=method,
            edgecolors="k",
        )

    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Learning Rate Sensitivity (Log Scale)")
    axes[0, 1].set_xlabel("Learning Rate")
    axes[0, 1].set_ylabel("Validation Accuracy")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].legend()

    # 3. Batch Size Impact
    batch_summary = df.groupby(["batch_size", "method"])["val_acc"].mean().unstack()
    batch_summary.plot(kind="bar", ax=axes[1, 0], rot=0, edgecolor="black")
    axes[1, 0].set_title("Mean Validation Accuracy by Batch Size")
    axes[1, 0].set_xlabel("Batch Size")
    axes[1, 0].set_ylabel("Mean Val Accuracy")
    axes[1, 0].grid(True, axis="y", linestyle="--", alpha=0.6)

    # 4. Best Accuracy Comparison
    summary = (
        df.groupby("method")
        .agg(best_acc=("val_acc", "max"), total_time=("time_sec", "sum"))
        .reset_index()
    )

    bars = axes[1, 1].bar(
        summary["method"],
        summary["best_acc"],
        color=["#1f77b4", "#ff7f0e"],
        edgecolor="black",
        width=0.4,
    )
    axes[1, 1].set_title("Top Validation Accuracy Reached")
    axes[1, 1].set_ylabel("Best Validation Accuracy")
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.02,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = FIGURES / "hyperparameter_tuning_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nVisualization saved successfully to:\n  {out_path}\n")


if __name__ == "__main__":
    visualize_tuning()