import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history_path, out_path, title="Training curves"):
    with open(history_path, "r") as f:
        h = json.load(f)
    epochs = range(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, h["train_loss"], label="train")
    axes[0].plot(epochs, h["val_loss"], label="validation")
    axes[0].set_title(f"{title}: loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, h["train_acc"], label="train")
    axes[1].plot(epochs, h["val_acc"], label="validation")
    axes[1].set_title(f"{title}: accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Training curves")
    args = parser.parse_args()
    plot_history(args.history, args.out, args.title)