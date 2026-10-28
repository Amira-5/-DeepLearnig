import itertools
import json
import random
import time
from pathlib import Path
import pandas as pd

from src.data.loaders import get_classification_loaders
from src.training.classification import build_model
from src.training.trainer import fit


ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "outputs" / "logs"

# Hyperparameter Search Space
GRID_SPACE = {
    "lr": [1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4],
    "batch_size": [16, 32],
}

RANDOM_BOUNDS = {
    "lr": (1e-4, 1e-3),
    "weight_decay": (1e-5, 1e-3),
    "batch_size": [16, 32],
}


def run_trial(cfg_template, lr, weight_decay, batch_size, epochs=3):
    """Executes a single training trial with given hyperparameters."""
    cfg = cfg_template.copy()
    cfg["data"]["batch_size"] = batch_size
    cfg["train"]["lr"] = lr
    cfg["train"]["weight_decay"] = weight_decay
    cfg["train"]["epochs"] = epochs

    train_loader, val_loader, _ = get_classification_loaders(
        batch_size=batch_size,
        num_workers=cfg["data"].get("num_workers", 2),
        image_size=cfg["data"].get("image_size", 224),
    )

    model = build_model(cfg)
    history = fit(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        checkpoint_path=None,
    )

    best_val_acc = max(history["val_acc"])
    return best_val_acc


def run_grid_search(cfg_template, epochs=3):
    """Exhaustively searches all grid combinations."""
    keys, values = zip(*GRID_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n--- Starting Grid Search ({len(combinations)} Trials) ---")
    results = []
    start_time = time.time()

    for idx, combo in enumerate(combinations, 1):
        print(f"\n[Grid Trial {idx}/{len(combinations)}] Params: {combo}")
        t0 = time.time()
        best_acc = run_trial(
            cfg_template, combo["lr"], combo["weight_decay"], combo["batch_size"], epochs
        )
        elapsed = time.time() - t0
        results.append({
            "trial": idx,
            "method": "Grid Search",
            "lr": combo["lr"],
            "weight_decay": combo["weight_decay"],
            "batch_size": combo["batch_size"],
            "val_acc": best_acc,
            "time_sec": elapsed,
        })

    total_time = time.time() - start_time
    return results, total_time


def run_random_search(cfg_template, num_trials=6, epochs=3):
    """Randomly samples hyperparameter combinations within bounds."""
    print(f"\n--- Starting Random Search ({num_trials} Trials) ---")
    results = []
    start_time = time.time()

    for idx in range(1, num_trials + 1):
        # Sample parameters
        lr = round(10 ** random.uniform(-4, -3), 6)
        weight_decay = round(10 ** random.uniform(-5, -3), 6)
        batch_size = random.choice(RANDOM_BOUNDS["batch_size"])

        combo = {"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size}
        print(f"\n[Random Trial {idx}/{num_trials}] Params: {combo}")
        t0 = time.time()
        best_acc = run_trial(
            cfg_template, lr, weight_decay, batch_size, epochs
        )
        elapsed = time.time() - t0
        results.append({
            "trial": idx,
            "method": "Random Search",
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "val_acc": best_acc,
            "time_sec": elapsed,
        })

    total_time = time.time() - start_time
    return results, total_time


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    
    # Base config template
    cfg_template = {
        "model": {"name": "simple_cnn", "num_classes": 200, "dropout": 0.3},
        "data": {"batch_size": 32, "image_size": 224, "num_workers": 2},
        "train": {"epochs": 3, "lr": 0.001, "weight_decay": 0.0001},
        "output": {},
    }

    # 1. Run Grid Search
    grid_results, grid_time = run_grid_search(cfg_template, epochs=3)
    
    # 2. Run Random Search
    random_results, random_time = run_random_search(cfg_template, num_trials=len(grid_results), epochs=3)

    # 3. Aggregate & Save Summary Comparison
    all_results = grid_results + random_results
    df = pd.DataFrame(all_results)
    df.to_csv(LOGS / "hyperparameter_tuning_results.csv", index=False)

    # Find best configurations
    best_grid = max(grid_results, key=lambda x: x["val_acc"])
    best_random = max(random_results, key=lambda x: x["val_acc"])

    summary = {
        "Grid Search": {
            "num_trials": len(grid_results),
            "total_time_sec": round(grid_time, 2),
            "best_val_acc": round(best_grid["val_acc"], 4),
            "best_params": {
                "lr": best_grid["lr"],
                "weight_decay": best_grid["weight_decay"],
                "batch_size": best_grid["batch_size"],
            },
        },
        "Random Search": {
            "num_trials": len(random_results),
            "total_time_sec": round(random_time, 2),
            "best_val_acc": round(best_random["val_acc"], 4),
            "best_params": {
                "lr": best_random["lr"],
                "weight_decay": best_random["weight_decay"],
                "batch_size": best_random["batch_size"],
            },
        },
    }

    with open(LOGS / "hyperparameter_tuning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Output Summary Table
    print("\n" + "=" * 60)
    print("        HYPERPARAMETER TUNING COMPARISON SUMMARY        ")
    print("=" * 60)
    summary_df = pd.DataFrame([
        {
            "Method": "Grid Search",
            "Trials": len(grid_results),
            "Time (s)": round(grid_time, 1),
            "Best Val Acc": f"{best_grid['val_acc']:.4f}",
            "Best Params (LR, WD, BS)": f"({best_grid['lr']}, {best_grid['weight_decay']}, {best_grid['batch_size']})",
        },
        {
            "Method": "Random Search",
            "Trials": len(random_results),
            "Time (s)": round(random_time, 1),
            "Best Val Acc": f"{best_random['val_acc']:.4f}",
            "Best Params (LR, WD, BS)": f"({best_random['lr']}, {best_random['weight_decay']}, {best_random['batch_size']})",
        },
    ])
    print(summary_df.to_string(index=False))
    print("=" * 60)
    print(f"\nDetailed logs saved to:\n  - {LOGS / 'hyperparameter_tuning_results.csv'}\n  - {LOGS / 'hyperparameter_tuning_summary.json'}\n")


if __name__ == "__main__":
    main()