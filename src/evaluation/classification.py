import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from src.data.loaders import get_classification_loaders
from src.training.classification import build_model
from src.training.trainer import get_device
from src.visualization.confusion import plot_confusion_matrix

from .metrics import compute_classification_metrics


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)
        ps.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(batch["label"].numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluate_classification(cfg, split="test"):
    d, o = cfg["data"], cfg["output"]

    _, val_loader, test_loader = get_classification_loaders(
        batch_size=d["batch_size"], num_workers=d["num_workers"],
        image_size=d["image_size"],
    )
    loader = test_loader if split == "test" else val_loader

    device = get_device()
    model = build_model(cfg)
    model.load_state_dict(torch.load(o["checkpoint"], map_location=device))
    model.to(device)

    y_true, y_pred = predict_loader(model, loader, device)
    metrics = compute_classification_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg["model"]["num_classes"])))

    name = Path(o["checkpoint"]).stem
    metrics_path = Path(f"outputs/logs/{name}_{split}_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm, f"outputs/figures/{name}_{split}_confusion.png",
                          title=f"{name} ({split}): confusion matrix")

    print(f"Evaluated on {split}: {len(y_true)} images")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics