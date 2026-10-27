import json
from pathlib import Path

from src.data.loaders import get_classification_loaders
from src.models.classifier import ImageClassifier
from src.models.encoder import SimpleCNNEncoder

from .trainer import fit


def build_model(cfg):
    m = cfg["model"]
    if m["name"] == "simple_cnn":
        encoder = SimpleCNNEncoder()
    else:
        raise ValueError(f"Unknown model: {m['name']}")
    return ImageClassifier(encoder, m["num_classes"], m["dropout"])


def train_classification(cfg):
    d, t, o = cfg["data"], cfg["train"], cfg["output"]

    train_loader, val_loader, _ = get_classification_loaders(
        batch_size=d["batch_size"],
        num_workers=d["num_workers"],
        image_size=d["image_size"],
    )

    model = build_model(cfg)
    history = fit(
        model, train_loader, val_loader,
        epochs=t["epochs"], lr=t["lr"], weight_decay=t["weight_decay"],
        checkpoint_path=o["checkpoint"], max_batches=t.get("max_batches"),
    )

    Path(o["history"]).parent.mkdir(parents=True, exist_ok=True)
    with open(o["history"], "w") as f:
        json.dump(history, f)
    return history