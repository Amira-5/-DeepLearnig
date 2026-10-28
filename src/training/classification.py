import json
from pathlib import Path

import torch
from src.data.loaders import get_classification_loaders
from src.models.classifier import ImageClassifier
from src.models.encoder import SimpleCNNEncoder, TransferLearningEncoder
from src.training.trainer import fit


def build_model(cfg: dict) -> ImageClassifier:
    """Build classification model based on config settings[cite: 2]."""
    m = cfg["model"]
    model_name = m["name"].lower()

    if model_name == "simple_cnn":
        encoder = SimpleCNNEncoder()
    elif model_name in {"resnet18", "resnet50", "efficientnet_b0"}:
        encoder = TransferLearningEncoder(
            name=model_name,
            pretrained=m.get("pretrained", True),
            freeze_backbone=m.get("freeze_backbone", False),
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    return ImageClassifier(
        encoder=encoder,
        num_classes=m.get("num_classes", 200),
        dropout=m.get("dropout", 0.3),
    )


def train_classification(cfg: dict) -> dict:
    d, t, o = cfg["data"], cfg["train"], cfg["output"]

    train_loader, val_loader, _ = get_classification_loaders(
        batch_size=d["batch_size"],
        num_workers=d["num_workers"],
        image_size=d["image_size"],
    )

    model = build_model(cfg)

    history = fit(
        model,
        train_loader,
        val_loader,
        epochs=t["epochs"],
        lr=t["lr"],
        weight_decay=t["weight_decay"],
        checkpoint_path=o["checkpoint"],
        max_batches=t.get("max_batches"),
    )

    Path(o["history"]).parent.mkdir(parents=True, exist_ok=True)
    with open(o["history"], "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history