import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.loaders import get_classification_loaders
from src.models.classifier import ImageClassifier
from src.models.encoder import TransferLearningEncoder
from src.training.trainer import get_device

LOGS = ROOT / "outputs" / "logs"
FIGURES = ROOT / "outputs" / "figures"
CHECKPOINT = ROOT / "outputs" / "checkpoints" / "cls_resnet18.pt"


class GradCAM:
    """Computes Grad-CAM heatmaps for the target convolutional layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Target class score for backprop
        score = logits[0, target_class]
        score.backward()

        # Compute importance weights: Global Average Pooling of gradients
        # gradients shape: [1, C, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of forward activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Normalize between 0 and 1
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize heatmap back to input image spatial size (224 x 224)
        cam_resized = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return cam_resized.squeeze().cpu().numpy(), target_class, logits.argmax(dim=1).item()


def denormalize(tensor):
    """Reverses ImageNet mean/std normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def overlay_heatmap(img_np, heatmap, alpha=0.5):
    """Overlays the Jet colormap heatmap onto the RGB image."""
    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap)[:, :, :3]  # Drop alpha channel
    overlay = (1 - alpha) * img_np + alpha * colored_heatmap
    return np.clip(overlay, 0, 1)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Running Grad-CAM on device: {device}")

    # 1. Build Model & Load Checkpoint
    encoder = TransferLearningEncoder(name="resnet18", pretrained=False)
    model = ImageClassifier(encoder, num_classes=200, dropout=0.3)

    if not CHECKPOINT.exists():
        print(f"Error: Could not find checkpoint at {CHECKPOINT}")
        return

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    # Target final convolutional layer (last residual block of ResNet-18)
    # encoder.features contains the resnet layers up to layer4
    target_layer = list(encoder.features.children())[-1]
    grad_cam = GradCAM(model, target_layer)

    # 2. Get Test Data Loader
    _, _, test_loader = get_classification_loaders(
        batch_size=1, num_workers=0, image_size=224
    )

    correct_samples = []
    incorrect_samples = []

    # Find samples for visualization
    for batch in test_loader:
        img_tensor = batch["image"].to(device)
        true_label = batch["label"].item()

        heatmap, target_cls, pred_cls = grad_cam.generate_heatmap(img_tensor, target_class=true_label)

        sample_info = {
            "tensor": img_tensor,
            "true": true_label,
            "pred": pred_cls,
            "heatmap": heatmap,
        }

        if true_label == pred_cls and len(correct_samples) < 3:
            correct_samples.append(sample_info)
        elif true_label != pred_cls and len(incorrect_samples) < 3:
            incorrect_samples.append(sample_info)

        if len(correct_samples) == 3 and len(incorrect_samples) == 3:
            break

    # 3. Plot Analysis Grid (Correct vs. Incorrect Predictions)
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    fig.suptitle("Grad-CAM Explainability Analysis: ResNet-18", fontsize=16, fontweight="bold")

    # Row 0: Correct Predictions
    for idx, sample in enumerate(correct_samples):
        img_np = denormalize(sample["tensor"].squeeze(0))
        overlay = overlay_heatmap(img_np, sample["heatmap"])

        # Original
        axes[0, idx * 2].imshow(img_np)
        axes[0, idx * 2].set_title(f"Correct (Class {sample['true']})\nOriginal", fontsize=9, color="green", fontweight="bold")
        axes[0, idx * 2].axis("off")

        # Grad-CAM Overlay
        axes[0, idx * 2 + 1].imshow(overlay)
        axes[0, idx * 2 + 1].set_title(f"Attention Heatmap\n(True: {sample['true']})", fontsize=9)
        axes[0, idx * 2 + 1].axis("off")

    # Row 1: Incorrect Predictions
    for idx, sample in enumerate(incorrect_samples):
        img_np = denormalize(sample["tensor"].squeeze(0))
        overlay = overlay_heatmap(img_np, sample["heatmap"])

        # Original
        axes[1, idx * 2].imshow(img_np)
        axes[1, idx * 2].set_title(f"Error (True: {sample['true']})\nOriginal", fontsize=9, color="red", fontweight="bold")
        axes[1, idx * 2].axis("off")

        # Grad-CAM Overlay
        axes[1, idx * 2 + 1].imshow(overlay)
        axes[1, idx * 2 + 1].set_title(f"Attention Heatmap\n(Pred: {sample['pred']})", fontsize=9)
        axes[1, idx * 2 + 1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = FIGURES / "gradcam_analysis.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nGrad-CAM heatmaps saved to:\n  {out_path}\n")


if __name__ == "__main__":
    main()