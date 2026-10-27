
"""Reusable local CUB dataset for all three tasks."""

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CUBProcessedDataset(Dataset):
    """Load processed CUB data for classification, colorization, or captioning."""

    # Supported tasks
    VALID_TASKS = {
        "classification",
        "colorization",
        "captioning",
        "all",
    }

    def __init__(
        self,
        metadata_csv,
        project_root=None,
        task="all",
        image_transform=None,
        grayscale_transform=None,
        target_transform=None,
    ):
        # Check that the selected task is valid
        if task not in self.VALID_TASKS:
            raise ValueError(
                f"task must be one of {sorted(self.VALID_TASKS)}"
            )

        # Get the path of the metadata CSV file
        self.csv = Path(metadata_csv).resolve()

        # Set the project root directory
        self.root = (
            Path(project_root).resolve()
            if project_root
            else self.csv.parents[3]
        )

        # Read the metadata and replace missing values with empty strings
        self.rows = pd.read_csv(self.csv).fillna("")

        # Keep only images that have captions when using the captioning task
        if task == "captioning":
            self.rows = self.rows[
                self.rows["caption_count"].astype(int) > 0
            ].reset_index(drop=True)

        # Save the task and image transformations
        self.task = task
        self.image_transform = image_transform
        self.grayscale_transform = grayscale_transform
        self.target_transform = target_transform

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.rows)

    def __getitem__(self, index):
        # Get the metadata for the selected image
        row = self.rows.iloc[index]

        # Open the image and convert it to RGB
        image_path = self.root / str(row["image_path"])

        with Image.open(image_path) as opened:
            rgb = opened.convert("RGB")

        # Load the image captions from the JSON string
        captions = (
            json.loads(row["captions_json"])
            if row["captions_json"]
            else []
        )

        # Store the basic information about the image
        output = {
            "file_name": str(row["file_name"]),
            "image_id": str(row["image_id"]),
            "label": torch.tensor(
                int(row["label"]),
                dtype=torch.long,
            ),
            "captions": captions,
        }

        # Prepare the RGB image for classification or captioning
        if self.task in {"classification", "captioning", "all"}:
            output["image"] = (
                self.image_transform(rgb)
                if self.image_transform
                else rgb
            )

        # Prepare the grayscale image and its color target
        if self.task in {"colorization", "all"}:
            # Convert the original image to grayscale
            gray = rgb.convert("L")

            # Apply transformations if provided
            gray = (
                self.grayscale_transform(gray)
                if self.grayscale_transform
                else gray
            )

            # Prepare the original RGB image as the color target
            target = (
                self.target_transform(rgb)
                if self.target_transform
                else rgb
            )

            # Return the required data for the selected task
            if self.task == "colorization":
                output.update(
                    image=gray,
                    target=target,
                )
            else:
                output.update(
                    grayscale_image=gray,
                    color_target=target,
                )

        # Return the prepared image data and metadata
        return output
