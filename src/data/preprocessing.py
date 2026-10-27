from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# 1. Project Paths and Settings

# Get the project root directory
ROOT = Path(__file__).resolve().parents[2]

# Define the data directories
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
IMAGES = PROC / "images"
META = PROC / "metadata"
REPORTS = ROOT / "outputs" / "eda"

# Short cache path to avoid the Windows 260-character path limit.
# Can be overridden with the HF_CACHE_DIR environment variable.
HF_CACHE = os.environ.get("HF_CACHE_DIR", "C:/hf_cache")

# Hugging Face dataset repositories
TRAIN_REPO = "Multimodal-Fatima/CUB_train"
TEST_REPO = "Multimodal-Fatima/CUB_test"

# Settings for the train-validation split
RANDOM_STATE = 42
VAL_SIZE = 0.20

# Set the logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


# 2. Caption Cleaning

def clean_caption(value: Any) -> str:
    """Remove extra spaces from a caption."""

    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()

    return ""


def split_captions(description: Any) -> list[str]:
    """Split captions by line and remove empty captions."""

    if not isinstance(description, str):
        return []

    return [
        caption
        for line in description.splitlines()
        if (caption := clean_caption(line))
    ]


# 3. Image Loading and Hashing

def get_rgb(record):
    """Load an image and convert it to RGB."""

    image = record["image"]

    # Handle images that are already loaded
    if isinstance(image, Image.Image):
        image.load()
        return image.convert("RGB")

    # Load images from a file path or file-like object
    with Image.open(image) as opened:
        opened.load()
        return opened.convert("RGB")


def digest_image(image):
    """Create a hash to identify exact pixel duplicates."""

    rgb = image.convert("RGB")

    # Include the image dimensions and pixel data in the hash
    image_data = (
        f"{rgb.width}x{rgb.height}|".encode()
        + rgb.tobytes()
    )

    return hashlib.sha256(image_data).hexdigest()


# 4. Scan and Clean the Dataset

def scan(train_ds, test_ds):
    """
    Analyze the dataset and exclude unreadable images,
    missing labels, and exact pixel duplicates.
    """

    REPORTS.mkdir(parents=True, exist_ok=True)

    # Store accepted images and excluded records
    accepted = []
    excluded = []
    duplicates = []

    # Store statistics for the EDA reports
    seen = {}
    sizes = Counter()
    class_counts = Counter()
    vocab = Counter()
    caption_lengths = []

    # Process the training and test splits
    for split, dataset in [
        ("train", train_ds),
        ("test", test_ds),
    ]:
        for index in tqdm(
            range(len(dataset)),
            desc=f"EDA {split}",
        ):
            record = dataset[index]

            # Collect the basic information about the record
            item = {
                "source_split": split,
                "source_index": index,
                "image_id": str(record.get("id", index)),
                "file_name": str(
                    record.get("file_name", f"{split}_{index}.jpg")
                ),
                "label": record.get("label"),
                "captions": split_captions(
                    record.get("description")
                ),
            }

            # Exclude records without a class label
            if item["label"] is None:
                excluded.append({
                    **item,
                    "reason": "missing_label",
                })
                continue

            item["label"] = int(item["label"])

            # Collect caption word statistics
            for caption in item["captions"]:
                words = re.findall(
                    r"\b[\w'-]+\b",
                    caption.lower(),
                )

                vocab.update(words)
                caption_lengths.append(len(words))

            try:
                # Load the image and calculate its hash
                image = get_rgb(record)
                image_hash = digest_image(image)

                # Store image information
                item.update(
                    width=image.width,
                    height=image.height,
                    pixel_sha256=image_hash,
                    caption_count=len(item["captions"]),
                )

                # Update image statistics
                sizes[(image.width, image.height)] += 1
                class_counts[(split, item["label"])] += 1

                # Check whether the exact image was seen before
                if image_hash in seen:
                    first = seen[image_hash]

                    duplicates.append({
                        "kept_split": first["source_split"],
                        "kept_index": first["source_index"],
                        "kept_file_name": first["file_name"],
                        "duplicate_split": split,
                        "duplicate_index": index,
                        "duplicate_file_name": item["file_name"],
                        "same_label": (
                            first["label"] == item["label"]
                        ),
                    })

                    # Exclude the duplicate from the processed dataset
                    excluded.append({
                        **item,
                        "reason": "exact_pixel_duplicate",
                        "duplicate_of_split": first["source_split"],
                        "duplicate_of_index": first["source_index"],
                    })

                else:
                    # Keep the first copy of each exact image
                    seen[image_hash] = item
                    accepted.append(item)

            except Exception as exc:
                # Record images that cannot be read
                excluded.append({
                    **item,
                    "reason": "unreadable_image",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    # Save EDA Reports

    # Save information about accepted images
    pd.DataFrame([
        {
            key: value
            for key, value in item.items()
            if key != "captions"
        }
        for item in accepted
    ]).to_csv(
        REPORTS / "accepted_image_inventory.csv",
        index=False,
    )

    # Save excluded records and duplicate details
    pd.DataFrame(excluded).to_json(
        REPORTS / "excluded_items.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )

    pd.DataFrame(duplicates).to_json(
        REPORTS / "exact_duplicates.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )

    # Save the number of images in each class and split
    pd.DataFrame([
        {
            "split": split,
            "label": label,
            "image_count": count,
        }
        for (split, label), count in sorted(class_counts.items())
    ]).to_csv(
        REPORTS / "class_distribution.csv",
        index=False,
    )

    # Create the EDA Summary

    report = {
        "source_train_images": len(train_ds),
        "source_test_images": len(test_ds),
        "accepted_unique_images": len(accepted),
        "excluded_items": len(excluded),

        "unreadable_images": sum(
            item["reason"] == "unreadable_image"
            for item in excluded
        ),

        "missing_labels": sum(
            item["reason"] == "missing_label"
            for item in excluded
        ),

        "exact_duplicate_exclusions": sum(
            item["reason"] == "exact_pixel_duplicate"
            for item in excluded
        ),

        "classes_observed": len({
            item["label"]
            for item in accepted
        }),

        "unique_image_dimensions": len(sizes),

        "most_common_dimensions": [
            {
                "width": width,
                "height": height,
                "count": count,
            }
            for (width, height), count in sizes.most_common(15)
        ],

        "caption_count_distribution": dict(
            sorted(
                Counter(
                    item["caption_count"]
                    for item in accepted
                ).items()
            )
        ),

        "caption_word_count_mean": (
            sum(caption_lengths) / len(caption_lengths)
            if caption_lengths
            else 0
        ),

        "caption_vocabulary_size_lowercase": len(vocab),
        "most_common_caption_words": vocab.most_common(40),

        "caption_cleaning": (
            "Split newline, normalize whitespace, remove blank "
            "captions; no guessed spelling corrections"
        ),

        "duplicate_policy": (
            "Exact RGB pixel hash; first encountered copy retained; "
            "train scanned before test"
        ),
    }

    # Save the summary report as JSON
    (REPORTS / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return accepted, report


# 5. Export Processed Images and Metadata

def export_split(source, rows, name):
    """Save images and create metadata for one dataset split."""

    IMAGES.mkdir(parents=True, exist_ok=True)
    output = []

    for item in tqdm(rows, desc=f"Export {name}"):
        try:
            # Load the original image again
            record = source[item["source_split"]][
                item["source_index"]
            ]
            image = get_rgb(record)

            # Create a safe and unique local filename
            stem = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                Path(item["file_name"]).stem,
            )[:80]

            filename = (
                f'{name}_{item["source_index"]:05d}_{stem}.png'
            )

            image_path = IMAGES / filename

            # Save the image as an RGB PNG
            image.save(
                image_path,
                format="PNG",
                optimize=True,
            )

            # Store the image metadata
            output.append({
                "image_path": image_path.relative_to(
                    ROOT
                ).as_posix(),

                "file_name": item["file_name"],
                "image_id": item["image_id"],
                "label": int(item["label"]),

                "captions_json": json.dumps(
                    item["captions"],
                    ensure_ascii=False,
                ),

                "caption_count": len(item["captions"]),
                "source_split": item["source_split"],
                "source_index": item["source_index"],
                "width": image.width,
                "height": image.height,
                "pixel_sha256": item["pixel_sha256"],
            })

        except Exception as exc:
            # Log export errors and save them for later review
            logging.exception(
                "Export failed for %s",
                item["file_name"],
            )

            with (
                REPORTS / "export_failures.jsonl"
            ).open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps({
                        "file_name": item["file_name"],
                        "reason": repr(exc),
                    }) + "\n"
                )

    # Return the metadata as a DataFrame
    return pd.DataFrame(output)


# 6. Main Preprocessing Pipeline

def main():
    """Run the complete dataset preprocessing pipeline."""

    # Create the required directories
    for directory in (RAW, META, IMAGES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)

    # Create the short HF cache directory
    Path(HF_CACHE).mkdir(parents=True, exist_ok=True)

    # Load the original training and test datasets
    logging.info(
        "Loading official Hugging Face splits; "
        "local cache (%s) will be reused.",
        HF_CACHE,
    )

    train = load_dataset(
        TRAIN_REPO,
        split="train",
        cache_dir=HF_CACHE,
    )

    test = load_dataset(
        TEST_REPO,
        split="test",
        cache_dir=HF_CACHE,
    )

    # Scan the data, remove excluded records, and save EDA reports
    accepted, report = scan(train, test)

    logging.info(
        "EDA summary: %s",
        json.dumps(report, ensure_ascii=False),
    )

    # Separate accepted records by their original split
    train_rows = [
        item for item in accepted
        if item["source_split"] == "train"
    ]

    test_rows = [
        item for item in accepted
        if item["source_split"] == "test"
    ]

    # Split the original training data into train and validation
    try:
        train_rows, val_rows = train_test_split(
            train_rows,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=[
                item["label"]
                for item in train_rows
            ],
        )

    except ValueError as exc:
        raise RuntimeError(
            "Stratified split failed; inspect "
            "outputs/eda/summary.json before changing the policy."
        ) from exc

    # Export all three splits and save their metadata
    source = {
        "train": train,
        "test": test,
    }

    splits = [
        ("train", train_rows),
        ("validation", val_rows),
        ("test", test_rows),
    ]

    for name, rows in splits:
        # Keep the original order within each source split
        rows = sorted(
            rows,
            key=lambda item: item["source_index"],
        )

        # Save the images and their metadata
        frame = export_split(source, rows, name)

        frame.to_csv(
            META / f"{name}.csv",
            index=False,
        )

        logging.info(
            "Saved %s: %s rows",
            name,
            len(frame),
        )

    # Save Preprocessing Information

    info = {
        "sources": {
            "train": TRAIN_REPO,
            "test": TEST_REPO,
        },

        "random_state": RANDOM_STATE,
        "validation_size": VAL_SIZE,

        "images": (
            "Local RGB PNG, original decoded resolution; "
            "resize/normalize at runtime"
        ),

        "grayscale": (
            "Generated from local RGB image at runtime; "
            "no duplicate grayscale files"
        ),

        "cleaning": (
            "Bad images/missing labels/exact pixel duplicates "
            "excluded from processed copy only"
        ),

        "captions": (
            "Split on newline, normalize whitespace, remove "
            "blank captions; spelling not auto-corrected"
        ),

        "caution": (
            "Review outputs/eda/excluded_items.json and "
            "exact_duplicates.json; near-duplicates and semantic "
            "caption errors require review"
        ),
    }

    # Write the preprocessing configuration to JSON
    (PROC / "preprocessing_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logging.info(
        "Finished. Processed dataset: %s",
        PROC,
    )


# 7. Run the Pipeline

if __name__ == "__main__":
    main()