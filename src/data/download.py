"""Download CUB-200-2011 (train + test) and check it is correct.

Run from the project root:
    python -m src.data.download
"""
from collections import Counter

from src.data.dataset import load_cub_raw, split_captions

EXPECTED_SIZES = {"train": 5994, "test": 5794}


def main():
    for split, expected in EXPECTED_SIZES.items():
        print(f"Loading {split} split (first time downloads it)...")
        data = load_cub_raw(split)

        size = len(data)
        status = "OK" if size == expected else f"WARNING: expected {expected}"
        print(f"  images: {size} ({status})")

        counts = Counter(len(split_captions(d)) for d in data["description"])
        print(f"  captions per image: {dict(counts)}")
        print(f"  classes: {data.features['label'].num_classes}")

    print("Done. The data is ready on this machine.")


if __name__ == "__main__":
    main()