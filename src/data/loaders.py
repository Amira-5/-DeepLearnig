from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import CUBProcessedDataset
from src.data.transforms import get_eval_transform, get_train_transform

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "processed" / "metadata"


def _make_loader(csv_name, transform, batch_size, shuffle, num_workers):
    ds = CUBProcessedDataset(
        META / csv_name,
        project_root=ROOT,
        task="classification",
        image_transform=transform,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def get_classification_loaders(batch_size=32, num_workers=4, image_size=224):
    train = _make_loader("train.csv", get_train_transform(image_size),
                         batch_size, True, num_workers)
    val = _make_loader("validation.csv", get_eval_transform(image_size),
                       batch_size, False, num_workers)
    test = _make_loader("test.csv", get_eval_transform(image_size),
                        batch_size, False, num_workers)
    return train, val, test