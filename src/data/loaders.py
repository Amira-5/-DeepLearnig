from torch.utils.data import DataLoader, Subset

from .dataset import CUBDataset, load_cub_raw
from .splits import make_train_val_indices
from .transforms import get_eval_transform, get_train_transform


def get_classification_loaders(batch_size=32, num_workers=0, val_size=0.2,
                               seed=42, image_size=224):
    train_raw = load_cub_raw("train")
    test_raw = load_cub_raw("test")

    train_idx, val_idx = make_train_val_indices(train_raw["label"], val_size, seed)

    # same images, different transforms: augmentation for train only
    train_ds = Subset(CUBDataset(train_raw, get_train_transform(image_size)), train_idx)
    val_ds = Subset(CUBDataset(train_raw, get_eval_transform(image_size)), val_idx)
    test_ds = CUBDataset(test_raw, get_eval_transform(image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader