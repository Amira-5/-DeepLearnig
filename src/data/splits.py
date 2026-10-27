from sklearn.model_selection import train_test_split


def make_train_val_indices(labels, val_size=0.2, seed=42):
    """Stratified split: every class appears in both train and validation."""
    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, stratify=labels, random_state=seed
    )
    return train_idx, val_idx