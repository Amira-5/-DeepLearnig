
from datasets import load_dataset
from torch.utils.data import Dataset



TRAIN_REPO = "Multimodal-Fatima/CUB_train"
TEST_REPO = "Multimodal-Fatima/CUB_test"


def load_cub_raw(split):
    """Load the requested CUB dataset split from Hugging Face."""
    if split == "train":
        return load_dataset(TRAIN_REPO, split="train")
    if split == "test":
        return load_dataset(TEST_REPO, split="test")

    raise ValueError("split must be 'train' or 'test'")


def split_captions(description):
    """Split a multiline description into individual captions."""
    captions = []

    for caption in description.split("\n"):
        caption = caption.strip()
        if caption:
            captions.append(caption)

    return captions


class CUBDataset(Dataset):
    """PyTorch Dataset for CUB images and classification labels."""

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        sample = self.hf_dataset[index]

        # Load the image and its class label
        image = sample["image"].convert("RGB")
        label = int(sample["label"])

        # Apply preprocessing when provided
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "description": sample["description"],
            "file_name": sample["file_name"],
            "id": sample["id"],
        }

class CUBDataset(Dataset):
    """Classification dataset: returns (image_tensor, label)."""

    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item["label"]