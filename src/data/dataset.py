from datasets import load_dataset

TRAIN_REPO = "Multimodal-Fatima/CUB_train"
TEST_REPO = "Multimodal-Fatima/CUB_test"


def load_cub_raw(split):
    if split == "train":
        return load_dataset(TRAIN_REPO, split="train")
    if split == "test":
        return load_dataset(TEST_REPO, split="test")
    raise ValueError("split must be 'train' or 'test'")


def split_captions(description):
    captions = []
    for c in description.split("\n"):
        c = c.strip()
        if c:
            captions.append(c)
    return captions