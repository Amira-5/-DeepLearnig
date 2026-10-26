import os
import matplotlib.pyplot as plt

from src.data.dataset import load_cub_raw, split_captions


def show_samples(split="train", n=10, out_dir="outputs/figures"):
    data = load_cub_raw(split)
    label_names = data.features["label"].names

    os.makedirs(out_dir, exist_ok=True)

    for i in range(n):
        item = data[i]
        caption = split_captions(item["description"])[0]

        plt.figure(figsize=(5, 5))
        plt.imshow(item["image"])
        plt.axis("off")
        plt.title(f"{label_names[item['label']]}\n{caption}", fontsize=8)
        plt.savefig(f"{out_dir}/sample_{i}.png", bbox_inches="tight")
        plt.close()

    print(f"Saved {n} images to {out_dir}")


if __name__ == "__main__":
    show_samples()