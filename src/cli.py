import argparse

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(prog="src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model")
    train_p.add_argument("--task", required=True, choices=["classification"])
    train_p.add_argument("--config", required=True)
    train_p.add_argument("--epochs", type=int, default=None)
    train_p.add_argument("--max-batches", type=int, default=None)

    args = parser.parse_args()

    if args.command == "train":
        cfg = load_config(args.config)
        if args.epochs is not None:
            cfg["train"]["epochs"] = args.epochs
        if args.max_batches is not None:
            cfg["train"]["max_batches"] = args.max_batches

        if args.task == "classification":
            from src.training.classification import train_classification
            train_classification(cfg)


if __name__ == "__main__":
    main()