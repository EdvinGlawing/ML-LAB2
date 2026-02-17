import argparse
import torch
import csv
import os

from dataset import CIFAR10Dataset
from model import SimpleCNN
from train import train
from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--experiment", type=str, default="exp")
    return parser.parse_args()


def log_results(experiment, lr, batch_size, epochs, accuracy):
    os.makedirs("experiments", exist_ok=True)
    file_path = "experiments/results.csv"

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["experiment", "lr", "batch_size", "epochs", "accuracy"])

        writer.writerow([experiment, lr, batch_size, epochs, accuracy])


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)

    model = SimpleCNN()

    model = train(
        model,
        train_dataset,
        device,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    accuracy = evaluate(model, test_dataset, device)

    log_results(
        args.experiment,
        args.lr,
        args.batch_size,
        args.epochs,
        accuracy
    )

    print("Final Accuracy:", accuracy)


if __name__ == "__main__":
    main()
