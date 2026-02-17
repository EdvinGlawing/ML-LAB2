import argparse
import csv
import os
import torch

from dataset import CIFAR10Dataset
from model import SimpleCNN
from train import train
from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate CNN on CIFAR-10")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--experiment", type=str, default="exp", help="Experiment name")
    return parser.parse_args()


def select_device():
    """
    Select the best available device.
    Prefer CUDA if available and usable.
    Fall back to CPU otherwise.
    """

    if torch.cuda.is_available():
        try:
            # Test a small CUDA operation to verify compatibility
            x = torch.tensor([1.0], device="cuda")
            y = x * 2
            print("Using GPU:", torch.cuda.get_device_name(0))
            return torch.device("cuda")
        except Exception as e:
            print("CUDA detected but unusable.")
            print("Reason:", e)
            print("Falling back to CPU.")

    print("Using CPU")
    return torch.device("cpu")


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

    device = select_device()

    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)

    model = SimpleCNN()

    model = train(
        model,
        train_dataset,
        device,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    accuracy = evaluate(model, test_dataset, device)

    log_results(
        args.experiment,
        args.lr,
        args.batch_size,
        args.epochs,
        accuracy,
    )

    print("Final Accuracy:", accuracy)


if __name__ == "__main__":
    main()