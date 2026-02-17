import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.dataset = datasets.CIFAR10(
            root="data/raw",
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label
