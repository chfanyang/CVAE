import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(config):
    dataset_name = config["dataset"]
    data_root = config["data_root"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    pin_memory = torch.cuda.is_available()

    transform = transforms.ToTensor()

    # Only download if dataset is not already present
    download = not os.path.isdir(os.path.join(data_root, dataset_name))

    if dataset_name == "svhn":
        train_dataset = datasets.SVHN(
            root=data_root, split="train", transform=transform, download=download
        )
        test_dataset = datasets.SVHN(
            root=data_root, split="test", transform=transform, download=download
        )
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, transform=transform, download=download
        )
        test_dataset = datasets.CIFAR10(
            root=data_root, train=False, transform=transform, download=download
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, test_loader
