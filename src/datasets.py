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

    if dataset_name == "svhn":
        train_dataset = datasets.SVHN(
            root=data_root, split="train", transform=transform, download=True
        )
        test_dataset = datasets.SVHN(
            root=data_root, split="test", transform=transform, download=True
        )
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, transform=transform, download=True
        )
        test_dataset = datasets.CIFAR10(
            root=data_root, train=False, transform=transform, download=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
