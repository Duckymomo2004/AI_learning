from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .common import get_data_root


def _load_torchvision():
    from torchvision import datasets, transforms

    return datasets, transforms


def _default_cifar_transforms(image_size: int = 32):
    _, transforms = _load_torchvision()
    resize = [transforms.Resize((image_size, image_size))] if image_size != 32 else []
    train_transform = transforms.Compose(
        resize
        + [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        resize
        + [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    return train_transform, test_transform


def build_cifar10_dataloaders(
    batch_size: int = 128,
    root: str | Path | None = None,
    image_size: int = 32,
    download: bool = True,
    num_workers: int = 0,
) -> tuple[Dict[str, DataLoader], Dict[str, int], list[str]]:
    data_root = get_data_root(root)
    datasets, _ = _load_torchvision()
    train_transform, test_transform = _default_cifar_transforms(image_size=image_size)

    train_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=download,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=download,
        transform=test_transform,
    )

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}
    return dataloaders, dataset_sizes, list(train_dataset.classes)


def build_imagefolder_dataloaders(
    train_dir: str | Path,
    val_dir: str | Path | None = None,
    batch_size: int = 8,
    image_size: int = 224,
    train_split: float = 0.8,
    subset_size: int | None = None,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[Dict[str, DataLoader], Dict[str, int], list[str]]:
    train_dir = Path(train_dir)
    val_dir = Path(val_dir) if val_dir is not None else None
    datasets, transforms = _load_torchvision()

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_full = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    class_names = list(train_full.classes)

    if val_dir is None:
        full_dataset: Dataset = train_full
    else:
        val_full = datasets.ImageFolder(root=str(val_dir), transform=test_transform)
        full_dataset = torch.utils.data.ConcatDataset([train_full, val_full])

    if subset_size is not None:
        subset_size = min(subset_size, len(full_dataset))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(full_dataset), generator=generator)[:subset_size].tolist()
        full_dataset = Subset(full_dataset, indices)

    train_size = math.floor(len(full_dataset) * train_split)
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_data, test_data = random_split(full_dataset, [train_size, test_size], generator=generator)

    dataloaders = {
        "train": DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "test": DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    dataset_sizes = {"train": len(train_data), "test": len(test_data)}
    return dataloaders, dataset_sizes, class_names


class RandomImageDataset(Dataset):
    def __init__(self, size: int, num_classes: int, image_size: tuple[int, int, int], seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, *image_size, generator=generator)
        self.labels = torch.randint(0, num_classes, (size,), generator=generator)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def build_fake_classification_dataloaders(
    num_classes: int = 10,
    image_size: tuple[int, int, int] = (3, 32, 32),
    train_size: int = 64,
    test_size: int = 32,
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[Dict[str, DataLoader], Dict[str, int], list[str]]:
    train_dataset = RandomImageDataset(train_size, num_classes, image_size, seed=0)
    test_dataset = RandomImageDataset(test_size, num_classes, image_size, seed=1)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}
    class_names = [f"class_{idx}" for idx in range(num_classes)]
    return dataloaders, dataset_sizes, class_names
