from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, Sampler

_TRANSFORM_TYPE_DICT = {
    "Brightness": ImageEnhance.Brightness,
    "Contrast": ImageEnhance.Contrast,
    "Sharpness": ImageEnhance.Sharpness,
    "Color": ImageEnhance.Color,
}

def _load_transforms():
    from torchvision import transforms

    return transforms


class ImageJitter:
    def __init__(self, transform_dict: dict[str, float]):
        self.transforms = [(_TRANSFORM_TYPE_DICT[name], alpha) for name, alpha in transform_dict.items()]

    def __call__(self, image: Image.Image) -> Image.Image:
        output = image
        rand_tensor = torch.rand(len(self.transforms))
        for index, (transformer, alpha) in enumerate(self.transforms):
            factor = alpha * (rand_tensor[index] * 2.0 - 1.0) + 1.0
            output = transformer(output).enhance(float(factor)).convert("RGB")
        return output


class EpisodicBatchSampler(Sampler[list[int]]):
    def __init__(self, n_classes: int, n_way: int, n_episodes: int):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self) -> int:
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[: self.n_way].tolist()


def build_episode_transform(
    image_size: int,
    aug: bool = False,
    normalize_param: dict[str, Sequence[float]] | None = None,
    jitter_param: dict[str, float] | None = None,
):
    transforms = _load_transforms()
    normalize_param = normalize_param or {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    jitter_param = jitter_param or {"Brightness": 0.4, "Contrast": 0.4, "Color": 0.4}
    transform_names = (
        ["Resize", "RandomResizedCrop", "ColorJitter", "RandomHorizontalFlip", "ToTensor", "Normalize"]
        if aug
        else ["Resize", "CenterCrop", "ToTensor", "Normalize"]
    )

    def build_transform(name: str):
        if name == "ImageJitter":
            return ImageJitter(jitter_param)
        method = getattr(transforms, name)
        if name in {"RandomResizedCrop", "Resize", "CenterCrop"}:
            return method(image_size)
        if name == "Normalize":
            return method(**normalize_param)
        return method()

    return transforms.Compose([build_transform(name) for name in transform_names])


class SetDataset(Dataset):
    def __init__(
        self,
        data_file: str | Path,
        batch_size: int,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        with open(data_file, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        self.class_ids = sorted(set(meta["image_labels"]))
        self.batch_size = batch_size
        if transform is None:
            transforms = _load_transforms()
            transform = transforms.ToTensor()
        self.transform = transform
        self.target_transform = target_transform
        self.class_to_images: dict[int, list[str]] = {class_id: [] for class_id in self.class_ids}

        for image_name, label in zip(meta["image_names"], meta["image_labels"]):
            self.class_to_images[int(label)].append(image_name)

    def __len__(self) -> int:
        return len(self.class_ids)

    def __getitem__(self, index: int):
        class_id = self.class_ids[index]
        image_paths = self.class_to_images[class_id]
        replace = len(image_paths) < self.batch_size
        sample_indices = np.random.choice(len(image_paths), self.batch_size, replace=replace)

        images = []
        targets = []
        for image_index in sample_indices:
            image = Image.open(image_paths[image_index]).convert("RGB")
            images.append(self.transform(image))
            targets.append(class_id if self.target_transform is None else self.target_transform(class_id))

        return torch.stack(images), torch.tensor(targets, dtype=torch.long)


class SetDataManager:
    def __init__(self, image_size: int, n_way: int, k_shot: int, n_query: int, n_episode: int):
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = k_shot + n_query
        self.n_episode = n_episode

    def get_data_loader(self, data_file: str | Path, aug: bool = False) -> DataLoader:
        transform = build_episode_transform(self.image_size, aug=aug)
        dataset = SetDataset(data_file=data_file, batch_size=self.batch_size, transform=transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)


class MiniImagenetEpisodeDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        n_way: int,
        k_shot: int,
        k_query: int,
        episodes: int,
        image_size: int = 84,
        start_index: int = 0,
    ):
        self.root = Path(root)
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.episodes = episodes
        self.image_size = image_size
        self.start_index = start_index

        json_name = f"{split}.json"
        json_path = self.root / json_name
        if split == "base" and not json_path.exists():
            json_path = self.root / "train.json"

        with open(json_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        self.class_to_images: dict[int, list[str]] = {}
        for image_path, label in zip(meta["image_names"], meta["image_labels"]):
            label = int(label)
            self.class_to_images.setdefault(label, []).append(image_path)

        self.class_ids = sorted(self.class_to_images)
        self.transform = build_episode_transform(image_size, aug=False)

    def __len__(self) -> int:
        return self.episodes

    def __getitem__(self, index: int):
        del index
        selected_class_ids = np.random.choice(self.class_ids, self.n_way, replace=False)
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for class_index, class_id in enumerate(selected_class_ids, start=self.start_index):
            image_paths = self.class_to_images[int(class_id)]
            num_samples = self.k_shot + self.k_query
            replace = len(image_paths) < num_samples
            sample_indices = np.random.choice(len(image_paths), num_samples, replace=replace)
            for offset, sample_index in enumerate(sample_indices):
                image = Image.open(image_paths[sample_index]).convert("RGB")
                tensor = self.transform(image)
                if offset < self.k_shot:
                    support_images.append(tensor)
                    support_labels.append(class_index - self.start_index)
                else:
                    query_images.append(tensor)
                    query_labels.append(class_index - self.start_index)

        return (
            torch.stack(support_images),
            torch.tensor(support_labels, dtype=torch.long),
            torch.stack(query_images),
            torch.tensor(query_labels, dtype=torch.long),
        )


def build_synthetic_relation_loader(
    n_way: int = 5,
    k_shot: int = 1,
    k_query: int = 2,
    episodes: int = 8,
    image_size: int = 84,
    batch_size: int = 1,
) -> DataLoader:
    dataset = []
    for index in range(episodes):
        generator = torch.Generator().manual_seed(index)
        dataset.append(
            (
                torch.randn(n_way * k_shot, 3, image_size, image_size, generator=generator),
                torch.arange(n_way).repeat_interleave(k_shot),
                torch.randn(n_way * k_query, 3, image_size, image_size, generator=generator),
                torch.arange(n_way).repeat_interleave(k_query),
            )
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def build_synthetic_meta_loader(
    n_way: int = 5,
    k_shot: int = 1,
    n_query: int = 1,
    n_episode: int = 8,
    num_classes: int = 8,
    image_size: int = 84,
) -> DataLoader:
    num_classes = max(num_classes, n_way)
    samples_per_class = k_shot + n_query
    generator = torch.Generator().manual_seed(1234)
    samples = torch.randn(num_classes, samples_per_class, 3, image_size, image_size, generator=generator)
    dataset = [
        (samples[class_index], torch.full((samples_per_class,), class_index, dtype=torch.long))
        for class_index in range(num_classes)
    ]
    sampler = EpisodicBatchSampler(len(dataset), n_way, n_episode)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)
