from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def cifar_loaders(name: str = "cifar100", root: str = "data", batch_size: int = 256, num_workers: int = 4):
    name = name.lower()
    mean_std = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    }
    if name not in mean_std:
        raise ValueError("name must be cifar10 or cifar100")
    mean, std = mean_std[name]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    train = cls(root, train=True, transform=train_tf, download=True)
    test = cls(root, train=False, transform=test_tf, download=True)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        10 if name == "cifar10" else 100,
    )


def imagenet_loaders(root: str, batch_size: int = 64, num_workers: int = 8, image_size: int = 224):
    root = Path(root)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = datasets.ImageFolder(root / "train", transform=train_tf)
    val = datasets.ImageFolder(root / "val", transform=val_tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        len(train.classes),
    )
