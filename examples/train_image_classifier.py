import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

import mlplat

DEFAULTS = {
    "workers": 0,
    "batch_size": 64,
    "steps": 200,
    "image_size": 64,
    "num_classes": 10,
    "preprocess_cost": 6,
    "device": "auto",
    "threads": 0,
    "lr": 0.05,
    "seed": 0,
    "slowdown_ms": 0.0,
    "dataset_size": 50000,
}


class SyntheticImages(Dataset):
    def __init__(self, size, image_size, num_classes, preprocess_cost, seed):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.preprocess_cost = preprocess_cost
        self.seed = seed
        grid = np.linspace(0, 2 * np.pi, image_size, dtype=np.float32)
        self.patterns = np.stack([
            np.stack([np.sin((k + 1) * grid)[None, :] * np.cos((channel + 1) * grid)[:, None]
                      for channel in range(3)])
            for k in range(num_classes)
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        label = int(rng.integers(self.num_classes))
        image = self.patterns[label] + rng.normal(0, 0.8, self.patterns[label].shape).astype(np.float32)
        return torch.from_numpy(self.augment(image, rng)), label

    def augment(self, image, rng):
        if rng.random() < 0.5:
            image = image[:, :, ::-1]
        shift = rng.integers(-4, 5, size=2)
        image = np.roll(image, tuple(shift), axis=(1, 2))
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        for _ in range(self.preprocess_cost):
            image = (kernel[0] * np.roll(image, 1, axis=1) + kernel[1] * image + kernel[2] * np.roll(image, -1, axis=1))
            image = (kernel[0] * np.roll(image, 1, axis=2) + kernel[1] * image + kernel[2] * np.roll(image, -1, axis=2))
            image = image + 0.01 * np.tanh(image)
        mean = image.mean(axis=(1, 2), keepdims=True)
        std = image.std(axis=(1, 2), keepdims=True) + 1e-5
        return np.ascontiguousarray((image - mean) / std, dtype=np.float32)


class SmallCnn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.head = nn.Linear(128, num_classes)

    def forward(self, images):
        x = F.max_pool2d(F.relu(self.conv1(images)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        return self.head(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


def load_settings():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("MLPLAT_CONFIG_PATH"))
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=None)
    args = parser.parse_args()
    settings = dict(DEFAULTS)
    if args.config:
        with open(args.config) as handle:
            settings.update((yaml.safe_load(handle) or {}).get("train", {}))
    settings.update({key: value for key, value in vars(args).items() if value is not None and key != "config"})
    return settings


def resolve_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    settings = load_settings()
    torch.manual_seed(settings["seed"])
    if settings["threads"]:
        torch.set_num_threads(settings["threads"])
    device = resolve_device(settings["device"])
    print(f"settings {settings} device {device} torch threads {torch.get_num_threads()}", flush=True)

    dataset = SyntheticImages(settings["dataset_size"], settings["image_size"], settings["num_classes"],
                              settings["preprocess_cost"], settings["seed"])
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True, num_workers=settings["workers"],
                        pin_memory=device.type == "cuda", persistent_workers=settings["workers"] > 0,
                        drop_last=True)
    model = SmallCnn(settings["num_classes"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=settings["lr"], momentum=0.9)

    step = 0
    started = time.perf_counter()
    while step < settings["steps"]:
        fetch_started = time.perf_counter()
        for images, labels in loader:
            data_time = time.perf_counter() - fetch_started
            compute_started = time.perf_counter()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if settings["slowdown_ms"]:
                time.sleep(settings["slowdown_ms"] / 1000.0)
            synchronize(device)
            compute_time = time.perf_counter() - compute_started
            accuracy = (logits.argmax(dim=1) == labels).float().mean()
            mlplat.log(step=step, loss=loss, accuracy=accuracy, data_time=data_time, compute_time=compute_time)
            step += 1
            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.3f} acc {accuracy.item():.3f} "
                      f"{step / (time.perf_counter() - started):.1f} steps/s", flush=True)
            if step >= settings["steps"]:
                break
            fetch_started = time.perf_counter()
    print(f"done: {step} steps in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
