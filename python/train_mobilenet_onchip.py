#!/usr/bin/env python3
"""Train MobileNetV2-0.25 on the project's MNIST dataset for FPGA packing."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CNN_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "software" / "datasets"
sys.path.insert(0, str(CNN_DIR))

from mobilenet_onchip_model import DEFAULT_CHECKPOINT, build_model, model_summary


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path, train_samples, test_samples):
    import torch

    data = np.load(path)
    train_x = torch.from_numpy(data["x_train"][:train_samples].astype(np.float32)[:, None] / 255.0)
    train_y = torch.from_numpy(data["y_train"][:train_samples].astype(np.int64))
    test_x = torch.from_numpy(data["x_test"][:test_samples].astype(np.float32)[:, None] / 255.0)
    test_y = torch.from_numpy(data["y_test"][:test_samples].astype(np.int64))
    return train_x, train_y, test_x, test_y


def augment(images):
    import torch

    result = images.clone()
    # Integer translations and light noise preserve the handwritten-digit task
    # while producing the broken/poor-stroke robustness requested for reports.
    for index in range(result.shape[0]):
        if torch.rand(()) < 0.45:
            dy = int(torch.randint(-2, 3, ()).item())
            dx = int(torch.randint(-2, 3, ()).item())
            shifted = torch.roll(result[index], (dy, dx), dims=(1, 2))
            if dy > 0:
                shifted[:, :dy, :] = 0
            elif dy < 0:
                shifted[:, dy:, :] = 0
            if dx > 0:
                shifted[:, :, :dx] = 0
            elif dx < 0:
                shifted[:, :, dx:] = 0
            result[index] = shifted
        if torch.rand(()) < 0.25:
            result[index].add_(torch.randn_like(result[index]) * 0.06).clamp_(0, 1)
    return result


def accuracy(model, images, labels, batch, device):
    import torch

    model.eval()
    correct = 0
    with torch.inference_mode():
        for begin in range(0, len(images), batch):
            x = images[begin:begin + batch].to(device)
            y = labels[begin:begin + batch].to(device)
            correct += int((model(x).argmax(1) == y).sum())
    return correct / len(images)


def main():
    import torch

    parser = argparse.ArgumentParser(description="Train board-sized MobileNetV2 MNIST")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "mnist.npz")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-samples", type=int, default=60000)
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    train_x, train_y, test_x, test_y = load_data(
        args.data, args.train_samples, args.test_samples
    )
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_accuracy = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_x))
        running_loss = 0.0
        for begin in range(0, len(order), args.batch):
            indices = order[begin:begin + args.batch]
            x = augment(train_x[indices]).to(device)
            y = train_y[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(indices)
        scheduler.step()
        clean_accuracy = accuracy(model, test_x, test_y, args.batch, device)
        print(
            f"epoch={epoch + 1:02d}/{args.epochs} "
            f"loss={running_loss / len(train_x):.5f} clean_acc={clean_accuracy:.4%}"
        )
        if clean_accuracy >= best_accuracy:
            best_accuracy = clean_accuracy
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "clean_accuracy": clean_accuracy,
                "seed": args.seed,
                "summary": model_summary(model),
            }, args.output)

    report = {
        "checkpoint": str(args.output.resolve()),
        "best_clean_accuracy": best_accuracy,
        "device": device,
        "epochs": args.epochs,
        **model_summary(model),
    }
    report_path = CNN_DIR / "last_mobilenet_onchip_training.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"checkpoint={args.output.resolve()}")
    print(f"report={report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
