#!/usr/bin/env python3
"""Verify BatchNorm folding, BF16 packing and board-profile accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CNN_DIR = ROOT / "cnn"
sys.path.insert(0, str(CNN_DIR))

from mobilenet_onchip_model import DEFAULT_CHECKPOINT, extract_affine_ops, load_checkpoint
from mnist_degradations import broken_strokes, poor_strokes
from pack_mobilenet_onchip import bf16_words, fold_batch_norm, packed_matrix


def bf16_tensor(value):
    import torch

    array = value.detach().cpu().numpy().astype(np.float32, copy=True)
    words = bf16_words(array)
    restored = (words.astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(restored).to(value.device)


def folded_forward(model, images, quantized=True, ops=None):
    import torch
    import torch.nn.functional as functional

    if ops is None:
        ops = extract_affine_ops(model)
    banks = {0: images}
    for op in ops:
        source = banks[op.src_bank]
        weight, bias = fold_batch_norm(op.module, op.batch_norm)
        weight_t = torch.from_numpy(weight).to(source.device)
        bias_t = torch.from_numpy(bias).to(source.device)
        if quantized:
            source = bf16_tensor(source)
            weight_t = bf16_tensor(weight_t)
            bias_t = bf16_tensor(bias_t)
        if op.kind == "linear":
            output = functional.linear(source.flatten(1), weight_t, bias_t)
        else:
            output = functional.conv2d(
                source,
                weight_t,
                bias_t,
                stride=op.module.stride,
                padding=op.module.padding,
                dilation=op.module.dilation,
                groups=op.module.groups,
            )
        if op.activation == "relu6":
            output = torch.clamp(output, 0, 6)
        if op.residual:
            output = output + banks[op.skip_bank]
        if quantized:
            output = bf16_tensor(output)
        banks[op.dst_bank] = output
    return banks[ops[-1].dst_bank]


def evaluate(model, ops, images, labels, batch, mode):
    correct = 0
    for begin in range(0, len(images), batch):
        logits = folded_forward(model, images[begin:begin + batch], quantized=True, ops=ops)
        correct += int((logits.argmax(1) == labels[begin:begin + batch]).sum())
    return correct / len(images)


def main():
    import torch

    parser = argparse.ArgumentParser(description="Verify MobileNet on-chip model")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "mnist.npz")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--weights", type=Path, default=CNN_DIR / "mobilenet_onchip_bf16.mem")
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    model.eval()
    ops = extract_affine_ops(model)
    expected_words = np.concatenate([
        bf16_words(packed_matrix(op).reshape(-1)) for op in ops
    ])
    actual_words = np.asarray([
        int(line.strip(), 16) for line in args.weights.read_text().splitlines() if line.strip()
    ], dtype=np.uint16)
    if not np.array_equal(expected_words, actual_words):
        raise SystemExit("Packed ROM mismatch")

    data = np.load(args.data)
    clean = data["x_test"][:args.samples].astype(np.float32) / 255.0
    conditions = {
        "clean": clean,
        "broken": broken_strokes(clean, 2026),
        "poor": poor_strokes(clean, 2027),
    }
    labels = torch.from_numpy(data["y_test"][:args.samples].astype(np.int64))
    scores = {}
    for mode, condition_images in conditions.items():
        tested = torch.from_numpy(condition_images[:, None])
        scores[mode] = evaluate(model, ops, tested, labels, args.batch, mode)
        print(f"{mode}_bf16_accuracy={scores[mode]:.4%}")

    images = torch.from_numpy(clean[:, None])
    with torch.inference_mode():
        baseline = model(images[:8])
        folded = folded_forward(model, images[:8], quantized=False, ops=ops)
    max_fold_delta = float((baseline - folded).abs().max())
    report = {
        "network": "MobileNetV2-0.25 MNIST",
        "layers": len(ops),
        "rom_words": int(len(actual_words)),
        "rom_exact": True,
        "batchnorm_fold_max_logit_delta": max_fold_delta,
        "samples_per_condition": args.samples,
        "bf16_accuracy": scores,
        "training_clean_accuracy": checkpoint.get("clean_accuracy") if isinstance(checkpoint, dict) else None,
    }
    report_path = CNN_DIR / "last_mobilenet_onchip_verification.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"batchnorm_fold_max_logit_delta={max_fold_delta:.8g}")
    print(f"rom_exact=PASS words={len(actual_words)}")
    print(f"report={report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
