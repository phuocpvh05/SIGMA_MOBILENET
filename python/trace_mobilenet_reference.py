#!/usr/bin/env python3
"""Generate/compare a BF16 layer-boundary reference for MobileNet RTL debug."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cnn"))

from mobilenet_onchip_model import extract_affine_ops, load_checkpoint
from pack_mobilenet_onchip import bf16_words, fold_batch_norm


def bf16_tensor(value):
    import torch

    array = value.detach().cpu().numpy().astype(np.float32, copy=True)
    words = bf16_words(array)
    return torch.from_numpy((words.astype(np.uint32) << 16).view(np.float32))


def read_words(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text(encoding="ascii").splitlines():
        token = line.strip().lower()
        if not token:
            continue
        values.append(0 if "x" in token else int(token, 16))
    return np.asarray(values, dtype=np.uint16)


def main() -> int:
    import torch
    import torch.nn.functional as functional

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-mem", type=Path, required=True)
    parser.add_argument("--stop-layer", type=int, required=True,
                        help="Next RTL layer; layers [0, stop-layer) are evaluated")
    parser.add_argument("--actual", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image_words = read_words(args.image_mem)[:784]
    image = (image_words.astype(np.uint32) << 16).view(np.float32)
    banks = {0: torch.from_numpy(image.reshape(1, 1, 28, 28).copy())}
    model, _ = load_checkpoint()
    ops = extract_affine_ops(model)
    if not 1 <= args.stop_layer <= len(ops):
        raise SystemExit("stop-layer must be in [1, 53]")

    with torch.inference_mode():
        for op in ops[:args.stop_layer]:
            source = bf16_tensor(banks[op.src_bank])
            weight, bias = fold_batch_norm(op.module, op.batch_norm)
            weight_t = bf16_tensor(torch.from_numpy(weight))
            bias_t = bf16_tensor(torch.from_numpy(bias))
            if op.kind == "linear":
                result = functional.linear(source.flatten(1), weight_t, bias_t)
            else:
                result = functional.conv2d(
                    source, weight_t, bias_t,
                    stride=op.module.stride, padding=op.module.padding,
                    dilation=op.module.dilation, groups=op.module.groups,
                )
            if op.activation == "relu6":
                result = torch.clamp(result, 0, 6)
            if op.residual:
                result = result + banks[op.skip_bank]
            banks[op.dst_bank] = bf16_tensor(result)

    last = ops[args.stop_layer - 1]
    tensor = banks[last.dst_bank][0]
    if tensor.ndim == 3:
        flat = tensor.permute(1, 2, 0).contiguous().numpy().reshape(-1)
    else:
        flat = tensor.numpy().reshape(-1)
    expected = bf16_words(flat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{int(word):04x}\n" for word in expected),
                           encoding="ascii")

    report = {
        "completed_layer": args.stop_layer - 1,
        "name": last.name,
        "dst_bank": last.dst_bank,
        "words": int(expected.size),
    }
    if args.actual:
        actual = read_words(args.actual)[:expected.size]
        if actual.size != expected.size:
            raise SystemExit(f"actual has {actual.size} words, expected {expected.size}")
        actual_fp = (actual.astype(np.uint32) << 16).view(np.float32)
        expected_fp = (expected.astype(np.uint32) << 16).view(np.float32)
        mismatch = actual != expected
        report.update({
            "bit_exact_words": int((~mismatch).sum()),
            "mismatch_words": int(mismatch.sum()),
            "max_abs_error": float(np.max(np.abs(actual_fp - expected_fp))),
            "mean_abs_error": float(np.mean(np.abs(actual_fp - expected_fp))),
            "actual_finite": bool(np.isfinite(actual_fp).all()),
        })
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
