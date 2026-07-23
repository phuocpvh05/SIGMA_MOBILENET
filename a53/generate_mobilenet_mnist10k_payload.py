#!/usr/bin/env python3
"""Generate the exact BF16 MNIST test payload used by the board benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CNN_DIR = ROOT / "software" / "python"
if str(CNN_DIR) not in sys.path:
    sys.path.insert(0, str(CNN_DIR))

from mnist_degradations import broken_strokes, poor_strokes


DATASET = ROOT / "software" / "datasets" / "mnist.npz"
OUTPUT_DIR = ROOT / "software" / "datasets"
IMAGE_OUTPUTS = {
    "clean": OUTPUT_DIR / "mobilenet_mnist10k_bf16.bin",
    "broken": OUTPUT_DIR / "mobilenet_mnist10k_broken_bf16.bin",
    "poor": OUTPUT_DIR / "mobilenet_mnist10k_poor_bf16.bin",
}
LABEL_OUTPUT = OUTPUT_DIR / "mobilenet_mnist10k_labels.bin"
MANIFEST_OUTPUT = OUTPUT_DIR / "mobilenet_mnist10k_manifest.json"
DEGRADATION_SEED = 2026


def bf16_words(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32).copy()
    bits += np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (bits >> 16).astype("<u2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    payload = np.load(DATASET)
    images_u8 = np.asarray(payload["x_test"], dtype=np.uint8)
    labels = np.asarray(payload["y_test"], dtype=np.uint8)
    if images_u8.shape != (10000, 28, 28):
        raise ValueError(f"Unexpected MNIST test shape: {images_u8.shape}")
    if labels.shape != (10000,):
        raise ValueError(f"Unexpected MNIST label shape: {labels.shape}")
    if np.any(labels > 9):
        raise ValueError("MNIST labels must be in the range 0..9")

    clean = images_u8.astype(np.float32) / np.float32(255.0)
    conditions = {
        "clean": clean,
        "broken": broken_strokes(clean, DEGRADATION_SEED),
        "poor": poor_strokes(clean, DEGRADATION_SEED + 1),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    condition_manifest = {}
    for name, images in conditions.items():
        images_bf16 = bf16_words(images)
        image_output = IMAGE_OUTPUTS[name]
        images_bf16.tofile(image_output)
        condition_manifest[name] = {
            "file": str(image_output.resolve()),
            "image_words": int(images_bf16.size),
            "image_bytes": image_output.stat().st_size,
            "image_sha256": sha256(image_output),
        }
    labels.tofile(LABEL_OUTPUT)

    expected_image_bytes = 10000 * 28 * 28 * 2
    for name, image_output in IMAGE_OUTPUTS.items():
        if image_output.stat().st_size != expected_image_bytes:
            raise RuntimeError(f"Generated {name} BF16 payload has the wrong size")
    if LABEL_OUTPUT.stat().st_size != 10000:
        raise RuntimeError("Generated label payload has the wrong size")

    manifest = {
        "source": str(DATASET.resolve()),
        "split": "x_test/y_test",
        "images": 10000,
        "shape": [10000, 28, 28],
        "input_format": "BF16 little-endian, pixel/255.0, round-to-nearest-even",
        "conditions": condition_manifest,
        "degradation_seed": DEGRADATION_SEED,
        "definitions": {
            "clean": "original MNIST x_test pixels scaled by 1/255",
            "broken": "remove two short pieces centred on real ink pixels",
            "poor": (
                "translate by up to 2 pixels, reduce contrast, add Gaussian "
                "noise sigma=0.10, and remove one local 3x3 ink region"
            ),
        },
        "label_bytes": LABEL_OUTPUT.stat().st_size,
        "label_sha256": sha256(LABEL_OUTPUT),
        "label_histogram": np.bincount(labels, minlength=10).tolist(),
        "first_labels": labels[:10].tolist(),
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, image_output in IMAGE_OUTPUTS.items():
        print(f"images_{name}={image_output}")
    print(f"labels={LABEL_OUTPUT}")
    print(f"manifest={MANIFEST_OUTPUT}")
    print(f"image_bytes_per_condition={expected_image_bytes}")
    print(f"first_labels={manifest['first_labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
