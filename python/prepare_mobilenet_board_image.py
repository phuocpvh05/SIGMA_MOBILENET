#!/usr/bin/env python3
"""Convert one handwritten image into the BF16 payload used by board JTAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CNN_DIR = ROOT / "rtl" / "model"
PYTHON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CNN_DIR))

from image_utils import load_handwritten_image
from mobilenet_onchip_model import DEFAULT_CHECKPOINT, load_checkpoint


def bf16_words(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32).copy()
    bits += np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (bits >> 16).astype(np.uint16)


def main():
    import torch

    parser = argparse.ArgumentParser(description="Prepare MobileNet board image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=CNN_DIR / "mobilenet_board_image.mem")
    args = parser.parse_args()

    image = load_handwritten_image(args.image)
    model, checkpoint = load_checkpoint(args.checkpoint)
    with torch.inference_mode():
        tensor = torch.from_numpy(image[None, None].astype(np.float32))
        logits = model(tensor)[0]
        prediction = int(logits.argmax())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    words = bf16_words(image.reshape(-1))
    args.output.write_text("".join(f"{int(word):04x}\n" for word in words), encoding="ascii")
    report = {
        "network": "MobileNetV2-0.25 MNIST",
        "image": str(args.image.resolve()),
        "image_mem": str(args.output.resolve()),
        "software_prediction": prediction,
        "training_clean_accuracy": checkpoint.get("clean_accuracy") if isinstance(checkpoint, dict) else None,
    }
    report_path = CNN_DIR / "last_mobilenet_board_image.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"image_mem={args.output.resolve()}")
    print(f"software_prediction={prediction}")
    print(f"report={report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
