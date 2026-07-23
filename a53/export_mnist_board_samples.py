#!/usr/bin/env python3
"""Export visible PNGs from all BF16 payloads embedded in the board ELF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GENERATED = ROOT / "software" / "datasets"
IMAGE_BINS = {
    "clean": GENERATED / "mobilenet_mnist10k_bf16.bin",
    "broken": GENERATED / "mobilenet_mnist10k_broken_bf16.bin",
    "poor": GENERATED / "mobilenet_mnist10k_poor_bf16.bin",
}
LABEL_BIN = GENERATED / "mobilenet_mnist10k_labels.bin"
MANIFEST = GENERATED / "mobilenet_mnist10k_manifest.json"
OUTPUT = ROOT / "docs" / "mobilenet_mnist10k_conditions"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    label_hash = sha256(LABEL_BIN)
    if label_hash != manifest["label_sha256"]:
        raise RuntimeError("Label payload hash does not match its manifest")

    labels = np.fromfile(LABEL_BIN, dtype=np.uint8)
    if labels.size != 10000:
        raise RuntimeError("Unexpected board payload dimensions")

    # Select one sample of each class from ten widely separated dataset regions.
    chosen: list[int] = []
    for digit in range(10):
        start = digit * 1000 + 500
        matches = np.flatnonzero(labels[start:] == digit)
        if not matches.size:
            raise RuntimeError(f"No sample found for label {digit}")
        chosen.append(start + int(matches[0]))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    tiles = []
    condition_hashes = {}
    for row, (condition, image_bin) in enumerate(IMAGE_BINS.items()):
        image_hash = sha256(image_bin)
        expected_hash = manifest["conditions"][condition]["image_sha256"]
        if image_hash != expected_hash:
            raise RuntimeError(
                f"{condition} BF16 payload hash does not match its manifest"
            )
        condition_hashes[condition] = image_hash
        words = np.fromfile(image_bin, dtype="<u2")
        if words.size != 10000 * 28 * 28:
            raise RuntimeError(f"Unexpected {condition} payload dimensions")

        # Decode the exact BF16 values consumed by RTL/Cortex-A53.
        pixels_f32 = (words.astype(np.uint32) << 16).view(np.float32)
        pixels_u8 = np.rint(np.clip(pixels_f32, 0.0, 1.0) * 255.0)
        images = pixels_u8.astype(np.uint8).reshape(10000, 28, 28)
        for column, index in enumerate(chosen):
            label = int(labels[index])
            filename = (
                f"{condition}_idx{index:05d}_label{label}.png"
            )
            path = OUTPUT / filename
            image = Image.fromarray(images[index], mode="L")
            image.save(path)
            records.append({
                "condition": condition,
                "index": index,
                "label": label,
                "png": str(path),
            })

            tile = Image.new("L", (112, 132), 24)
            enlarged = image.resize((112, 112), Image.Resampling.NEAREST)
            tile.paste(enlarged, (0, 0))
            caption = f"{condition} {label}" if column == 0 else f"{label}"
            ImageDraw.Draw(tile).text((4, 115), caption, fill=255)
            tiles.append((row, column, tile))

    sheet = Image.new("L", (10 * 112, 3 * 132), 0)
    for row, column, tile in tiles:
        sheet.paste(tile, (column * 112, row * 132))
    sheet_path = OUTPUT / "mobilenet_clean_broken_poor_contact_sheet.png"
    sheet.save(sheet_path)

    proof = {
        "sources_used_by_board_assembly": {
            name: str(path) for name, path in IMAGE_BINS.items()
        },
        "labels_used_by_board_assembly": str(LABEL_BIN),
        "condition_sha256": condition_hashes,
        "label_sha256": label_hash,
        "payload_images": 10000,
        "shape": [10000, 28, 28],
        "samples": records,
        "contact_sheet": str(sheet_path),
    }
    proof_path = OUTPUT / "mobilenet_conditions_proof.json"
    proof_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(f"Exported {len(records)} PNGs from the exact board BF16 payloads")
    print(f"Contact sheet: {sheet_path}")
    print(f"Proof: {proof_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
