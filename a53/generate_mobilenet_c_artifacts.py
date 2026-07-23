#!/usr/bin/env python3
"""Generate compact C descriptors and optional embedded BF16 payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
CNN_DIR = ROOT / "rtl" / "model"


def read_mem(path: Path) -> list[int]:
    return [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]


def emit_words(name: str, words: list[int]) -> list[str]:
    lines = [f"static const uint16_t {name}[{len(words)}] = {{"]
    for start in range(0, len(words), 12):
        chunk = ", ".join(f"0x{word:04x}u" for word in words[start : start + 12])
        lines.append(f"    {chunk},")
    lines.append("};")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=CNN_DIR / "mobilenet_onchip_manifest.json")
    parser.add_argument("--weights", type=Path, default=CNN_DIR / "mobilenet_onchip_bf16.mem")
    parser.add_argument("--image", type=Path, default=CNN_DIR / "mobilenet_board_image.mem")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "generated")
    parser.add_argument("--embed", action="store_true", help="also generate the large bare-metal payload header")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    layers = manifest["layers"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    descriptor = [
        "// Generated from cnn/mobilenet_onchip_manifest.json; do not edit.",
        "#ifndef SIGMA_MOBILENET_LAYERS_GENERATED_H",
        "#define SIGMA_MOBILENET_LAYERS_GENERATED_H",
        "#include <stdint.h>",
        f"#define SIGMA_MOBILE_LAYER_COUNT {len(layers)}u",
        f"#define SIGMA_MOBILE_WEIGHT_WORDS {manifest['weight_words']}u",
        f"#define SIGMA_MOBILE_BANK_WORDS {manifest['activation_bank_words']}u",
        "#define SIGMA_MOBILE_IMAGE_WORDS 784u",
        "typedef struct {",
        "    uint32_t weight_offset;",
        "    uint16_t in_c, out_c;",
        "    uint8_t in_h, in_w, out_h, out_w;",
        "    uint8_t kernel, stride, padding;",
        "    uint8_t kind, src_bank, dst_bank, skip_bank;",
        "    uint8_t relu6, residual;",
        "} sigma_mobilenet_layer_t;",
        f"static const sigma_mobilenet_layer_t sigma_mobilenet_layers[{len(layers)}] = {{",
    ]
    for layer in layers:
        descriptor.append(
            "    {"
            f"{layer['weight_offset']}u, "
            f"{layer['input'][0]}u, {layer['output'][0]}u, "
            f"{layer['input'][1]}u, {layer['input'][2]}u, "
            f"{layer['output'][1]}u, {layer['output'][2]}u, "
            f"{layer['kernel']}u, {layer['stride']}u, {layer['padding']}u, "
            f"{layer['kind_code']}u, {layer['src_bank']}u, {layer['dst_bank']}u, "
            f"{layer['skip_bank']}u, {int(layer['relu6'])}u, {int(layer['residual'])}u"
            f"}}, // {layer['index']}: {layer['name']}"
        )
    descriptor.extend(["};", "#endif"])
    layer_header = args.output_dir / "mobilenet_layers_generated.h"
    layer_header.write_text("\n".join(descriptor) + "\n", encoding="ascii")

    if args.embed:
        weights = read_mem(args.weights)
        image = read_mem(args.image)
        if len(weights) != manifest["weight_words"]:
            raise SystemExit(f"weight count mismatch: {len(weights)}")
        if len(image) != 784:
            raise SystemExit(f"image count mismatch: {len(image)}")
        payload = [
            "// Generated BF16 data for the standalone Cortex-A53 benchmark.",
            "#ifndef SIGMA_MOBILENET_PAYLOAD_GENERATED_H",
            "#define SIGMA_MOBILENET_PAYLOAD_GENERATED_H",
            "#include <stdint.h>",
        ]
        payload.extend(emit_words("sigma_mobilenet_weights", weights))
        payload.extend(emit_words("sigma_mobilenet_image", image))
        payload.append("#endif")
        (args.output_dir / "mobilenet_payload_generated.h").write_text(
            "\n".join(payload) + "\n", encoding="ascii"
        )

    print(f"layers={layer_header}")
    if args.embed:
        print(f"payload={args.output_dir / 'mobilenet_payload_generated.h'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
