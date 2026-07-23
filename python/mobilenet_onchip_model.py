"""Board-sized MobileNetV2 profile used by the autonomous SIGMA controller.

This is the Torchvision MobileNetV2 topology with ``width_mult=0.25`` and a
single MNIST input channel.  Keeping the canonical inverted-residual blocks,
depthwise convolutions, pointwise convolutions, residual adds, ReLU6 and global
pool makes it a real MobileNetV2 workload, while the reduced width lets the
folded BF16 weights fit in the XCZU5EV programmable-logic memories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PYTHON_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PYTHON_DIR / "mobilenetv2_025_mnist.pt"
WIDTH_MULT = 0.25
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1
NUM_CLASSES = 10


@dataclass
class AffineOp:
    name: str
    kind: str
    module: Any
    batch_norm: Any | None
    activation: str
    residual: bool
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    src_bank: int
    dst_bank: int
    skip_bank: int


def build_model():
    """Create the exact train/pack/reference model."""
    import torch.nn as nn
    from torchvision.models import mobilenet_v2

    model = mobilenet_v2(
        weights=None,
        width_mult=WIDTH_MULT,
        num_classes=NUM_CLASSES,
        dropout=0.0,
    )
    first = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        IMAGE_CHANNELS,
        first.out_channels,
        kernel_size=first.kernel_size,
        stride=first.stride,
        padding=first.padding,
        dilation=first.dilation,
        groups=1,
        bias=False,
    )
    return model


def load_checkpoint(path: Path = DEFAULT_CHECKPOINT, device: str = "cpu"):
    import torch

    model = build_model()
    payload = torch.load(path, map_location=device, weights_only=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, payload


def _conv_bn_activation(container):
    """Return Conv2d, BatchNorm2d and activation kind from a small sequence."""
    import torch.nn as nn

    conv = next((item for item in container if isinstance(item, nn.Conv2d)), None)
    bn = next((item for item in container if isinstance(item, nn.BatchNorm2d)), None)
    relu6 = any(isinstance(item, nn.ReLU6) for item in container)
    if conv is None:
        raise ValueError(f"No Conv2d in {container}")
    return conv, bn, "relu6" if relu6 else "none"


def extract_affine_ops(model) -> list[AffineOp]:
    """Flatten MobileNetV2 into the order consumed by the RTL micro-controller.

    Three activation banks are allocated statically.  A residual source bank is
    never overwritten until the projection result has been added to it.
    """
    import torch
    import torch.nn as nn

    shapes: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {}
    handles = []

    def remember(name):
        def hook(_module, inputs, output):
            inp = inputs[0]
            shapes[name] = (tuple(inp.shape[1:]), tuple(output.shape[1:]))
        return hook

    named_conv = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            named_conv.append((name, module))
            handles.append(module.register_forward_hook(remember(name)))
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
    model.train(was_training)
    for handle in handles:
        handle.remove()

    ops: list[AffineOp] = []
    current_bank = 0

    def add(name, conv, bn, activation, residual, src, dst, skip):
        in_shape, out_shape = shapes[name]
        if len(in_shape) == 1:
            in_shape = (in_shape[0], 1, 1)
            out_shape = (out_shape[0], 1, 1)
            kind = "linear"
        elif conv.groups == conv.in_channels and conv.groups > 1:
            kind = "depthwise"
        else:
            kind = "conv"
        ops.append(AffineOp(
            name=name,
            kind=kind,
            module=conv,
            batch_norm=bn,
            activation=activation,
            residual=residual,
            input_shape=in_shape,
            output_shape=out_shape,
            src_bank=src,
            dst_bank=dst,
            skip_bank=skip,
        ))

    # Initial Conv2dNormActivation.
    conv, bn, activation = _conv_bn_activation(model.features[0])
    add("features.0.0", conv, bn, activation, False, current_bank, 1, 0)
    current_bank = 1

    for feature_index in range(1, 18):
        block = model.features[feature_index]
        block_source = current_bank
        sequences = []
        index = 0
        while index < len(block.conv):
            child = block.conv[index]
            if child.__class__.__name__ == "Conv2dNormActivation":
                conv, bn, activation = _conv_bn_activation(child)
                sequences.append((f"features.{feature_index}.conv.{index}.0", conv, bn, activation))
                index += 1
            elif isinstance(child, nn.Conv2d):
                bn = block.conv[index + 1]
                sequences.append((f"features.{feature_index}.conv.{index}", child, bn, "none"))
                index += 2
            else:
                index += 1

        used = {block_source}
        src = block_source
        for sequence_index, (name, conv, bn, activation) in enumerate(sequences):
            is_projection = sequence_index + 1 == len(sequences)
            if block.use_res_connect:
                # Intermediate results use the two banks not holding the skip.
                choices = [bank for bank in range(3) if bank != block_source and bank != src]
                if not choices:
                    choices = [bank for bank in range(3) if bank != block_source]
                dst = choices[0]
            else:
                dst = next(bank for bank in range(3) if bank != src)
            add(
                name, conv, bn, activation,
                bool(block.use_res_connect and is_projection),
                src, dst, block_source,
            )
            src = dst
            used.add(dst)
        current_bank = src

    conv, bn, activation = _conv_bn_activation(model.features[18])
    final_bank = next(bank for bank in range(3) if bank != current_bank)
    add("features.18.0", conv, bn, activation, False, current_bank, final_bank, 0)
    current_bank = final_bank

    linear = model.classifier[1]
    output_bank = next(bank for bank in range(3) if bank != current_bank)
    add("classifier.1", linear, None, "none", False, current_bank, output_bank, 0)
    return ops


def model_summary(model) -> dict[str, int | float]:
    ops = extract_affine_ops(model)
    return {
        "width_mult": WIDTH_MULT,
        "affine_layers": len(ops),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "max_activation_words": max(
            op.output_shape[0] * op.output_shape[1] * op.output_shape[2]
            for op in ops
        ),
    }
