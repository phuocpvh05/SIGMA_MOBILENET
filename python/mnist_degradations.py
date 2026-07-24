"""Deterministic MNIST degradation functions shared by all benchmarks."""

from __future__ import annotations

import numpy as np


def broken_strokes(images: np.ndarray, seed: int) -> np.ndarray:
    """Remove two short pieces centred on real ink pixels."""
    rng = np.random.default_rng(seed)
    output = images.copy()
    for image in output:
        ink = np.argwhere(image > 0.30)
        if not len(ink):
            continue
        for _ in range(2):
            cy, cx = ink[int(rng.integers(len(ink)))]
            if rng.random() < 0.5:
                image[max(0, cy - 1):cy + 2, max(0, cx - 2):cx + 3] = 0
            else:
                image[max(0, cy - 2):cy + 3, max(0, cx - 1):cx + 2] = 0
    return output


def poor_strokes(images: np.ndarray, seed: int) -> np.ndarray:
    """Apply small translation, contrast loss, noise and one local gap."""
    rng = np.random.default_rng(seed)
    output = np.empty_like(images)
    for index, source in enumerate(images):
        dy, dx = rng.integers(-2, 3, size=2)
        image = np.roll(source, (int(dy), int(dx)), axis=(0, 1))
        if dy > 0:
            image[:dy] = 0
        elif dy < 0:
            image[dy:] = 0
        if dx > 0:
            image[:, :dx] = 0
        elif dx < 0:
            image[:, dx:] = 0
        image = np.clip(
            image * rng.uniform(0.65, 1.0)
            + rng.normal(0, 0.10, image.shape),
            0,
            1,
        )
        ink = np.argwhere(image > 0.35)
        if len(ink):
            cy, cx = ink[int(rng.integers(len(ink)))]
            image[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = 0
        output[index] = image
    return output
