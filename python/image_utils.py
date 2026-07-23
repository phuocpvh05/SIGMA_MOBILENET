"""Image loading and MNIST-style handwritten preprocessing."""

import struct
import zlib
from pathlib import Path

import numpy as np


def _load_pillow(path):
    """Load common compressed image formats and composite transparency on white."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ValueError(
            "JPEG/WEBP/TIFF input requires Pillow; install it with 'python -m pip install pillow'"
        ) from exc

    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        pixels = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb = pixels[:, :, :3]
    alpha = pixels[:, :, 3]
    # Match the PNG loader: transparent pixels represent white paper.
    rgb = rgb * alpha[:, :, None] + (1.0 - alpha[:, :, None])
    return (
        rgb[:, :, 0] * 0.299
        + rgb[:, :, 1] * 0.587
        + rgb[:, :, 2] * 0.114
    ).astype(np.float32)


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _load_png(path):
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature")
    position = 8
    compressed = bytearray()
    width = height = color_type = bit_depth = interlace = None
    while position < len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if bit_depth != 8 or channels is None or interlace != 0:
        raise ValueError("PNG must be non-interlaced 8-bit grayscale/RGB/RGBA")
    raw = zlib.decompress(bytes(compressed))
    stride = width * channels
    rows = np.zeros((height, stride), dtype=np.uint8)
    offset = 0
    previous = np.zeros(stride, dtype=np.uint8)
    for row in range(height):
        filter_type = raw[offset]
        offset += 1
        scan = np.frombuffer(raw[offset:offset + stride], dtype=np.uint8).astype(np.int16)
        offset += stride
        result = np.zeros(stride, dtype=np.int16)
        for index in range(stride):
            left = result[index - channels] if index >= channels else 0
            up = int(previous[index])
            upper_left = int(previous[index - channels]) if index >= channels else 0
            if filter_type == 0:
                value = scan[index]
            elif filter_type == 1:
                value = scan[index] + left
            elif filter_type == 2:
                value = scan[index] + up
            elif filter_type == 3:
                value = scan[index] + ((left + up) // 2)
            elif filter_type == 4:
                value = scan[index] + _paeth(left, up, upper_left)
            else:
                raise ValueError(f"Unsupported PNG filter {filter_type}")
            result[index] = value & 0xFF
        rows[row] = result.astype(np.uint8)
        previous = rows[row]
    pixels = rows.reshape(height, width, channels).astype(np.float32) / 255.0
    if color_type == 0:
        return pixels[:, :, 0]
    if color_type == 4:
        gray, alpha = pixels[:, :, 0], pixels[:, :, 1]
        return gray * alpha + (1 - alpha)
    rgb = pixels[:, :, :3]
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    if channels == 4:
        alpha = pixels[:, :, 3]
        gray = gray * alpha + (1 - alpha)
    return gray


def _load_bmp(path):
    data = Path(path).read_bytes()
    if data[:2] != b"BM":
        raise ValueError("Invalid BMP signature")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    header_size = struct.unpack_from("<I", data, 14)[0]
    if header_size < 40:
        raise ValueError("Unsupported BMP header")
    width, signed_height = struct.unpack_from("<ii", data, 18)
    planes, bits = struct.unpack_from("<HH", data, 26)
    compression = struct.unpack_from("<I", data, 30)[0]
    if planes != 1 or bits not in (24, 32) or compression != 0:
        raise ValueError("BMP must be uncompressed 24-bit or 32-bit")
    height = abs(signed_height)
    bytes_per_pixel = bits // 8
    row_bytes = ((width * bytes_per_pixel + 3) // 4) * 4
    image = np.zeros((height, width), dtype=np.float32)
    for output_row in range(height):
        source_row = output_row if signed_height < 0 else height - 1 - output_row
        begin = pixel_offset + source_row * row_bytes
        row = np.frombuffer(data[begin:begin + width * bytes_per_pixel], dtype=np.uint8)
        row = row.reshape(width, bytes_per_pixel)
        image[output_row] = (
            row[:, 2] * 0.299 + row[:, 1] * 0.587 + row[:, 0] * 0.114
        ) / 255.0
    return image


def _load_pgm(path):
    data = Path(path).read_bytes()
    tokens = []
    position = 0
    while len(tokens) < 4:
        while position < len(data) and chr(data[position]).isspace():
            position += 1
        if position < len(data) and data[position] == ord("#"):
            while position < len(data) and data[position] not in (10, 13):
                position += 1
            continue
        begin = position
        while position < len(data) and not chr(data[position]).isspace():
            position += 1
        tokens.append(data[begin:position].decode("ascii"))
    magic, width, height, maximum = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    while position < len(data) and chr(data[position]).isspace():
        position += 1
    if magic == "P5":
        pixels = np.frombuffer(data[position:position + width * height], dtype=np.uint8)
    elif magic == "P2":
        pixels = np.asarray([int(value) for value in data[position:].split()], dtype=np.float32)
    else:
        raise ValueError("PGM must use P2 or P5 format")
    return pixels.reshape(height, width).astype(np.float32) / maximum


def _resize_bilinear(image, new_height, new_width):
    if image.shape == (new_height, new_width):
        return image.astype(np.float32)
    ys = np.linspace(0, image.shape[0] - 1, new_height)
    xs = np.linspace(0, image.shape[1] - 1, new_width)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = image[y0[:, None], x0[None, :]] * (1 - wx) + image[y0[:, None], x1[None, :]] * wx
    bottom = image[y1[:, None], x0[None, :]] * (1 - wx) + image[y1[:, None], x1[None, :]] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def _resize_preserve_thin_ink(image, new_height, new_width):
    """Downsample ink using source-area maxima so narrow strokes stay connected."""
    result = np.zeros((new_height, new_width), dtype=np.float32)
    for y in range(new_height):
        y0 = int(np.floor(y * image.shape[0] / new_height))
        y1 = max(y0 + 1, int(np.ceil((y + 1) * image.shape[0] / new_height)))
        for x in range(new_width):
            x0 = int(np.floor(x * image.shape[1] / new_width))
            x1 = max(x0 + 1, int(np.ceil((x + 1) * image.shape[1] / new_width)))
            result[y, x] = image[y0:y1, x0:x1].max(initial=0)
    return result


def preprocess_handwritten(image):
    image = np.clip(np.asarray(image, dtype=np.float32), 0, 1)
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    # Convert either black-on-white paper or white-on-black canvas to MNIST ink.
    ink = 1.0 - image if np.median(border) > 0.5 else image
    ink[ink < max(0.08, float(ink.max()) * 0.08)] = 0
    coordinates = np.argwhere(ink > 0)
    if not len(coordinates):
        raise ValueError("No handwriting found in the image")
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    cropped = ink[y0:y1, x0:x1]
    scale = min(20 / cropped.shape[0], 20 / cropped.shape[1])
    new_height = max(1, int(round(cropped.shape[0] * scale)))
    new_width = max(1, int(round(cropped.shape[1] * scale)))
    resized = _resize_bilinear(cropped, new_height, new_width)
    if float(resized.sum()) < 15.0 and scale < 1.0:
        resized = _resize_preserve_thin_ink(cropped, new_height, new_width)
    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_height) // 2
    left = (28 - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    # Shift the center of mass toward the MNIST image center.
    mass = canvas.sum()
    yy, xx = np.indices(canvas.shape)
    cy = int(round(float((canvas * yy).sum() / mass)))
    cx = int(round(float((canvas * xx).sum() / mass)))
    canvas = np.roll(canvas, 14 - cy, axis=0)
    canvas = np.roll(canvas, 14 - cx, axis=1)
    return np.clip(canvas, 0, 1)


def load_handwritten_image(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".png":
        image = _load_png(path)
    elif suffix == ".bmp":
        image = _load_bmp(path)
    elif suffix in (".jpg", ".jpeg", ".webp", ".tif", ".tiff"):
        image = _load_pillow(path)
    elif suffix in (".pgm", ".pnm"):
        image = _load_pgm(path)
    elif suffix == ".npy":
        image = np.load(path)
    else:
        raise ValueError(
            "Supported custom image formats: PNG, JPG, JPEG, WEBP, TIFF, BMP, PGM, NPY"
        )
    if image.ndim != 2:
        raise ValueError("Custom image must be grayscale or RGB-compatible")
    return preprocess_handwritten(image)
