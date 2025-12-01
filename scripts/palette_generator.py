from __future__ import annotations
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw
import os


def generate_palette_image(colors: List[str], swatch_size: Tuple[int, int] = (64, 64), columns: int = 0,
                           bg: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Generate a palette image from a list of colors.

    colors: list of hex strings (e.g. '#ff00aa') or RGB tuples
    swatch_size: size of each color square
    columns: number of columns; if 0 will try to make a roughly square layout
    bg: background color
    """
    if not colors:
        raise ValueError("colors list must not be empty")

    # Normalize colors to RGB tuples
    def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
        if isinstance(h, (list, tuple)):
            return (int(h[0]), int(h[1]), int(h[2]))
        s = h.lstrip('#')
        if len(s) == 3:
            s = ''.join([c*2 for c in s])
        return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))

    rgb_colors = [_hex_to_rgb(c) for c in colors]

    n = len(rgb_colors)
    if columns <= 0:
        columns = int(n**0.5)
        columns = max(1, columns)
    rows = (n + columns - 1) // columns

    sw_w, sw_h = swatch_size
    img_w = sw_w * columns
    img_h = sw_h * rows

    out = Image.new('RGB', (img_w, img_h), bg)
    draw = ImageDraw.Draw(out)

    for i, col in enumerate(rgb_colors):
        cx = (i % columns) * sw_w
        cy = (i // columns) * sw_h
        draw.rectangle([cx, cy, cx + sw_w - 1, cy + sw_h - 1], fill=col)

    return out


def save_palette(colors: List[str], out_path: str, swatch_size: Tuple[int, int] = (64, 64), columns: int = 0):
    img = generate_palette_image(colors, swatch_size=swatch_size, columns=columns)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return out_path
