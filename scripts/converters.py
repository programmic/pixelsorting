# converters.py

import colorsys

def convert(v, mode="lum") -> int:
    mode_str = str(mode).lower()
    if mode_str == "lum":
        return get_luminance(v)
    elif mode_str == "hue":
        return get_hue(v)
    elif mode_str == "r":
        return get_r(v)
    elif mode_str == "g":
        return get_g(v)
    elif mode_str == "b":
        return get_b(v)
    else:
        print("ERROR: Unsupported sort mode:", mode)
        raise ValueError(f"Unsupported sort mode: {mode}")

def get_luminance(v) -> int:
    # L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    if len(v) == 3:
        r, g, b = v
    elif len(v) == 4:
        r, g, b, _ = v
    else:
        raise ValueError(f"Invalid input for luminance calculation: {v}. Expected a tuple of length 3 or 4 (RGB or RGBA).")
    return int(0.2126 * r + 0.7152 * g + 0.0722 * b)

def get_hue(v) -> int:
    r, g, b = [x / 255.0 for x in v]
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return int(h * 360)

def get_r(v) -> int: return v[0]
def get_g(v) -> int: return v[1]
def get_b(v) -> int: return v[2]


# Backward-compatible aliases for legacy callers
getLuminance = get_luminance
getHUE = get_hue
getR = get_r
getG = get_g
getB = get_b


def rotate_coords(coords, img_size, angle):
    width, height = img_size
    if angle == 90:
        return [(y, width - 1 - x) for x, y in coords]
    elif angle == -90:
        return [(height - 1 - y, x) for x, y in coords]
    return coords  # 0° oder unsupported

