# converters.py

import colorsys

def convert(v, mode="lum") -> int:
    match mode:
        case "lum": return get_luminance(v)
        case "hue": return get_hue(v)
        case "r"  : return get_r(v)
        case "g"  : return get_g(v)
        case "b"  : return get_b(v)
        case _:
            print("ERROR: Unsupported sort mode:", mode)
            raise

def get_luminance(v) -> int:
    # L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    if len(v) == 3:
        r, g, b = v
    else:
        print(f"\033[32m{v}\033[0;0m")
        raise
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def get_hue(v) -> int:
    r, g, b = [x / 255.0 for x in v]
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return int(h * 360)

def get_r(v) -> int: return v[0]
def get_g(v) -> int: return v[1]
def get_b(v) -> int: return v[2]


def rotate_coords(coords, img_size, angle):
    width, height = img_size
    if angle == 90:
        return [(y, width - 1 - x) for x, y in coords]
    elif angle == -90:
        return [(height - 1 - y, x) for x, y in coords]
    return coords  # 0° oder unsupported

