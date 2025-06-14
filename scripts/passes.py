# passes.py

from PIL import Image
import converters, math, random, threading
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from timing import timing
from math import exp, pi


def contrastMask(img: Image.Image, limLower, limUpper):
    out: Image.Image = Image.new("L", img.size)
    lowest = math.inf
    highest = -math.inf
    width, height = img.size
    for x in tqdm(range(width), desc="Processing contrast mask"):
        for y in range(height):
            val = math.floor(converters.getLuminance(img.getpixel((x, y))))
            
            if val < lowest:
                lowest = val
            if val > highest:
                highest = val
            if limLower < val <= limUpper:
                out.putpixel((x, y), 255)
            else:
                out.putpixel((x, y), 0)
    return out


def luminanceMask(img: Image.Image) -> Image.Image:
    out: Image.Image = Image.new("RGB", img.size)
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            val = math.floor(converters.getLuminance(img.getpixel((x,y))))
            out.putpixel((x,y), (val, val, val))
    return out


def getCoherentImageChunks(img: Image.Image, rotate=False) -> list[list[tuple[int, int]]]:
    img = img.convert("RGB")
    if rotate: img = img.rotate(90, expand=True)
    width, height = img.size
    visited = [[False for _ in range(height)] for _ in range(width)]
    chunks = []

    def get_neighbors(x, y):
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny

    for x in tqdm(range(width), desc="Finding coherent chunks"):
        for y in range(height):
            if not visited[x][y] and img.getpixel((x, y)) == (255, 255, 255):
                stack = [(x, y)]
                chunk = []
                visited[x][y] = True
                while stack:
                    cx, cy = stack.pop()
                    chunk.append((cx, cy))
                    for nx, ny in get_neighbors(cx, cy):
                        if not visited[nx][ny] and img.getpixel((nx, ny)) == (255, 255, 255):
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                if chunk:
                    chunks.append(chunk)
    return chunks

def toVerticalChunks(chunks: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    out = []
    for chunk in chunks:
        chunk = sorted(chunk)
        vChunks = []
        v = chunk[0][0]
        tmp = []
        for x in chunk:
            if x[0] == v:
                tmp.append(x)
            else:
                vChunks.append(tmp)
                tmp = [x]
                v = x[0]
        if tmp:
            vChunks.append(tmp)
        out.extend(vChunks)
    return out

def splitConnectedChunks(vChunks: list[list[tuple[int, int]]]) -> list[list[tuple[int,int]]]:
    out = []

    for vChunk in vChunks:
        if not vChunk:
            continue

        group = [vChunk[0]]
        for i in range(1, len(vChunk)):
            prev = vChunk[i - 1]
            curr = vChunk[i]

            # prüfen, ob direkt angrenzend (hier: vertikal -> y-Wert +1)
            if curr[1] == prev[1] + 1:
                group.append(curr)
            else:
                out.append(group)
                group = [curr]

        out.append(group)

    return out

def visualizeChunks(img: Image.Image, chunks, rotate=False):
    out = Image.new("RGB", img.size)
    for chunk in tqdm(chunks, desc="Visualizing chunks"):
        color = tuple(random.randint(0, 255) for _ in range(3))
        for x, y in chunk:
            out.putpixel((x, y), color)
    if rotate: out = out.rotate(-90, expand=True)
    return out


def sort(img: Image.Image, 
         chunks,
         mode="lum",
         flipDir=False,
         rotate=True
         ):
    if rotate:
        img = img.rotate(90, expand=True)

    out = img.copy()
    lock = threading.Lock()

    img_pixels = {(x, y): img.getpixel((x, y)) for x in range(img.width) for y in range(img.height)}

    def process_chunk(chunk):
        chunk_dict = {}

        for x, y in chunk:
            if x not in chunk_dict:
                chunk_dict[x] = []
            chunk_dict[x].append((y, img_pixels[(x, y)]))

        for x in chunk_dict:
            if not flipDir:
                sorted_pixels = sorted(chunk_dict[x], key=lambda tup: converters.convert(tup[1], mode=mode))
            else:
                sorted_pixels = sorted(chunk_dict[x], key=lambda tup: -converters.convert(tup[1], mode=mode) + 254)

            target_ys = sorted([y for y, _ in chunk_dict[x]])

            with lock:
                for i in range(len(target_ys)):
                    out.putpixel((x, target_ys[i]), sorted_pixels[i][1])

    threads = []
    for chunk in chunks:
        t = threading.Thread(target=process_chunk, args=(chunk,))
        t.start()
        threads.append(t)

    for t in tqdm(threads, desc="Waiting for chunks to finish"):
        t.join()

    if rotate:
        out = out.rotate(-90, expand=True)

    return out

@timing
def blurBox(img: Image.Image, kernel: int, num_threads: int = 64) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    original_pixels = img.load()

    output = Image.new("RGB", img.size)
    output_pixels = output.load()

    def blur_pixel(x: int, y: int) -> tuple[int, int, int]:
        x_min = max(x - kernel, 0)
        x_max = min(x + kernel + 1, width)
        y_min = max(y - kernel, 0)
        y_max = min(y + kernel + 1, height)

        r_total = g_total = b_total = 0
        count = 0

        for yy in range(y_min, y_max):
            for xx in range(x_min, x_max):
                r, g, b = original_pixels[xx, yy]
                r_total += r
                g_total += g
                b_total += b
                count += 1

        return (r_total // count, g_total // count, b_total // count)

    def process_rows(y_start: int, y_end: int):
        for y in range(y_start, y_end):
            for x in range(width):
                output_pixels[x, y] = blur_pixel(x, y)

    # Calculate thread area
    chunk_height = height // num_threads
    ranges = []
    for i in range(num_threads):
        y_start = i * chunk_height
        y_end = (i + 1) * chunk_height if i < num_threads - 1 else height
        ranges.append((y_start, y_end))

    # Threads starten
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(lambda args: process_rows(*args), ranges), total=len(ranges), desc="Applying threaded blur"))

    return output

@timing
def blurGaussian(img: Image.Image, kernel: int, sigma: float = 1.0, num_threads: int = 64) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    original_pixels = img.load()
    output = Image.new("RGB", img.size)
    output_pixels = output.load()

    # Gaussian kernel generation
    size = 2 * kernel + 1
    gaussian_kernel = [[0.0 for _ in range(size)] for _ in range(size)]
    sum_val = 0.0
    for y in range(-kernel, kernel + 1):
        for x in range(-kernel, kernel + 1):
            val = exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * pi * sigma**2)
            gaussian_kernel[y + kernel][x + kernel] = val
            sum_val += val
    # Normalize kernel
    for y in range(size):
        for x in range(size):
            gaussian_kernel[y][x] /= sum_val

    def blur_pixel(x: int, y: int) -> tuple[int, int, int]:
        r_total = g_total = b_total = 0.0
        for ky in range(-kernel, kernel + 1):
            for kx in range(-kernel, kernel + 1):
                nx = min(max(x + kx, 0), width - 1)
                ny = min(max(y + ky, 0), height - 1)
                weight = gaussian_kernel[ky + kernel][kx + kernel]
                r, g, b = original_pixels[nx, ny]
                r_total += r * weight
                g_total += g * weight
                b_total += b * weight
        return (int(r_total), int(g_total), int(b_total))

    def process_rows(y_start: int, y_end: int):
        for y in range(y_start, y_end):
            for x in range(width):
                output_pixels[x, y] = blur_pixel(x, y)

    chunk_height = height // num_threads
    ranges = []
    for i in range(num_threads):
        y_start = i * chunk_height
        y_end = (i + 1) * chunk_height if i < num_threads - 1 else height
        ranges.append((y_start, y_end))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(lambda args: process_rows(*args), ranges), total=len(ranges), desc="Applying gaussian blur"))

    return output


def gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    """Erzeuge eine 1D-Gaussian-Kernel."""
    radius = kernel_size // 2
    ax = np.arange(-radius, radius + 1)
    kernel = np.exp(-(ax**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def convolve_separable(img_array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_width = len(kernel) // 2
    padded = np.pad(img_array, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')

    temp = np.zeros_like(padded)
    output = np.zeros_like(padded)

    height, width = padded.shape[:2]

    # Horizontal Blur
    for c in range(3):
        for y in range(height):
            for x in range(pad_width, width - pad_width):
                window = padded[y, x - pad_width:x + pad_width + 1, c]
                temp[y, x, c] = np.sum(window * kernel)

    # Vertikal Blur
    for c in range(3):
        for y in range(pad_width, height - pad_width):
            for x in range(pad_width, width - pad_width):
                window = temp[y - pad_width:y + pad_width + 1, x, c]
                output[y, x, c] = np.sum(window * kernel)

    # Crop wieder rausnehmen
    cropped = output[pad_width:-pad_width, pad_width:-pad_width, :]
    return cropped


def process_chunk(sub_img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blurt ein gepaddetes Chunk und entfernt die Padding-Ränder."""
    blurred = convolve_separable(sub_img, kernel)
    return blurred


def blurGaussian1d(img: Image.Image, kernel: int, sigma: float = None, num_processes: int = 4) -> Image.Image:
    img = img.convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    height = img_array.shape[0]
    width = img_array.shape[1]

    # sigma automatisch bestimmen, falls nicht gesetzt
    if sigma is None:
        sigma = kernel / 2.0

    kernel_size = 2 * kernel + 1
    g_kernel = gaussian_kernel_1d(kernel_size, sigma)
    pad = kernel  # top & bottom padding

    # Chunks vorbereiten mit Padding
    chunk_height = height // num_processes
    chunks = []

    for i in range(num_processes):
        y_start = i * chunk_height
        y_end = (i + 1) * chunk_height if i < num_processes - 1 else height

        pad_start = max(y_start - pad, 0)
        pad_end = min(y_end + pad, height)

        chunk = img_array[pad_start:pad_end]
        chunks.append((chunk.copy(), g_kernel.copy()))  # copy for multiprocessing safety

    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_chunk, chunk, g_kernel) for chunk, g_kernel in chunks]
        for f in tqdm(futures, desc="Applying Gaussian Blur"):
            results.append(f.result())

    # Alle Blurred-Chunks zusammensetzen
    final_array = []

    for i, arr in enumerate(results):
        if i == 0:
            final_array.append(arr[:chunk_height])
        elif i == num_processes - 1:
            final_array.append(arr[-(height - chunk_height * i):])
        else:
            final_array.append(arr[pad:pad + chunk_height])

    combined = np.vstack(final_array).clip(0, 255).astype(np.uint8)
    return Image.fromarray(combined)


@timing
def subtractImages(i1: Image.Image, i2: Image.Image):
    if i1.size != i2.size:
        raise ValueError("Images must be the same size")

    out = i1.copy()

    for x in tqdm(range(i1.size[0]), desc="Calculating image delta"):
        for y in range(i1.size[1]):
            p1 = i1.getpixel((x, y))
            p2 = i2.getpixel((x, y))
            if isinstance(p1, int):  # L mode
                diff = max(0, p1 - p2)
                out.putpixel((x, y), diff)
            else:  # RGB or RGBA
                diff = tuple(max(0, a - b) for a, b in zip(p1, p2))
                out.putpixel((x, y), diff)
    return out

def adjustBrightness(img: Image.Image, gamma: float) -> Image.Image:
    out = img.copy()
    width, height = img.size

    for x in tqdm(range(width), desc="Adjusting image gamma"):
        for y in range(height):
            pixel = img.getpixel((x, y))
            
            if isinstance(pixel, int):  # 'L' mode (grayscale)
                v = min(int(pixel * gamma), 255)
                out.putpixel((x, y), v)

            elif len(pixel) == 3:  # RGB
                r, g, b = pixel
                new_pixel = (
                    min(int(r * gamma), 255),
                    min(int(g * gamma), 255),
                    min(int(b * gamma), 255)
                )
                out.putpixel((x, y), new_pixel)

            elif len(pixel) == 4:  # RGBA
                r, g, b, a = pixel
                new_pixel = (
                    min(int(r * gamma), 255),
                    min(int(g * gamma), 255),
                    min(int(b * gamma), 255),
                    a  # alpha bleibt gleich
                )
                out.putpixel((x, y), new_pixel)

            else:
                raise ValueError(f"Unsupported pixel format: {pixel}")

    return out