# passes.py

from PIL import Image
import converters, math, random, threading, os
from tqdm import tqdm
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from timing import timing
from math import exp, pi
from typing import List, Tuple
import pyopencl as cl
import contextlib
import sys


def ensure_rgba(img: Image.Image) -> Image.Image:
    """Ensure image is in RGBA mode, preserving alpha if present."""
    if img.mode == 'RGBA':
        return img
    elif img.mode == 'RGB':
        # Add alpha channel with full opacity
        return img.convert('RGBA')
    elif img.mode == 'L':
        # Convert grayscale to RGBA
        rgb = img.convert('RGB')
        return rgb.convert('RGBA')
    else:
        # Handle other modes
        return img.convert('RGBA')


def get_pixel_rgba(pixel) -> tuple:
    """Get RGBA values from pixel, handling different modes."""
    if isinstance(pixel, int):
        # Grayscale
        return (pixel, pixel, pixel, 255)
    elif len(pixel) == 3:
        # RGB
        r, g, b = pixel
        return (r, g, b, 255)
    elif len(pixel) == 4:
        # RGBA
        return pixel
    else:
        # Fallback
        return (0, 0, 0, 255)


def put_pixel_rgba(img: Image.Image, x: int, y: int, rgba: tuple):
    """Put RGBA pixel, handling different image modes."""
    mode = img.mode
    if mode == 'RGBA':
        img.putpixel((x, y), rgba)
    elif mode == 'RGB':
        img.putpixel((x, y), rgba[:3])
    elif mode == 'L':
        # Convert RGBA to grayscale
        r, g, b, a = rgba
        gray = int(0.299 * r + 0.587 * g + 0.114 * b)
        img.putpixel((x, y), gray)
    else:
        img.putpixel((x, y), rgba[:3])


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def scale_image(img: Image.Image, copyImage: Image.Image, downscale: float= 0) -> Image.Image:
    if not 0 <= downscale <= 100: raise ValueError("ScaleImageError: Unsupported downscale value")
    
    # Create output image with transparent background (0 alpha)
    if copyImage.mode == 'RGBA':
        out = Image.new('RGBA', copyImage.size, (0, 0, 0, 0))
    else:
        # Convert to RGBA for transparency support
        out = Image.new('RGBA', copyImage.size, (0, 0, 0, 0))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    
    if downscale == 100:
        # At 100% downscale, image is invisible (fully transparent)
        return out
    
    if downscale == 0:
        # No scaling, center original image
        scale_factor = 1.0
    else:
        # Scale down by downscale percentage
        scale_factor = (100 - downscale) / 100.0
    
    # Calculate scaled dimensions
    new_width = max(1, int(img.width * scale_factor))
    new_height = max(1, int(img.height * scale_factor))
    
    # Scale the image
    if scale_factor < 1.0:
        scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        scaled_img = img
    
    # Calculate center position
    x_offset = (copyImage.width - new_width) // 2
    y_offset = (copyImage.height - new_height) // 2
    
    # Paste the scaled image centered on the canvas
    out.paste(scaled_img, (x_offset, y_offset))
    
    return out



def contrast_mask(img: Image.Image, lim_lower, lim_upper):
    out: Image.Image = Image.new("L", img.size)
    lowest = math.inf
    highest = -math.inf
    width, height = img.size
    for x in tqdm(range(width), desc="Processing contrast mask"):
        for y in range(height):
            val = math.floor(converters.get_luminance(img.getpixel((x, y))))
            
            if val < lowest:
                lowest = val
            if val > highest:
                highest = val
            if lim_lower < val <= lim_upper:
                out.putpixel((x, y), 255)
            else:
                out.putpixel((x, y), 0)
    return out

def luminance_mask(img: Image.Image) -> Image.Image:
    out: Image.Image = Image.new("RGB", img.size)
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            val = math.floor(converters.get_luminance(img.getpixel((x,y))))
            out.putpixel((x,y), (val, val, val))
    return out

def get_coherent_image_chunks(img: Image.Image, rotate=False) -> list[list[tuple[int, int]]]:
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

def to_vertical_chunks(chunks: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    out = []
    for chunk in chunks:
        chunk = sorted(chunk)
        v_chunks = []
        v = chunk[0][0]
        tmp = []
        for x in chunk:
            if x[0] == v:
                tmp.append(x)
            else:
                v_chunks.append(tmp)
                tmp = [x]
                v = x[0]
        if tmp:
            v_chunks.append(tmp)
        out.extend(v_chunks)
    return out

def split_connected_chunks(v_chunks: list[list[tuple[int, int]]]) -> list[list[tuple[int,int]]]:
    out = []

    for v_chunk in v_chunks:
        if not v_chunk:
            continue

        group = [v_chunk[0]]
        for i in range(1, len(v_chunk)):
            prev = v_chunk[i - 1]
            curr = v_chunk[i]

            # prüfen, ob direkt angrenzend (hier: vertikal -> y-Wert +1)
            if curr[1] == prev[1] + 1:
                group.append(curr)
            else:
                out.append(group)
                group = [curr]

        out.append(group)

    return out

def visualize_chunks(img: Image.Image, chunks, rotate=False):
    out = Image.new("RGB", img.size)
    for chunk in tqdm(chunks, desc="Visualizing chunks"):
        color = tuple(random.randint(0, 255) for _ in range(3))
        for x, y in chunk:
            out.putpixel((x, y), color)
    if rotate: out = out.rotate(-90, expand=True)
    return out

def wrap_sort(img: Image.Image,
              mode: str,
              vSplitting: bool,
              flipHorz: bool,
              flipVert: bool,
              rotate: str
              ) -> Image.Image:
    """
    Wrap sort function that translates UI settings to backend sort parameters.
    
    Args:
        img: Input PIL Image
        mode: Sort mode ("lum", "hue", "r", "g", "b")
        vSplitting: Vertical splitting flag (unused in sort, kept for compatibility)
        flipHorz: Horizontal flip flag
        flipVert: Vertical flip flag
        rotate: Rotation string ("0", "90", "180", "-90")
    
    Returns:
        Processed PIL Image
    """
    if mode.lower() not in ["lum", "hue", "r", "g", "b"]:
        raise ValueError("PixelSortError: Unsupported sort mode")
    
    # Translate rotation string to boolean for sort function
    rotate_bool = rotate != "0"
    
    # Translate flip flags to flip direction
    flip_dir = flipHorz or flipVert
    
    # Create mask based on vSplitting if needed
    mask = None
    
    # Call the actual sort function with translated parameters
    return sort(img, mode=mode, flip_dir=flip_dir, rotate=rotate_bool, mask=mask)

def sort(img: Image.Image,
         mode: str = "lum",
         flip_dir: bool = False,
         rotate: bool = True,
         mask: Image.Image = None
         ) -> Image.Image:
    # Generate chunks from mask if provided, otherwise sort the whole image
    if mask:
        chunks = get_coherent_image_chunks(mask, rotate)
    else:
        # Create a white mask for the whole image if no mask provided
        mask = Image.new("RGB", img.size, (255, 255, 255))
        chunks = get_coherent_image_chunks(mask, rotate)
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
            if not flip_dir:
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
def blur_box(img: Image.Image, kernel: int, num_threads: int = 64) -> Image.Image:
    img_rgba = ensure_rgba(img)
    width, height = img.size
    original_pixels = img_rgba.load()
    output = Image.new("RGBA", img.size)
    output_pixels = output.load()

    def blur_pixel(x: int, y: int) -> tuple[int, int, int, int]:
        x_min = max(x - kernel, 0)
        x_max = min(x + kernel + 1, width)
        y_min = max(y - kernel, 0)
        y_max = min(y + kernel + 1, height)

        r_total = g_total = b_total = a_total = 0
        count = 0

        for yy in range(y_min, y_max):
            for xx in range(x_min, x_max):
                r, g, b, a = original_pixels[xx, yy]
                r_total += r
                g_total += g
                b_total += b
                a_total += a
                count += 1

        return (r_total // count, g_total // count, b_total // count, a_total // count)

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
def blur_box_gpu(img: Image.Image, kernel_radius: int) -> Image.Image:
    """GPU-accelerated box blur with proper resource cleanup."""
    from scripts.gpu_context import opencl_context
    
    img = img.convert("RGB")
    img_np = np.array(img).astype(np.uint8)
    height, width, channels = img_np.shape

    with opencl_context() as (ctx, queue):
        # OpenCL-Programm (Box Blur)
        program_src = f"""
        __kernel void box_blur(__global const uchar* input,
                               __global uchar* output,
                               const int width,
                               const int height,
                               const int channels,
                               const int radius)
        {{
            int x = get_global_id(0);
            int y = get_global_id(1);
            
            if (x >= width || y >= height)
                return;

            for (int c = 0; c < channels; c++) {{
                int sum = 0;
                int count = 0;

                for (int ky = -radius; ky <= radius; ky++) {{
                    for (int kx = -radius; kx <= radius; kx++) {{
                        int nx = clamp(x + kx, 0, width - 1);
                        int ny = clamp(y + ky, 0, height - 1);
                        int idx = (ny * width + nx) * channels + c;
                        sum += input[idx];
                        count++;
                    }}
                }}

                int out_idx = (y * width + x) * channels + c;
                output[out_idx] = sum / count;
            }}
        }}
        """
        with suppress_output():
            program = cl.Program(ctx, program_src).build()

        # Eingabe & Ausgabe-Puffer
        flat_input = img_np.flatten()
        output_np = np.empty_like(flat_input)

        mf = cl.mem_flags
        input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat_input)
        output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_np.nbytes)

        # Kernel ausführen
        kernel = program.box_blur
        kernel.set_args(input_buf, output_buf,
                        np.int32(width), np.int32(height),
                        np.int32(channels), np.int32(kernel_radius))

        global_size = (width, height)
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
        cl.enqueue_copy(queue, output_np, output_buf)
        queue.finish()

        # In Bild zurückverwandeln
        output_img = output_np.reshape((height, width, channels))
        return Image.fromarray(output_img, 'RGB')

@timing
def blur_gaussian(img: Image.Image, kernel: int, sigma: float = 1.0, num_threads: int = 64) -> Image.Image:
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

def __gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    """Erzeuge eine 1D-Gaussian-Kernel."""
    radius = kernel_size // 2
    ax = np.arange(-radius, radius + 1)
    kernel = np.exp(-(ax**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def __convolve_separable(img_array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
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

def __process_chunk(sub_img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blurt ein gepaddetes Chunk und entfernt die Padding-Ränder."""
    blurred = __convolve_separable(sub_img, kernel)
    return blurred

def __blur_gaussian_1d(img: Image.Image, kernel: int, sigma: float = None, num_processes: int = 4) -> Image.Image:
    img = img.convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    height = img_array.shape[0]

    # sigma automatisch bestimmen, falls nicht gesetzt
    if sigma is None:
        sigma = kernel / 2.0

    kernel_size = 2 * kernel + 1
    g_kernel = __gaussian_kernel_1d(kernel_size, sigma)
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
        futures = [executor.submit(__process_chunk, chunk, g_kernel) for chunk, g_kernel in chunks]
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

def subtract_images(i1: Image.Image, i2: Image.Image) -> Image.Image:
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

def adjust_brightness(img: Image.Image, gamma: float) -> Image.Image:
    out = img.copy()
    width, height = img.size

    for x in tqdm(range(width), desc="Adjusting image gamma"):
        inv_gamma = 1.0 / gamma
        for y in range(height):
            pixel = img.getpixel((x, y))
            
            if isinstance(pixel, int):  # 'L' mode (grayscale)
                v = min(int(pixel + (gamma * (255 - r))), 255),
                out.putpixel((x, y), v)

            elif len(pixel) == 3:  # RGB
                r, g, b = pixel
                new_pixel = (
                    int((r / 255.0) ** inv_gamma * 255),
                    int((g / 255.0) ** inv_gamma * 255),
                    int((b / 255.0) ** inv_gamma * 255)
                )
                out.putpixel((x, y), new_pixel)

            elif len(pixel) == 4:  # RGBA
                r, g, b, a = pixel
                new_pixel = (
                    int((r / 255.0) ** inv_gamma * 255),
                    int((g / 255.0) ** inv_gamma * 255),
                    int((b / 255.0) ** inv_gamma * 255),
                    a  # alpha bleibt gleich
                )
                out.putpixel((x, y), new_pixel)

            else:
                raise ValueError(f"Unsupported pixel format: {pixel}")

    return out

def __get_standard_deviation(p1, p2):
    r1 ,g1, b1 = p1
    r2, g2, b2 = p2
    return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)

def __box_blur_np(arr, kernel):
    """Fast box blur for 2D numpy arrays with padding."""
    k = 2 * kernel + 1
    # Cumulative sum over rows and columns
    cumsum = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    # Pad cumsum to simplify indexing
    cumsum = np.pad(cumsum, ((1, 0), (1, 0)), mode='constant', constant_values=0)
    h, w = arr.shape
    out = np.empty_like(arr)
    for i in tqdm(range(h), desc="Converting image for kuwahara filter"):
        for j in range(w):
            y1 = i
            x1 = j
            y2 = min(i + k, h)
            x2 = min(j + k, w)
            y0 = max(i - kernel, 0)
            x0 = max(j - kernel, 0)
            out[i, j] = (
                cumsum[y2, x2]
                - cumsum[y2, x0]
                - cumsum[y0, x2]
                + cumsum[y0, x0]
            ) / (k * k)
    return out

def __process_rows(y_start, y_end, padded_gray, padded_img, kernel, height, width):
    result = np.zeros((y_end - y_start, width, 3), dtype=np.float32)
    half = kernel + 1

    for y in range(y_start, y_end):
        for x in range(width):
            gray_window = padded_gray[y:y+2*kernel+1, x:x+2*kernel+1]
            color_window = padded_img[y:y+2*kernel+1, x:x+2*kernel+1, :]

            regions_gray = [
                gray_window[:half, :half],
                gray_window[:half, kernel:],
                gray_window[kernel:, kernel:],
                gray_window[kernel:, :half]
            ]
            regions_color = [
                color_window[:half, :half, :],
                color_window[:half, kernel:, :],
                color_window[kernel:, kernel:, :],
                color_window[kernel:, :half, :]
            ]

            variances = [np.var(r) for r in regions_gray]
            idx = np.argmin(variances)
            mean_color = np.mean(regions_color[idx], axis=(0, 1))
            result[y - y_start, x] = mean_color

    return result

def kuwahara(img: Image.Image, kernel: int, num_threads: int = None) -> Image.Image:
    if num_threads is None:
        num_threads = os.cpu_count() or 4

    img_np = np.array(img, dtype=np.float32)  # shape (H, W, 3)
    height, width, _ = img_np.shape

    img_gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114])
    padded_gray = np.pad(img_gray, kernel, mode='reflect')
    padded_img = np.pad(img_np, ((kernel, kernel), (kernel, kernel), (0, 0)), mode='reflect')

    out = np.zeros_like(img_np)

    # Zeilen in Blöcke aufteilen
    chunk_size = height // num_threads
    ranges = [(i*chunk_size, (i+1)*chunk_size if i < num_threads - 1 else height) for i in range(num_threads)]

    print(f"Processing with {num_threads} threads...")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(__process_rows, y_start, y_end, padded_gray, padded_img, kernel, height, width)
            for y_start, y_end in ranges
        ]

        for i, future in enumerate(tqdm(futures, desc="Combining thread results")):
            result_chunk = future.result()
            out[ranges[i][0]:ranges[i][1]] = result_chunk

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def kuwahara_gpu(img: Image.Image, kernel_size: float) -> Image.Image:
    """
    Enhanced kuwahara filter with GPU acceleration and CPU fallback.
    Handles OpenCL errors gracefully and provides fallback to CPU implementation.
    Uses proper OpenCL resource cleanup to prevent memory leaks.
    """
    # Convert kernel_size to integer
    kernel_size = int(kernel_size)
    try:
        from gpu_context import opencl_context
    except ImportError:
        print("Could not import GPU context, using CPU fallback...")
        return kuwahara(img, kernel_size)
    
    def _kuwahara_cpu_fallback(img: Image.Image, ksize: int) -> Image.Image:
        """CPU fallback implementation when GPU/OpenCL is unavailable."""
        print("GPU/OpenCL unavailable, using CPU fallback...")
        return kuwahara(img, ksize)
    
    try:
        img = img.convert("RGB")
        img_np = np.array(img).astype(np.float32)
        height, width, channels = img_np.shape
        
        with opencl_context() as (ctx, queue):
            # Build OpenCL program
            program_src = f"""
            __kernel void kuwahara_filter(
                __global const float* input,
                __global float* output,
                const int width,
                const int height,
                const int channels,
                const int radius
            ) {{
                int x = get_global_id(0);
                int y = get_global_id(1);

                if (x >= width || y >= height)
                    return;

                float mean[4][3] = {{0}};
                float var[4] = {{0}};
                int count[4] = {{0}};

                // Calculate regions and their statistics
                // Top-left region
                for (int dy = -radius; dy <= 0; dy++) {{
                    for (int dx = -radius; dx <= 0; dx++) {{
                        int nx = clamp(x + dx, 0, width - 1);
                        int ny = clamp(y + dy, 0, height - 1);
                        int idx = (ny * width + nx) * channels;
                        
                        for (int c = 0; c < channels; c++) {{
                            mean[0][c] += input[idx + c];
                        }}
                        
                        float gray = 0.299f * input[idx] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
                        var[0] += gray * gray;
                        count[0]++;
                    }}
                }}

                // Top-right region
                for (int dy = -radius; dy <= 0; dy++) {{
                    for (int dx = 0; dx <= radius; dx++) {{
                        int nx = clamp(x + dx, 0, width - 1);
                        int ny = clamp(y + dy, 0, height - 1);
                        int idx = (ny * width + nx) * channels;
                        
                        for (int c = 0; c < channels; c++) {{
                            mean[1][c] += input[idx + c];
                        }}
                        
                        float gray = 0.299f * input[idx] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
                        var[1] += gray * gray;
                        count[1]++;
                    }}
                }}

                // Bottom-left region
                for (int dy = 0; dy <= radius; dy++) {{
                    for (int dx = -radius; dx <= 0; dx++) {{
                        int nx = clamp(x + dx, 0, width - 1);
                        int ny = clamp(y + dy, 0, height - 1);
                        int idx = (ny * width + nx) * channels;
                        
                        for (int c = 0; c < channels; c++) {{
                            mean[2][c] += input[idx + c];
                        }}
                        
                        float gray = 0.299f * input[idx] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
                        var[2] += gray * gray;
                        count[2]++;
                    }}
                }}

                // Bottom-right region
                for (int dy = 0; dy <= radius; dy++) {{
                    for (int dx = 0; dx <= radius; dx++) {{
                        int nx = clamp(x + dx, 0, width - 1);
                        int ny = clamp(y + dy, 0, height - 1);
                        int idx = (ny * width + nx) * channels;
                        
                        for (int c = 0; c < channels; c++) {{
                            mean[3][c] += input[idx + c];
                        }}
                        
                        float gray = 0.299f * input[idx] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
                        var[3] += gray * gray;
                        count[3]++;
                    }}
                }}

                // Find region with minimum variance
                float min_var = 1e20f;
                int best_region = 0;

                for (int i = 0; i < 4; i++) {{
                    if (count[i] > 0) {{
                        float variance = var[i] / count[i];
                        if (variance < min_var) {{
                            min_var = variance;
                            best_region = i;
                        }}
                    }}
                }}

                // Calculate mean color for best region
                int out_idx = (y * width + x) * channels;
                for (int c = 0; c < channels; c++) {{
                    output[out_idx + c] = mean[best_region][c] / max(count[best_region], 1);
                }}
            }}
            """
            
            # Build program
            with suppress_output():
                program = cl.Program(ctx, program_src).build()
            
            # Prepare buffers
            flat_input = img_np.flatten()
            output_np = np.empty_like(flat_input)
            
            mf = cl.mem_flags
            input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat_input)
            output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_np.nbytes)
            
            # Execute kernel
            compute_kernel = program.kuwahara_filter
            compute_kernel.set_args(input_buf, output_buf,
                            np.int32(width), np.int32(height),
                            np.int32(channels), np.int32(kernel_size))
            
            global_size = (width, height)
            cl.enqueue_nd_range_kernel(queue, compute_kernel, global_size, None)
            cl.enqueue_copy(queue, output_np, output_buf)
            queue.finish()
            
            # Reshape output
            output_img = output_np.reshape((height, width, channels))
            output_img = np.clip(output_img, 0, 255).astype(np.uint8)
            return Image.fromarray(output_img, 'RGB')
        
    except Exception as e:
        print(f"OpenCL error: {str(e)}. Falling back to CPU implementation.")
        return _kuwahara_cpu_fallback(img, kernel_size)

def anisotropic_kuwahara_gpu(img: Image.Image, kernel_size: float, regions: int = 8) -> Image.Image:
    """
    Anisotropic Kuwahara filter with GPU acceleration and CPU fallback.
    Uses local structure tensor to adapt region orientation.
    """
    kernel_size = int(kernel_size)

    try:
        from gpu_context import opencl_context
    except ImportError:
        print("Could not import GPU context, using CPU fallback...")
        return kuwahara(img, kernel_size)  # fallback to your CPU implementation

    def _cpu_fallback(img: Image.Image, ksize: int, regions: int) -> Image.Image:
        print("GPU/OpenCL unavailable, using CPU fallback...")
        return kuwahara(img, ksize)  # keep your classic CPU Kuwahara as fallback

    try:
        img = img.convert("RGB")
        img_np = np.array(img).astype(np.float32)
        height, width, channels = img_np.shape

        with opencl_context() as (ctx, queue):
            program_src = f"""
            __kernel void anisotropic_kuwahara(
                __global const float* input,
                __global float* output,
                const int width,
                const int height,
                const int channels,
                const int radius,
                const int regions
            ) {{
                int x = get_global_id(0);
                int y = get_global_id(1);

                if (x >= width || y >= height)
                    return;

                // --- compute gradients for structure tensor ---
                int idx = (y * width + x) * channels;
                float gx = 0.0f, gy = 0.0f;
                if (x > 0 && x < width-1 && y > 0 && y < height-1) {{
                    int idx_left  = (y * width + (x-1)) * channels;
                    int idx_right = (y * width + (x+1)) * channels;
                    int idx_up    = ((y-1) * width + x) * channels;
                    int idx_down  = ((y+1) * width + x) * channels;
                    
                    float gray_left  = 0.299f*input[idx_left] + 0.587f*input[idx_left+1] + 0.114f*input[idx_left+2];
                    float gray_right = 0.299f*input[idx_right] + 0.587f*input[idx_right+1] + 0.114f*input[idx_right+2];
                    float gray_up    = 0.299f*input[idx_up] + 0.587f*input[idx_up+1] + 0.114f*input[idx_up+2];
                    float gray_down  = 0.299f*input[idx_down] + 0.587f*input[idx_down+1] + 0.114f*input[idx_down+2];

                    gx = gray_right - gray_left;
                    gy = gray_down - gray_up;
                }}
                float orientation = atan2(gy, gx);

                // --- accumulate statistics in wedge-shaped regions ---
                float mean[16][3]; // up to 16 regions
                float var[16];
                int count[16];
                for (int i = 0; i < regions; i++) {{
                    for (int c = 0; c < 3; c++) mean[i][c] = 0.0f;
                    var[i] = 0.0f;
                    count[i] = 0;
                }}

                for (int dy = -radius; dy <= radius; dy++) {{
                    for (int dx = -radius; dx <= radius; dx++) {{
                        int nx = clamp(x + dx, 0, width - 1);
                        int ny = clamp(y + dy, 0, height - 1);
                        int nidx = (ny * width + nx) * channels;

                        float gray = 0.299f*input[nidx] + 0.587f*input[nidx+1] + 0.114f*input[nidx+2];
                        float angle = atan2((float)dy, (float)dx) - orientation;

                        // map angle to [0, 2pi)
                        while (angle < 0) angle += 6.2831853f;
                        while (angle >= 6.2831853f) angle -= 6.2831853f;

                        int region = (int)(angle / (6.2831853f / regions));
                        if (region >= regions) region = regions-1;

                        for (int c = 0; c < channels; c++) {{
                            mean[region][c] += input[nidx + c];
                        }}
                        var[region] += gray * gray;
                        count[region]++;
                    }}
                }}

                // choose region with minimum variance
                float min_var = 1e20f;
                int best_region = 0;
                for (int i = 0; i < regions; i++) {{
                    if (count[i] > 0) {{
                        float variance = var[i] / count[i];
                        if (variance < min_var) {{
                            min_var = variance;
                            best_region = i;
                        }}
                    }}
                }}

                // assign pixel color
                int out_idx = (y * width + x) * channels;
                for (int c = 0; c < channels; c++) {{
                    output[out_idx + c] = mean[best_region][c] / max(count[best_region], 1);
                }}
            }}
            """

            # Build program
            with suppress_output():
                program = cl.Program(ctx, program_src).build()

            # Prepare buffers
            flat_input = img_np.flatten()
            output_np = np.empty_like(flat_input)

            mf = cl.mem_flags
            input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat_input)
            output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_np.nbytes)

            # Execute kernel
            compute_kernel = program.anisotropic_kuwahara
            compute_kernel.set_args(input_buf, output_buf,
                                    np.int32(width), np.int32(height),
                                    np.int32(channels), np.int32(kernel_size), np.int32(regions))

            global_size = (width, height)
            cl.enqueue_nd_range_kernel(queue, compute_kernel, global_size, None)
            cl.enqueue_copy(queue, output_np, output_buf)
            queue.finish()

            # Reshape output
            output_img = output_np.reshape((height, width, channels))
            output_img = np.clip(output_img, 0, 255).astype(np.uint8)
            return Image.fromarray(output_img, 'RGB')

    except Exception as e:
        print(f"OpenCL error: {str(e)}. Falling back to CPU implementation.")
        return _cpu_fallback(img, kernel_size, regions)


def kuwahara_grays(img: Image.Image, kernel: int) -> Image.Image:
    print("converting image...")
    img_np = np.array(img.convert("L"), dtype=np.float32)
    print("1/6")

    img_box_blur = blur_box(img, kernel, num_threads=96)
    print("2/6")

    img_box_blur_l = img_box_blur.convert("L")
    print("3/6")

    m_m = np.array(img_box_blur_l, dtype=np.float32)
    print("4/6")
    m_p_squared = img_np * img_np
    print("5/6")

    num_rows, num_cols = img_np.shape
    print("6/6")
    

    m_v = __box_blur_np(m_p_squared, kernel) - m_m * m_m
    m_v = np.maximum(m_v, 0)

    m_m = np.pad(m_m, kernel, mode='reflect')
    m_v = np.pad(m_v, kernel, mode='reflect')

    m_o = np.empty_like(img_np)

    for ii in tqdm(range(num_rows), desc="Applying Kuwahara filter"):
        for jj in range(num_cols):
            rr = ii + kernel
            cc = jj + kernel

            variances = [
                m_v[rr - kernel, cc - kernel],  # Top-left
                m_v[rr - kernel, cc + kernel],  # Top-right
                m_v[rr + kernel, cc + kernel],  # Bottom-right
                m_v[rr + kernel, cc - kernel],  # Bottom-left
            ]
            std_arg = np.argmin(variances)

            means = [
                m_m[rr - kernel, cc - kernel],  # Top-left
                m_m[rr - kernel, cc + kernel],  # Top-right
                m_m[rr + kernel, cc + kernel],  # Bottom-right
                m_m[rr + kernel, cc - kernel],  # Bottom-left
            ]
            m_o[ii, jj] = means[std_arg]

    m_o = np.clip(m_o, 0, 255).astype(np.uint8)
    return Image.fromarray(m_o)

def cristalline_expansion(img: Image.Image, c: int) -> Image.Image:
    max_x, max_y = img.size
    out = Image.new("RGB", (max_x, max_y))

    if c <= 0:
        raise ValueError("Sample count must be positive")
    elif c >= max_x * max_y:
        print("Warning: Sample count is too high")
        c = int(max_x * max_y / 5000)
        print("Set new sample count:", c)

    all_coords = [(x, y) for x in tqdm(range(max_x), desc="Generating Seeds") for y in range(max_y)]
    points = random.sample(all_coords, c)

    field = -1 * np.ones((max_y, max_x), dtype=np.int32)
    colors = []

    growth = deque()

    for i, (x, y) in enumerate(points):
        field[y, x] = i
        growth.append((x, y, i))  # x, y, crystal-id
        colors.append(img.getpixel((x, y)))  

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    out_pixels = out.load()

    for (x, y), color in zip(points, colors):
        out_pixels[x, y] = color

    print("  Growing crystals: Step 0",end="\r")
    count = 0
    while growth:
        x, y, cid = growth.popleft()

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < max_x and 0 <= ny < max_y and field[ny, nx] == -1:
                field[ny, nx] = cid
                growth.append((nx, ny, cid))
                out_pixels[nx, ny] = colors[cid]
        if count % 10000 == 0:
            print("  Growing Crystals: Step", count, end="\r")
        count += 1

    return out

def mix_percent(a:Image.Image, b: Image.Image, p: int):
    if a.size != b.size:
        raise ValueError("Error: Images must be same size")
    if not ( 0 < p <= 100 ):
        raise ValueError("Error Percent alue may not be outside 0 < p <= 100")
    
    out = Image.new("RGB", a.size)
    for x in tqdm(range(a.size[0]), desc="Mixing images"):
        for y in range(a.size[1]):
            pixel_a = a.getpixel((x,y))
            pixel_b = b.getpixel((x,y))

            v_r = max(0, min(int((pixel_a[0] * (100-p) + pixel_b[0] * p) / 100), 255))
            v_g = max(0, min(int((pixel_a[1] * (100-p) + pixel_b[1] * p) / 100), 255))
            v_b = max(0, min(int((pixel_a[2] * (100-p) + pixel_b[2] * p) / 100), 255))

            out.putpixel((x,y), (v_r,v_g,v_b))
    return out

# Function aliases to match render pass names from renderPasses.json
def mix_by_percent(img1: Image.Image, img2: Image.Image, mix_factor: float) -> Image.Image:
    """Alias for mix_percent to match render pass name."""
    return mix_percent(img1, img2, int(mix_factor))

def blur(img: Image.Image, blur_type: str, blur_kernel: int) -> Image.Image:
    """Alias for blur functions to match render pass name."""
    if blur_type == "Box":
        return blur_box(img, int(blur_kernel))
    elif blur_type == "Gaussian":
        return blur_gaussian(img, int(blur_kernel))
    else:
        return blur_box(img, int(blur_kernel))

def invert(img: Image.Image, invert_type: str, impact_factor: float) -> Image.Image:
    """
    Inverts image colors based on specified type and impact factor.
    
    Args:
        img: Input PIL Image
        invert_type: Type of inversion ("RGB", "R", "G", "B", "Luminance")
        impact_factor: Strength of inversion effect (0-100)
    
    Returns:
        Inverted PIL Image
    """
    out = img.copy()
    width, height = img.size
    
    impact = impact_factor / 100.0
    
    for x in tqdm(range(width), desc=f"Inverting {invert_type}"):
        for y in range(height):
            pixel = img.getpixel((x, y))
            
            # Grayscale (single int)
            if isinstance(pixel, int):
                inverted = 255 - pixel
                new_val = int((1 - impact) * pixel + impact * inverted)
                out.putpixel((x, y), new_val)
            
            # RGB tuple
            else:
                r, g, b = pixel[:3]
                
                if invert_type == "RGB":
                    inv_r, inv_g, inv_b = 255 - r, 255 - g, 255 - b
                elif invert_type == "R":
                    inv_r, inv_g, inv_b = 255 - r, g, b
                elif invert_type == "G":
                    inv_r, inv_g, inv_b = r, 255 - g, b
                elif invert_type == "B":
                    inv_r, inv_g, inv_b = r, g, 255 - b
                elif invert_type == "Luminance":
                    lum = int(0.299 * r + 0.587 * g + 0.114 * b)
                    inv_lum = 255 - lum
                    diff = inv_lum - lum
                    inv_r, inv_g, inv_b = r + diff, g + diff, b + diff
                else:
                    raise ValueError("Invalid invert_type. Use: 'RGB', 'R', 'G', 'B', 'Luminance'")
                
                # Clamp to [0, 255]
                inv_r, inv_g, inv_b = max(0, min(255, inv_r)), max(0, min(255, inv_g)), max(0, min(255, inv_b))
                
                # Apply impact factor (blend original with inverted)
                new_r = int((1 - impact) * r + impact * inv_r)
                new_g = int((1 - impact) * g + impact * inv_g)
                new_b = int((1 - impact) * b + impact * inv_b)
                
                if len(pixel) == 4:  # Preserve alpha
                    out.putpixel((x, y), (new_r, new_g, new_b, pixel[3]))
                else:
                    out.putpixel((x, y), (new_r, new_g, new_b))
    
    return out

def alpha_over(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """
    Perform alpha compositing (over operation) of img2 over img1.
    Only img2 (top layer) needs to have alpha channel, img1 can be any mode.
    
    Args:
        img1: Background image (bottom layer, can be RGB or RGBA)
        img2: Foreground image (top layer with alpha)
    
    Returns:
        Composited image with proper alpha blending
    """
    if img1.size != img2.size:
        raise ValueError("Images must be of same size")
    
    # Ensure img1 is RGB (ignore alpha if present), img2 must be RGBA
    img1_rgb = img1.convert("RGB")
    img2_rgba = ensure_rgba(img2)
    
    width, height = img1.size
    output = Image.new("RGB", (width, height))
    
    # Get pixel access
    pixels1 = img1_rgb.load()
    pixels2 = img2_rgba.load()
    output_pixels = output.load()
    
    # Perform alpha compositing for each pixel
    for x in tqdm(range(width), desc="Alpha compositing"):
        for y in range(height):
            # Get RGB values for background, RGBA for foreground
            r1, g1, b1 = pixels1[x, y]
            r2, g2, b2, a2 = pixels2[x, y]
            
            # Normalize alpha value to 0-1 range
            alpha2 = a2 / 255.0
            
            # Perform over operation: C = C2 * α2 + C1 * (1 - α2)
            out_r = int(r2 * alpha2 + r1 * (1 - alpha2))
            out_g = int(g2 * alpha2 + g1 * (1 - alpha2))
            out_b = int(b2 * alpha2 + b1 * (1 - alpha2))
            
            output_pixels[x, y] = (out_r, out_g, out_b)
    
    return output
    