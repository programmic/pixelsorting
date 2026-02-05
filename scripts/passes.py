# passes.py
"""Image processing functions for pixel sorting and manipulation."""

from __future__ import annotations
from typing import List, Tuple, Optional, Union
import contextlib
import math
import os
import random
import sys
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from math import exp, pi

import colorsys
import numpy as np
import pyopencl as cl
from PIL import Image
from tqdm import tqdm
from enum import Enum

from . import converters
from .timing import timing

def getImageData(img: Image.Image) -> dict:
    """Returns dictionary with image metadata.\nDictionary: width, height, mode, info, format"""
    width, height = img.size
    mode = img.mode
    info = img.info
    format = img.format
    return {
        "width": width,
        "height": height,
        "mode": mode,
        "info": info,
        "format": format
    }

def _ensure_rgba(img: Image.Image) -> Image.Image: #globalignore
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

def _get_pixel_rgba(pixel: Union[int, Tuple[int, ...]]) -> Tuple[int, int, int, int]: #globalignore
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

def _put_pixel_rgba(img: Image.Image, x: int, y: int, rgba: Tuple[int, int, int, int]) -> None: #globalignore
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
def _suppress_output(): #globalignore
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

def scale_image(img: Image.Image, copyImage: Image.Image, downscale: float = 0) -> Image.Image:  #globalignore
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

def generate_contrast_mask(img: Image.Image, limMin: int, limMax: int) -> Image.Image:
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
            if limMin < val <= limMax:
                out.putpixel((x, y), 255)
            else:
                out.putpixel((x, y), 0)
    return out

def luminance_mask(img: Image.Image, mask: Optional[Image.Image] = None) -> Image.Image:
    out: Image.Image = Image.new("RGB", img.size)
    
    # Convert mask to grayscale if provided
    mask_gray = None
    if mask:
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.Resampling.LANCZOS)
        mask_gray = mask.convert("L")
    
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            val = math.floor(converters.get_luminance(img.getpixel((x,y))))
            
            # Apply mask intensity if mask is provided
            if mask_gray:
                mask_val = mask_gray.getpixel((x, y))
                # Scale the luminance value by mask brightness (0-255)
                val = int(val * (mask_val / 255.0))
            
            out.putpixel((x,y), (val, val, val))
    return out

def get_coherent_image_chunks(img: Image.Image, rotate: bool = False) -> list[list[tuple[int, int]]]: #globalignore
    img = img.convert("RGB")
    if rotate: img = img.rotate(90, expand=True)
    width, height = img.size
    visited = [[False for _ in range(height)] for _ in range(width)]
    chunks = []

    def get_neighbors(x, y): #globalignore
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

def to_vertical_chunks(chunks: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]: #globalignore
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

def split_connected_chunks(v_chunks: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]: #globalignore
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

def visualize_chunks(img: Image.Image, chunks: list[list[tuple[int, int]]], rotate: bool = False) -> Image.Image: #globalignore
    out = Image.new("RGB", img.size)
    for chunk in tqdm(chunks, desc="Visualizing chunks"):
        color = tuple(random.randint(0, 255) for _ in range(3))
        for x, y in chunk:
            out.putpixel((x, y), color)
    if rotate: out = out.rotate(-90, expand=True)
    return out #globalignore

def wrap_sort(img: Image.Image,
              mode: 'str',
              vSplitting: bool,
              flipHorz: bool,
              flipVert: bool,
              rotate: str,
              progress: Optional[callable] = None
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
    
    # Translate rotation string to boolean for sort function
    rotate_bool = rotate != "0"
    
    # Translate flip flags to flip direction
    flip_dir = flipHorz or flipVert
    
    # Create mask based on vSplitting if needed
    mask = None
    
    # Call the actual sort function with translated parameters, including vSplitting
    try:
        if progress:
            progress({'percent': 0, 'message': 'Starting sort'})
    except Exception:
        pass
    out = sort(img, mode, vSplitting=vSplitting, flip_dir=flip_dir, rotate=rotate_bool, mask=mask, progress=progress)
    try:
        if progress:
            progress({'percent': 100, 'message': 'Sort completed'})
    except Exception:
        pass
    return out

def sort(img: Image.Image,
         mode: str = "lum",
         vSplitting: bool = True,
         flip_dir: bool = False,
         rotate: bool = True,
         mask: Optional[Image.Image] = None,
         progress: Optional[callable] = None
         ) -> Image.Image: #globalignore
    # Generate chunks from mask if provided, otherwise sort the whole image
    if mask:
        # Ensure mask is in RGB format for get_coherent_image_chunks
        if mask.mode != 'RGB':
            mask = mask.convert('RGB')
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.Resampling.LANCZOS)
        chunks = get_coherent_image_chunks(mask, rotate)
    else:
        # Create a white mask for the whole image if no mask provided
        mask = Image.new("RGB", img.size, (255, 255, 255))
        chunks = get_coherent_image_chunks(mask, rotate)
    # If vertical splitting requested, break chunks into vertical runs
    if vSplitting:
        try:
            v_chunks = to_vertical_chunks(chunks)
            chunks = split_connected_chunks(v_chunks)
        except Exception:
            # In case of any issue with splitting, fall back to original chunks
            pass
    # Do not show the mask during processing (avoids popping GUI windows during
    # automated runs/tests). If you need to debug visually, enable explicitly.
    
    if rotate:
        img = img.rotate(90, expand=True)

    out = img.copy()
    lock = threading.Lock()

    img_pixels = {(x, y): img.getpixel((x, y)) for x in range(img.width) for y in range(img.height)}

    def process_chunk(chunk):
        """
        Process a single chunk of pixels, applying the sorting effect only to white areas of the mask.
        """
        # Compute the list of coordinates that are white in the mask and the
        # corresponding pixel values from the source image. We must preserve
        # the coordinate list so we can write sorted pixels back to the exact
        # positions that were part of the sort region.
        coords_to_sort = [coord for coord in chunk if mask.getpixel(coord) == (255, 255, 255)]
        if not coords_to_sort:
            return
        chunk_pixels = [img_pixels[coord] for coord in coords_to_sort]

        # Sort the chunk based on the selected mode
        if mode == "lum":
            # Use the project's luminance helper for perceptual luminance
            try:
                chunk_pixels.sort(key=lambda px: converters.get_luminance(px))
            except Exception:
                chunk_pixels.sort(key=lambda px: sum(px[:3]) / 3)
        elif mode == "hue":
            chunk_pixels.sort(key=lambda px: colorsys.rgb_to_hsv(*px[:3])[0])
        elif mode == "r":
            chunk_pixels.sort(key=lambda px: px[0])
        elif mode == "g":
            chunk_pixels.sort(key=lambda px: px[1])
        elif mode == "b":
            chunk_pixels.sort(key=lambda px: px[2])

        # Write sorted pixels back to the image at the same coordinates that
        # were selected for sorting (coords_to_sort). Use the lock to avoid
        # concurrent writes to the PIL image object.
        with lock:
            for coord, pixel in zip(coords_to_sort, chunk_pixels):
                out.putpixel(coord, pixel)

    threads = []
    for chunk in chunks:
        t = threading.Thread(target=process_chunk, args=(chunk,))
        t.start()
        threads.append(t)

    total_chunks = len(threads) if threads else 1
    for i, t in enumerate(threads):
        t.join()
        try:
            if progress:
                pct = int(((i + 1) / total_chunks) * 100)
                progress({'percent': pct, 'message': f'Chunk {i+1}/{total_chunks}'})
        except Exception:
            pass

    if rotate:
        out = out.rotate(-90, expand=True)

    return out

@timing
def blur_box(img: Image.Image, kernel: int, num_threads: int = 64) -> Image.Image: #globalignore
    img_rgba = _ensure_rgba(img)
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
def blur_box_gpu(img: Image.Image, blur_kernel: int) -> Image.Image: #globalignore
    """GPU-accelerated box blur with proper resource cleanup. Accepts kernel_size and converts to radius."""
    from gpu_context import opencl_context

    img = img.convert("RGB")
    img_np = np.array(img).astype(np.uint8)
    height, width, channels = img_np.shape

    # Convert kernel_size to radius
    radius = (blur_kernel - 1) // 2

    with opencl_context() as (ctx, queue):
        # OpenCL-Programm (Box Blur)
        shader_path = os.path.join(os.path.dirname(__file__), "shaders", "box_blur.opencl")
        with open(shader_path, "r", encoding="utf-8") as f:
            program_src = f.read()

        with _suppress_output():
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
                        np.int32(channels), np.int32(radius))

        global_size = (width, height)
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
        cl.enqueue_copy(queue, output_np, output_buf)
        queue.finish()

        # In Bild zurückverwandeln
        output_img = output_np.reshape((height, width, channels))
        return Image.fromarray(output_img, 'RGB')

@timing
def blur_gaussian(img: Image.Image, kernel: int, sigma: float = 1.0, num_threads: int = 64) -> Image.Image: #globalignore
    """Optimized Gaussian blur using separable convolution with NumPy."""
    # Convert to numpy array for faster processing
    img_rgba = _ensure_rgba(img)
    img_array = np.array(img_rgba, dtype=np.float32)
    height, width, channels = img_array.shape
    
    # Generate 1D Gaussian kernel
    kernel_size = 2 * kernel + 1
    x = np.arange(-kernel, kernel + 1)
    gaussian_1d = np.exp(-x**2 / (2 * sigma**2))
    gaussian_1d /= gaussian_1d.sum()
    
    # Apply horizontal convolution
    temp = np.zeros_like(img_array)
    for c in range(channels):
        for y in range(height):
            for x in range(width):
                x_min = max(x - kernel, 0)
                x_max = min(x + kernel + 1, width)
                kernel_start = max(kernel - x, 0)
                kernel_end = min(kernel + (width - x), kernel_size)
                
                window = img_array[y, x_min:x_max, c]
                kernel_slice = gaussian_1d[kernel_start:kernel_end]
                temp[y, x, c] = np.sum(window * kernel_slice)
    
    # Apply vertical convolution
    output_array = np.zeros_like(img_array)
    for c in range(channels):
        for x in range(width):
            for y in range(height):
                y_min = max(y - kernel, 0)
                y_max = min(y + kernel + 1, height)
                kernel_start = max(kernel - y, 0)
                kernel_end = min(kernel + (height - y), kernel_size)
                
                window = temp[y_min:y_max, x, c]
                kernel_slice = gaussian_1d[kernel_start:kernel_end]
                output_array[y, x, c] = np.sum(window * kernel_slice)
    
    # Convert back to PIL Image
    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    return Image.fromarray(output_array, 'RGBA')

@timing
def blur_gaussian_fast(img: Image.Image, kernel: int, sigma: float = 1.0) -> Image.Image: #globalignore
    """Highly optimized Gaussian blur using vectorized NumPy operations."""
    img_rgba = _ensure_rgba(img)
    img_array = np.array(img_rgba, dtype=np.float32)
    
    # Generate 1D Gaussian kernel
    kernel_size = 2 * kernel + 1
    x = np.arange(-kernel, kernel + 1)
    gaussian_1d = np.exp(-x**2 / (2 * sigma**2))
    gaussian_1d /= gaussian_1d.sum()
    
    # Pad image for convolution
    pad_width = kernel
    padded = np.pad(img_array, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')
    
    # Apply horizontal convolution
    temp = np.zeros_like(padded)
    for y in range(padded.shape[0]):
        for x in range(pad_width, padded.shape[1] - pad_width):
            window = padded[y, x - pad_width:x + pad_width + 1, :]
            temp[y, x, :] = np.sum(window * gaussian_1d[:, np.newaxis], axis=0)
    
    # Apply vertical convolution
    output_padded = np.zeros_like(padded)
    for x in range(pad_width, padded.shape[1] - pad_width):
        for y in range(pad_width, padded.shape[0] - pad_width):
            window = temp[y - pad_width:y + pad_width + 1, x, :]
            output_padded[y, x, :] = np.sum(window * gaussian_1d[:, np.newaxis], axis=0)
    
    # Remove padding and convert back
    output_array = output_padded[pad_width:-pad_width, pad_width:-pad_width, :]
    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    return Image.fromarray(output_array, 'RGBA')

def __gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray: #globalignore
    """Erzeuge eine 1D-Gaussian-Kernel."""
    radius = kernel_size // 2
    ax = np.arange(-radius, radius + 1)
    kernel = np.exp(-(ax**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def __convolve_separable(img_array: np.ndarray, kernel: np.ndarray) -> np.ndarray: #globalignore
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

def __process_chunk(sub_img: np.ndarray, kernel: np.ndarray) -> np.ndarray: #globalignore
    """Blurt ein gepaddetes Chunk und entfernt die Padding-Ränder."""
    blurred = __convolve_separable(sub_img, kernel)
    return blurred

def __blur_gaussian_1d(img: Image.Image, kernel: int, sigma: float = None, num_processes: int = 4) -> Image.Image: #globalignore
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

def subtract_images(img1: Image.Image, img2: Image.Image) -> Image.Image:
    if img1.size != img2.size:
        raise ValueError("Images must be the same size")

    # Preserve original mode to convert result back at the end
    original_mode = img1.mode

    # Ensure both images are in the same mode to avoid mixed-type pixels
    img1_rgba = _ensure_rgba(img1)
    img2_rgba = _ensure_rgba(img2)

    out = Image.new("RGBA", img1_rgba.size)
    for x in tqdm(range(img1_rgba.size[0]), desc="Calculating image delta"):
        for y in range(img1_rgba.size[1]):
            p1 = img1_rgba.getpixel((x, y))
            p2 = img2_rgba.getpixel((x, y))
            # p1 and p2 are now 4-tuples (R,G,B,A)
            diff_rgb = tuple(max(0, int(a) - int(b)) for a, b in zip(p1[:3], p2[:3]))
            # Keep result fully opaque (or you could compute alpha difference if desired)
            out.putpixel((x, y), (diff_rgb[0], diff_rgb[1], diff_rgb[2], 255))

    # Convert back to the original mode for compatibility with callers
    if original_mode == "RGBA":
        return out
    else:
        return out.convert(original_mode)

def adjust_brightness(img: Image.Image, gamma: float) -> Image.Image: #globalignore
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

def __get_standard_deviation(p1, p2): #globalignore
    r1 ,g1, b1 = p1
    r2, g2, b2 = p2
    return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)

def __box_blur_np(arr, kernel): #globalignore
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

def __process_rows(y_start, y_end, padded_gray, padded_img, kernel, height, width): #globalignore
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

def kuwahara_wrapper(
        img: Image.Image,
        kernel: int,
        regions: int = 8,
        isAnisotropic: bool = False,
        stylePapari: bool = False,
        progress: Optional[callable] = None
        ) -> Image.Image:
    """Wrapper for Kuwahara that reports simple start/finish progress."""
    try:
        if progress:
            progress({'percent': 0, 'message': 'Starting Kuwahara'})
    except Exception:
        pass
    print(f"Running Kuwahara Filter (Kernel: {kernel}, Regions: {regions}, Anisotropic: {isAnisotropic}, Papari Style: {stylePapari})")
    if isAnisotropic:
        if stylePapari:
            out = anisotropic_kuwahara_papari_gpu(img, kernel, regions)
        else:
            out = anisotropic_kuwahara_gpu(img, kernel, regions)
    else:
        out = kuwahara_gpu(img, kernel)
    try:
        if progress:
            progress({'percent': 100, 'message': 'Kuwahara completed'})
    except Exception:
        pass
    return out

def kuwahara(img: Image.Image, kernel: int, num_threads: int = None, progress: Optional[callable] = None) -> Image.Image: #globalignore
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

        # collect results and report progress per-chunk
        for i, future in enumerate(futures):
            result_chunk = future.result()
            out[ranges[i][0]:ranges[i][1]] = result_chunk
            try:
                if progress:
                    pct = int(((i + 1) / len(futures)) * 100)
                    progress({'percent': pct, 'message': f'Kuwahara chunk {i+1}/{len(futures)}'})
            except Exception:
                pass

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def kuwahara_gpu(img: Image.Image, kernel_size: float) -> Image.Image: #globalignore
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

        import os
        cl_path = os.path.join(os.path.dirname(__file__), "shaders", "kuwahara_filter.opencl")
        with open(cl_path, "r", encoding="utf-8") as f:
            program_src = f.read()

        with opencl_context() as (ctx, queue):
            with _suppress_output():
                program = cl.Program(ctx, program_src).build()

            flat_input = img_np.flatten()
            output_np = np.empty_like(flat_input)

            mf = cl.mem_flags
            input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat_input)
            output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_np.nbytes)

            compute_kernel = program.kuwahara_filter
            compute_kernel.set_args(input_buf, output_buf,
                            np.int32(width), np.int32(height),
                            np.int32(channels), np.int32(kernel_size))

            global_size = (width, height)
            cl.enqueue_nd_range_kernel(queue, compute_kernel, global_size, None)
            cl.enqueue_copy(queue, output_np, output_buf)
            queue.finish()

            output_img = output_np.reshape((height, width, channels))
            output_img = np.clip(output_img, 0, 255).astype(np.uint8)
            # If the GPU kernel produced an all-black image (common failure mode
            # for some OpenCL drivers/kernels), fall back to the CPU implementation.
            try:
                if np.count_nonzero(output_img) == 0:
                    print("kuwahara_gpu: GPU produced all-black output, falling back to CPU implementation.")
                    return _kuwahara_cpu_fallback(img, kernel_size)
            except Exception:
                # If anything goes wrong with the sanity check, proceed to return
                # the GPU result to avoid masking other issues.
                pass
            return Image.fromarray(output_img, 'RGB')
    except Exception as e:
        print(f"OpenCL error: {str(e)}. Falling back to CPU implementation.")
        return _kuwahara_cpu_fallback(img, kernel_size)

def anisotropic_kuwahara_gpu(img: Image.Image, kernel_size: float, regions: int = 8) -> Image.Image: #globalignore
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

        import os
        cl_path = os.path.join(os.path.dirname(__file__), "shaders", "anisotropic_kuwahara.opencl")
        with open(cl_path, "r", encoding="utf-8") as f:
            program_src = f.read()

        with opencl_context() as (ctx, queue):
            with _suppress_output():
                program = cl.Program(ctx, program_src).build()

            flat_input = img_np.flatten()
            output_np = np.empty_like(flat_input)

            mf = cl.mem_flags
            input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat_input)
            output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_np.nbytes)

            compute_kernel = program.anisotropic_kuwahara
            compute_kernel.set_args(input_buf, output_buf,
                                    np.int32(width), np.int32(height),
                                    np.int32(channels), np.int32(kernel_size), np.int32(regions))

            global_size = (width, height)
            cl.enqueue_nd_range_kernel(queue, compute_kernel, global_size, None)
            cl.enqueue_copy(queue, output_np, output_buf)
            queue.finish()

            output_img = output_np.reshape((height, width, channels))
            output_img = np.clip(output_img, 0, 255).astype(np.uint8)
            # Sanity check: if GPU returned all zeros, fallback to CPU
            try:
                if np.count_nonzero(output_img) == 0:
                    print("anisotropic_kuwahara_gpu: GPU produced all-black output, falling back to CPU implementation.")
                    return _cpu_fallback(img, kernel_size, regions)
            except Exception:
                pass
            return Image.fromarray(output_img, 'RGB')
    except Exception as e:
        print(f"OpenCL error: {str(e)}. Falling back to CPU implementation.")
        return _cpu_fallback(img, kernel_size, regions)

def anisotropic_kuwahara_papari_gpu( #globalignore
    img: Image.Image,
    kernel_size: int | float,
    regions: int = 8,
    tensor_sigma: float = 2.0,
    anisotropy: float = 3.0,
    epsilon: float = 1e-3,
) -> Image.Image: #globalignore
    """
    Edge-preserving anisotropic Kuwahara (Papari-style), GPU (OpenCL) + CPU fallback.

    Parameters
    ----------
    img : PIL.Image
    kernel_size : int|float
        Neighborhood radius (kernel "radius"). Values ~3..8 are typical.
    regions : int
        Number of angular sectors (e.g. 6, 8, 12).
    tensor_sigma : float
        Gaussian smoothing (std) for structure tensor (stabilizes orientation).
    anisotropy : float
        Axis ratio for the elliptical weighting (>=1). Larger => stronger along-edge smoothing,
        less across-edge smoothing.
    epsilon : float
        Small positive number to stabilize divisions / eigen computations.

    Returns
    -------
    PIL.Image
    """

    try:
        import pyopencl as cl
        _HAS_PYOPENCL = True
    except Exception:
        _HAS_PYOPENCL = False

    radius = int(kernel_size)
    if radius < 1:
        return img.copy()

    # Try to get a GPU context
    ctx = None
    queue = None
    try:
        try:
            from gpu_context import opencl_context  # optional helper if the user has it
            ctx, queue = opencl_context()
        except Exception:
            if not _HAS_PYOPENCL:
                raise ImportError("pyopencl not available")
            # Create default context/queue
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")
            # pick first GPU device, else CPU
            dev = None
            for p in platforms:
                gpus = [d for d in p.get_devices() if d.type & cl.device_type.GPU]
                if gpus:
                    dev = gpus[0]; break
            if dev is None:
                # fallback to any device
                for p in platforms:
                    ds = p.get_devices()
                    if ds:
                        dev = ds[0]; break
            if dev is None:
                raise RuntimeError("No OpenCL devices found")
            ctx = cl.Context([dev])
            queue = cl.CommandQueue(ctx)
    except Exception as e:
        raise RuntimeError(f"[AKF] OpenCL unavailable ({e}).")

    # -------- GPU path --------
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    H, W, C = arr.shape
    gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.float32) / 255.0

    import os
    cl_path = os.path.join(os.path.dirname(__file__), "shaders", "papari_aniso_kuwahara.opencl")
    with open(cl_path, "r", encoding="utf-8") as f:
        prg_src = f.read()

    with _suppress_output():
        program = cl.Program(ctx, prg_src).build()

    mf = cl.mem_flags
    img_flat = arr.astype(np.float32).ravel()
    gray_flat = gray.astype(np.float32).ravel()
    out_flat = np.empty_like(img_flat)

    buf_img = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_flat)
    buf_gray = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gray_flat)
    buf_out = cl.Buffer(ctx, mf.WRITE_ONLY, out_flat.nbytes)

    kernel = program.papari_aniso_kuwahara
    kernel.set_args(
        buf_img, buf_gray, buf_out,
        np.int32(W), np.int32(H), np.int32(C),
        np.int32(radius), np.int32(regions),
        np.float32(tensor_sigma), np.float32(anisotropy), np.float32(epsilon)
    )

    global_size = (W, H)
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
    cl.enqueue_copy(queue, out_flat, buf_out)
    queue.finish()

    out = out_flat.reshape((H, W, C))
    out = np.clip(out, 0, 255).astype(np.uint8)
    # If GPU output is entirely black, that usually indicates a kernel or
    # transfer problem — fall back to the CPU implementation which is more
    # robust across environments.
    try:
        if np.count_nonzero(out) == 0:
            print("anisotropic_kuwahara_papari_gpu: GPU produced all-black output, falling back to CPU implementation.")
            return kuwahara(img, radius)
    except Exception:
        pass
    return Image.fromarray(out, mode='RGB')

def kuwahara_grays(img: Image.Image, kernel: int) -> Image.Image: #globalignore
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
        c = int(max_x * max_y / 500)
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

def mix_percent(a:Image.Image, b: Image.Image, p: int) -> Image.Image:
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
def mix_by_percent(img1: Image.Image, img2: Image.Image, mix_factor: float) -> Image.Image: #globalignore
    """Alias for mix_percent to match render pass name."""
    return mix_percent(img1, img2, int(mix_factor))

def blur(img: Image.Image, blur_type: str, blur_kernel: int) -> Image.Image: #globalignore
    """Alias for blur functions to match render pass name."""
    if      blur_type == "Box"       : return blur_box_gpu(  img, kernel=int(blur_kernel))
    elif    blur_type == "Gaussian"  : return blur_gaussian( img, kernel=int(blur_kernel))
    else                            : return blur_box_gpu(  img, kernel=int(blur_kernel))

def invert(img: Image.Image, invert_type: str, impact_factor: float = 1.0) -> Image.Image:
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
                elif invert_type == "Lum":
                    lum = int(0.299 * r + 0.587 * g + 0.114 * b)
                    inv_lum = 255 - lum
                    diff = inv_lum - lum
                    inv_r, inv_g, inv_b = r + diff, g + diff, b + diff
                else:
                    raise ValueError("Invalid invert_type. Use: 'RGB', 'R', 'G', 'B', 'Lum'")
                
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
    Alpha over operation for compositing two images with alpha transparency.
    
    Args:
        img1: Background image (will be placed underneath)
        img2: Foreground image (will be placed on top with alpha transparency)
    
    Returns:
        Composited image in RGBA mode
    
    Raises:
        ValueError: If inputs are invalid or None
        RuntimeError: If compositing fails
    """
    try:
        # Validate inputs
        if img1 is None or img2 is None:
            raise ValueError("Both images must be provided")
        
        # Ensure both images are in RGBA mode
        img1_rgba = _ensure_rgba(img1)
        img2_rgba = _ensure_rgba(img2)
        
        # Handle size mismatches
        if img1_rgba.size != img2_rgba.size:
            print(f"Image size mismatch: {img1_rgba.size} vs {img2_rgba.size}. Resizing foreground image.")
            img2_rgba = img2_rgba.resize(img1_rgba.size, Image.Resampling.LANCZOS)
        
        # Perform manual alpha over compositing
        result = Image.new("RGBA", img1_rgba.size)
        for y in range(result.size[1]):
            for x in range(result.size[0]):
                r, g, b, a = result.getpixel((x, y))
                if a == 0:
                    result.putpixel((x, y), (0, 0, 0, 0))
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Alpha over operation failed: {str(e)}")
    
def difference(img1: Image.Image, img2: Image.Image) -> Image.Image:
    if img1 is None or img2 is None:
        raise ValueError("Both images must be provided")
    
    img1_rgba = _ensure_rgba(img1)
    img2_rgba = _ensure_rgba(img2)

    if img1_rgba.size != img2_rgba.size:
        print(f"Image size mismatch: {img1_rgba.size} vs {img2_rgba.size}. Resizing second image.")
        img2_rgba = img2_rgba.resize(img1_rgba.size, Image.Resampling.LANCZOS)
    
    arr1 = np.array(img1_rgba, dtype=np.int16)
    arr2 = np.array(img2_rgba, dtype=np.int16)

    diff_rgb = np.abs(arr1[:, :, :3] - arr2[:, :, :3]).astype(np.uint8)
    alpha = np.full((arr1.shape[0], arr1.shape[1], 1), 255, dtype=np.uint8)  # opaque
    diff = np.concatenate([diff_rgb, alpha], axis=2)

    return Image.fromarray(diff, mode="RGBA")

def maxAdd(img1: Image.Image, img2: Image.Image) -> Image.Image:
    if img1 is None or img2 is None:
        raise ValueError("Both images must be provided")
    
    img1_rgba = _ensure_rgba(img1)
    img2_rgba = _ensure_rgba(img2)

    if img1_rgba.size != img2_rgba.size:
        print(f"Image size mismatch: {img1_rgba.size} vs {img2_rgba.size}. Resizing second image.")
        img2_rgba = img2_rgba.resize(img1_rgba.size, Image.Resampling.LANCZOS)
    
    arr1 = np.array(img1_rgba, dtype=np.int16)
    arr2 = np.array(img2_rgba, dtype=np.int16)

    diff_rgb = np.maximum(arr1[:, :, :3], arr2[:, :, :3]).astype(np.uint8)
    alpha = np.full((arr1.shape[0], arr1.shape[1], 1), 255, dtype=np.uint8)  # opaque
    diff = np.concatenate([diff_rgb, alpha], axis=2)

    return Image.fromarray(diff, mode="RGBA")

def multiply(img: Image.Image, factor: float, allowValueOverflow:bool = False) -> Image.Image:
    """
    Multiply image by a factor, optionally allowing value overflow (default: False).
    
    Args:
        img: Input image
        factor: Multiplication factor
        allowValueOverflow: If True, values > 255 are not wrapped around to 0 (default: False)
    Returns:
        Output image after multiplication
    """

    if img is None:
        raise ValueError("Image must be provided")
    
    img_rgba = _ensure_rgba(img)
    out = Image.new("RGBA", img_rgba.size)

    for x in range(img_rgba.size[0]):
        for y in range(img_rgba.size[1]):
            pixel = img_rgba.getpixel((x, y))
            new_r = int(pixel[0] * factor)
            new_g = int(pixel[1] * factor)
            new_b = int(pixel[2] * factor)

            if not allowValueOverflow:
                new_r = min(new_r, 255)
                new_g = min(new_g, 255)
                new_b = min(new_b, 255)
            else:
                new_r = new_r % 255
                new_g = new_g % 255
                new_b = new_b % 255
            out.putpixel((x, y), (new_r, new_g, new_b))
    
    return out

def lerp(img1: Image.Image, img2: Image.Image,  mask: Image.Image) -> Image.Image:
    if mask.mode != "L":
        mask = mask.convert("L")

    for x in range(img1.size[0]):
        for y in range(img1.size[1]):
            pixel1 = img1.getpixel((x, y))
            pixel2 = img2.getpixel((x, y))
            # mask is converted to 'L' (grayscale) earlier, so getpixel returns
            # an int. Use it directly to compute the interpolation weight.
            mv = mask.getpixel((x, y))
            mask_value = (mv[0] / 255.0) if isinstance(mv, tuple) else (mv / 255.0)

            new_r = int(pixel1[0] * (1 - mask_value) + pixel2[0] * mask_value)
            new_g = int(pixel1[1] * (1 - mask_value) + pixel2[1] * mask_value)
            new_b = int(pixel1[2] * (1 - mask_value) + pixel2[2] * mask_value)

            img1.putpixel((x, y), (new_r, new_g, new_b))

    return img1

def sharpen(img: Image.Image, strength: float, kernel_size: int) -> Image.Image:
    if strength < 0:
        raise ValueError("Strength must be non-negative")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer")

    # Ensure both images are RGB for shape consistency
    img_rgb = img.convert('RGB')
    blurred_rgb = blur_gaussian(img_rgb, kernel=kernel_size).convert('RGB')
    img_np = np.array(img_rgb, dtype=np.float32)
    blurred_np = np.array(blurred_rgb, dtype=np.float32)

    sharpened_np = img_np + strength * (img_np - blurred_np)
    sharpened_np = np.clip(sharpened_np, 0, 255).astype(np.uint8)

    return Image.fromarray(sharpened_np, mode='RGB')

def meanShiftCluster(
    img: Image.Image,
    spatial_radius: int = 5,
    color_radius: int = 10,
    max_iter: int = 100,
    flat_regions: bool = True
) -> Image.Image:
    if spatial_radius < 1:
        raise ValueError("Spatial radius must be positive")
    if color_radius < 1:
        raise ValueError("Color radius must be positive")
    if max_iter < 1:
        raise ValueError("Max iterations must be positive")
    img = img.convert("RGB")
    img_np = np.array(img, dtype=np.float32)
    height, width, channels = img_np.shape

    # First pass: compute the converged mode (mean) for each pixel
    modes = np.zeros((height, width, 3), dtype=np.float32)
    spatial_radius_sq = spatial_radius * spatial_radius
    color_radius_sq = color_radius * color_radius

    for y in tqdm(range(height), desc="Computing modes (mean-shift)"):
        for x in range(width):
            mean = img_np[y, x].copy()
            for iteration in range(max_iter):
                sum_color = np.zeros(3, dtype=np.float32)
                count = 0
                y_start = max(0, y - spatial_radius)
                y_end = min(height, y + spatial_radius + 1)
                x_start = max(0, x - spatial_radius)
                x_end = min(width, x + spatial_radius + 1)

                for ny in range(y_start, y_end):
                    for nx in range(x_start, x_end):
                        spatial_dist_sq = (ny - y) ** 2 + (nx - x) ** 2
                        if spatial_dist_sq > spatial_radius_sq:
                            continue
                        color_dist_sq = np.sum((img_np[ny, nx] - mean) ** 2)
                        if color_dist_sq <= color_radius_sq:
                            sum_color += img_np[ny, nx]
                            count += 1

                if count > 0:
                    new_mean = sum_color / count
                    if np.linalg.norm(new_mean - mean) < 1e-3:
                        break
                    mean = new_mean

            modes[y, x] = np.clip(mean, 0, 255)

    # If the caller wants the old behavior (gradients / per-pixel modes),
    # return the per-pixel converged mean directly.
    if not flat_regions:
        out_np = modes.astype(np.uint8)
        return Image.fromarray(out_np, mode='RGB')

    # Second pass: group similar modes into discrete clusters (so each region
    # gets a single representative color instead of smooth gradients). We
    # cluster based on color distance using color_radius as the merge radius.
    labels = -1 * np.ones((height, width), dtype=np.int32)
    centers: list[np.ndarray] = []
    next_label = 0

    for y in range(height):
        for x in range(width):
            mode = modes[y, x]
            assigned = False
            for i, center in enumerate(centers):
                if np.sum((mode - center) ** 2) <= color_radius_sq:
                    labels[y, x] = i
                    assigned = True
                    break
            if not assigned:
                centers.append(mode.copy())
                labels[y, x] = next_label
                next_label += 1

    # Compute average color for each label using the original image pixels
    avg_colors = np.zeros((next_label, 3), dtype=np.float32)
    counts = np.zeros((next_label,), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            lbl = labels[y, x]
            if lbl >= 0:

                avg_colors[lbl] += img_np[y, x]
                counts[lbl] += 1

    # Avoid division by zero (shouldn't happen, but be safe)
    for i in range(next_label):
        if counts[i] > 0:
            avg_colors[i] = avg_colors[i] / counts[i]
        else:
            avg_colors[i] = centers[i]

    # Build output image where each pixel receives its cluster average color
    out_np = np.zeros_like(img_np)
    for y in range(height):
        for x in range(width):
            lbl = labels[y, x]
            out_np[y, x] = np.clip(avg_colors[lbl], 0, 255)

    out_np = out_np.astype(np.uint8)
    return Image.fromarray(out_np, mode='RGB')

def meanShiftClusteringGPU(
        img: Image.Image,
        spatial_radius: int = 5,
        color_radius: int = 10,
        max_iter: int = 100
) -> Image.Image:
    if spatial_radius < 1:
        raise ValueError("Spatial radius must be positive")
    if color_radius < 1:
        raise ValueError("Color radius must be positive")
    if max_iter < 1:
        raise ValueError("Max iterations must be positive")

    try:
        from gpu_context import opencl_context
    except ImportError:
        print("\033[91mCould not import GPU context, using CPU fallback...\033[0m")
        return meanShiftCluster(img, spatial_radius, color_radius, max_iter)

    def _cpu_fallback(img: Image.Image, s_rad: int, c_rad: int, m_iter: int) -> Image.Image:
        """CPU fallback implementation when GPU/OpenCL is unavailable."""
        print("\033[91mGPU/OpenCL unavailable, using CPU fallback...\033[0m")
        return meanShiftCluster(img, s_rad, c_rad, m_iter)

    try:
        img = img.convert("RGB")
        img_np = np.array(img).astype(np.float32)
        height, width, channels = img_np.shape

        import os
        cl_path = os.path.join(os.path.dirname(__file__), "shaders", "mean_shift.opencl")
        with open(cl_path, "r", encoding="utf-8") as f:
            program_src = f.read()

        with opencl_context() as (ctx, queue):
            with _suppress_output():
                program = cl.Program(ctx, program_src).build()

            flat_input = img_np.flatten()
            output_np = np.empty_like(flat_input)

            mf = cl.mem_flags
            input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat_input)
            output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_np.nbytes)

            compute_kernel = program.mean_shift
            compute_kernel.set_args(input_buf, output_buf,
                            np.int32(width), np.int32(height),
                            np.int32(channels), np.int32(spatial_radius),
                            np.int32(color_radius), np.int32(max_iter))

            global_size = (width, height)
            cl.enqueue_nd_range_kernel(queue, compute_kernel, global_size, None)
            cl.enqueue_copy(queue, output_np, output_buf)
            queue.finish()

            output_img = output_np.reshape((height, width, channels))
            output_img = np.clip(output_img, 0, 255).astype(np.uint8)
            return Image.fromarray(output_img, 'RGB')
    except Exception as e:
        print(f"OpenCL error: {str(e)}. Falling back to CPU implementation.")
        return _cpu_fallback(img, spatial_radius, color_radius, max_iter)

class DitherMethods(Enum):
    NONE = 0
    FLOYD_STEINBERG = 1
    ATKINSON = 2

def dither(
    img: Image.Image,
    num_colors: int,
    method: DitherMethods | str,
    palette_selection: str = "median_cut",
    palette: Optional[list] = None,
    progress: Optional[callable] = None,
) -> Image.Image:
    """Apply dithering and reduce colors.

    New parameter `palette_selection` controls how the final palette is
    chosen when converting to `num_colors` colors:
      - "median_cut": use PIL's median cut quantizer (default, preserves
        previous behaviour)
      - "most_represented": choose the `num_colors` most frequent colors
        in the (error-diffused) image
      - "most_different": choose `num_colors` colors that are maximally
        different (farthest-point sampling on unique colors)
    """

    if not 2 < num_colors <= 256:
        raise ValueError("Number of colors must be at least 3 and at most 256")

    if method == DitherMethods.NONE:
        print("No dithering applied, no dither method specified; returning original image.")
        return img

    if isinstance(method, str):
        method = method.upper()
        if method.lower() == "floyd_steinberg":
            method = DitherMethods.FLOYD_STEINBERG
        elif method.lower() == "atkinson":
            method = DitherMethods.ATKINSON
        else:
            raise ValueError(f"Unknown dithering method: {method}")

    img = img.convert("RGB")
    img_np = np.array(img).astype(np.float32)

    # We'll perform error-diffusion after we compute/decide on the palette
    # (so palette-aware diffusion can map to the chosen colors). Build a
    # uint8 snapshot for palette selection steps below.
    img_uint8 = np.clip(img_np, 0, 255).astype(np.uint8)

    # Convert to PIL image (clipped) for potential fallback and for the
    # median-cut path below.
    result_img = Image.fromarray(img_uint8, 'RGB')

    # Helper: select the most represented colors from the image
    def _select_most_represented(pixels: np.ndarray, k: int) -> np.ndarray:
        # pixels: (H,W,3) uint8
        flat = pixels.reshape(-1, 3).astype(np.uint8)
        uniq, counts = np.unique(flat.reshape(-1, 3), axis=0, return_counts=True)
        idx = np.argsort(counts)[::-1]
        chosen = uniq[idx[:k]]
        return chosen.astype(np.uint8)

    # Helper: farthest-first (greedy) selection to maximize palette diversity
    def _select_most_different(pixels: np.ndarray, k: int) -> np.ndarray:
        flat = pixels.reshape(-1, 3).astype(np.float32)
        # unique colors to reduce work
        uniq = np.unique(flat.reshape(-1, 3), axis=0).astype(np.float32)
        if uniq.shape[0] <= k:
            return uniq.astype(np.uint8)

        # start with the color farthest from the mean (deterministic)
        mean = uniq.mean(axis=0)
        # use luminance-weighted distance so diversity prefers perceptual differences
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        d0 = np.sum(((uniq - mean[None, :]) * weights[None, :]) ** 2, axis=1)
        centers = [uniq[np.argmax(d0)]]

        # Greedy farthest point sampling
        for _ in range(1, k):
            # compute luminance-weighted squared distances to current centers
            diffs = (uniq[:, None, :] - np.array(centers)[None, :, :]) * weights[None, None, :]
            dists = np.sum(diffs ** 2, axis=2)
            min_d = dists.min(axis=1)
            next_idx = np.argmax(min_d)
            centers.append(uniq[next_idx])

        return np.array(centers, dtype=np.uint8)

    # Palette selection: determine centers (palette colors) unless we
    # will use PIL's median-cut quantizer which handles both palette and
    # optional dithering itself.
    ps = palette_selection.lower() if isinstance(palette_selection, str) else str(palette_selection)

    centers = None
    if palette is not None:
        # normalize palette into an (N,3) uint8 array
        def _hex_to_rgb(h: str) -> tuple:
            h = h.lstrip('#')
            if len(h) == 3:
                h = ''.join([c*2 for c in h])
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        parsed = []
        for p in palette:
            if isinstance(p, str):
                parsed.append(_hex_to_rgb(p))
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                parsed.append((int(p[0]), int(p[1]), int(p[2])))
            else:
                raise ValueError(f"Invalid palette entry: {p}")
        centers = np.array(parsed, dtype=np.uint8)
        # Truncate if too long; do not override requested num_colors
        if centers.shape[0] > num_colors:
            centers = centers[:num_colors]

    # If we are using median_cut, delegate to PIL (it supports dithering)
    if ps.lower() == "median_cut":
        try:
            dither_flag = Image.FLOYDSTEINBERG if method == DitherMethods.FLOYD_STEINBERG else Image.NONE
            qimg = result_img.quantize(colors=num_colors, method=Image.MEDIANCUT, dither=dither_flag)
            return qimg.convert('RGB')
        except Exception:
            return result_img

    # If centers still None, compute according to palette_selection
    img_uint8 = np.clip(img_np, 0, 255).astype(np.uint8)
    if centers is None:
        if ps.lower() in ("most_represented", "represented", "frequent"):
            centers = _select_most_represented(img_uint8, num_colors)
        elif ps.lower() in ("most_different", "different", "diverse", "farthest"):
            centers = _select_most_different(img_uint8, num_colors)
        else:
            raise ValueError(f"Unknown palette_selection: {palette_selection}")

    # Ensure centers has at most num_colors entries
    if centers is not None and centers.shape[0] > num_colors:
        centers = centers[:num_colors]

    # Now perform dithering. If using an error-diffusion method, run the
    # palette-aware diffusion which maps pixels to the nearest center
    # during scanning and diffuses the resulting error.
    if method == DitherMethods.FLOYD_STEINBERG:
        out_np = __dither_floyd_steinberg(img_np.copy(), num_colors, centers, progress=progress)
        return Image.fromarray(out_np, 'RGB')
    elif method == DitherMethods.ATKINSON:
        out_np = __dither_atkinson(img_np.copy(), num_colors, centers, progress=progress)
        return Image.fromarray(out_np, 'RGB')

    # Fallback: simple nearest-center mapping (no error diffusion)
    h, w = img_uint8.shape[:2]
    pixels = img_uint8.reshape(-1, 3).astype(np.int16)
    centers_i = centers.astype(np.int16)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    diffs = (pixels[:, None, :].astype(np.float32) - centers_i[None, :, :].astype(np.float32)) * weights[None, None, :]
    dists = np.sum(diffs ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    quant_pixels = centers_i[labels].astype(np.uint8).reshape((h, w, 3))

    return Image.fromarray(quant_pixels, 'RGB')

def __dither_floyd_steinberg(img_np: np.ndarray, num_colors: int, centers: Optional[np.ndarray] = None, progress: Optional[callable] = None) -> np.ndarray:
    height, width, channels = img_np.shape

    if centers is None:
        quantization_level = 256 // max(1, num_colors)
        for y in range(height):
            for x in range(width):
                old_pixel = img_np[y, x].copy()
                new_pixel = np.round(old_pixel / quantization_level) * quantization_level
                img_np[y, x] = new_pixel
                error = old_pixel - new_pixel

                if x + 1 < width:
                    img_np[y, x + 1] += error * 7 / 16
                if x - 1 >= 0 and y + 1 < height:
                    img_np[y + 1, x - 1] += error * 3 / 16
                if y + 1 < height:
                    img_np[y + 1, x] += error * 5 / 16
                if x + 1 < width and y + 1 < height:
                    img_np[y + 1, x + 1] += error * 1 / 16

            # report progress per processed row
            try:
                if progress:
                    pct = int(((y + 1) / height) * 100)
                    progress({'percent': pct, 'message': f'Dithering row {y+1}/{height}'})
            except Exception:
                pass

        return np.clip(img_np, 0, 255).astype(np.uint8)

    # Palette-aware diffusion: map pixels to nearest center using
    # luminance-weighted distance, then diffuse the error.
    centers_f = centers.astype(np.float32)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    img = img_np.copy()
    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x].copy()
            diffs = (old_pixel[None, :] - centers_f) * weights[None, :]
            dists = np.sum(diffs * diffs, axis=1)
            idx = int(np.argmin(dists))
            new_pixel = centers_f[idx]
            img[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < width:
                img[y, x + 1] += error * 7 / 16
            if x - 1 >= 0 and y + 1 < height:
                img[y + 1, x - 1] += error * 3 / 16
            if y + 1 < height:
                img[y + 1, x] += error * 5 / 16
            if x + 1 < width and y + 1 < height:
                img[y + 1, x + 1] += error * 1 / 16

        # progress per row for palette-aware diffusion
        try:
            if progress:
                pct = int(((y + 1) / height) * 100)
                progress({'percent': pct, 'message': f'Dithering row {y+1}/{height}'})
        except Exception:
            pass

    return np.clip(img, 0, 255).astype(np.uint8)

def __dither_atkinson(img_np: np.ndarray, num_colors: int, centers: Optional[np.ndarray] = None, progress: Optional[callable] = None) -> np.ndarray:
    height, width, _ = img_np.shape

    if centers is None:
        quantization_level = 256 // max(1, num_colors)
        for y in range(height):
            for x in range(width):
                old_pixel = img_np[y, x].copy()
                new_pixel = np.round(old_pixel / quantization_level) * quantization_level
                img_np[y, x] = new_pixel
                error = old_pixel - new_pixel

                if x + 1 < width:
                    img_np[y, x + 1] += error * 1 / 8
                if x + 2 < width:
                    img_np[y, x + 2] += error * 1 / 8
                if x - 1 >= 0 and y + 1 < height:
                    img_np[y + 1, x - 1] += error * 1 / 8
                if y + 1 < height:
                    img_np[y + 1, x] += error * 1 / 8
                if x + 1 < width and y + 1 < height:
                    img_np[y + 1, x + 1] += error * 1 / 8
                if y + 2 < height:
                    img_np[y + 2, x] += error * 1 / 8

            # report progress per processed row
            try:
                if progress:
                    pct = int(((y + 1) / height) * 100)
                    progress({'percent': pct, 'message': f'Dithering row {y+1}/{height}'})
            except Exception:
                pass

        return np.clip(img_np, 0, 255).astype(np.uint8)

    # Palette-aware Atkinson diffusion
    centers_f = centers.astype(np.float32)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    img = img_np.copy()
    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x].copy()
            diffs = (old_pixel[None, :] - centers_f) * weights[None, :]
            dists = np.sum(diffs * diffs, axis=1)
            idx = int(np.argmin(dists))
            new_pixel = centers_f[idx]
            img[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < width:
                img[y, x + 1] += error * 1 / 8
            if x + 2 < width:
                img[y, x + 2] += error * 1 / 8
            if x - 1 >= 0 and y + 1 < height:
                img[y + 1, x - 1] += error * 1 / 8
            if y + 1 < height:
                img[y + 1, x] += error * 1 / 8
            if x + 1 < width and y + 1 < height:
                img[y + 1, x + 1] += error * 1 / 8
            if y + 2 < height:
                img[y + 2, x] += error * 1 / 8

        # progress per row for palette-aware Atkinson diffusion
        try:
            if progress:
                pct = int(((y + 1) / height) * 100)
                progress({'percent': pct, 'message': f'Dithering row {y+1}/{height}'})
        except Exception:
            pass

    return np.clip(img, 0, 255).astype(np.uint8)

def quantize(
        img: Image.Image,
        num_colors: int = 16,
        progress: Optional[callable] = None,
    ) -> Image.Image:
    if not 2 < num_colors <= 256:
        raise ValueError("Number of colors must be at least 3 and at most 256")
    
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)

    # If there are already no more unique colors than requested, return copy
    uniq = np.unique(pixels.astype(np.uint8), axis=0)
    if uniq.shape[0] <= num_colors:
        return Image.fromarray(pixels.reshape((h, w, 3)).astype(np.uint8), "RGB")

    # K-Means++ initialization
    rng = np.random.default_rng(0)
    n = pixels.shape[0]
    centers = np.empty((num_colors, 3), dtype=np.float32)
    first_idx = rng.integers(0, n)
    centers[0] = pixels[first_idx]
    for k in range(1, num_colors):
        # distance from each pixel to nearest existing center
        dists = np.min(np.sum((pixels[:, None, :] - centers[:k][None, :, :]) ** 2, axis=2), axis=1)
        probs = dists / dists.sum()
        choose = rng.choice(n, p=probs)
        centers[k] = pixels[choose]

    # K-Means iterations (vectorized)
    max_iter = 30
    tol = 1e-2
    for it in range(max_iter):
        # assign
        dists = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # shape (N, K)
        labels = np.argmin(dists, axis=1)

        # recompute centers
        new_centers = np.zeros_like(centers)
        changed = False
        for k in range(num_colors):
            mask = labels == k
            if mask.any():
                new_centers[k] = pixels[mask].mean(axis=0)
            else:
                # reinitialize empty cluster to a random pixel
                new_centers[k] = pixels[rng.integers(0, n)]
        shift = np.linalg.norm(new_centers - centers, axis=1).sum()
        centers = new_centers
        # report KMeans progress
        try:
            if progress:
                pct = int(((it + 1) / max_iter) * 100)
                progress({'percent': pct, 'message': f'KMeans iter {it+1}/{max_iter}'})
        except Exception:
            pass

        if shift <= tol:
            break

    # Map pixels to palette and build output image
    dists = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    quant_pixels = centers[labels].astype(np.uint8).reshape((h, w, 3))

    return Image.fromarray(quant_pixels, "RGB")

def reduce_size(
    img: Image.Image,
    resolution: str | tuple[str, str] | int = "1080x1920",
    fit_mode: str = "fit",
    upscale: bool = False
) -> Image.Image:
    """
    Resize image to the given target resolution.

    Parameters:
    - img: PIL.Image
    - resolution: "WIDTHxHEIGHT" string or (width, height) tuple or width (height will be calculated to preserve aspect ratio)
    - fit_mode: "fit" (preserve aspect, fit inside), "crop" (fill and center-crop), or "stretch"
    - upscale: if False, do not enlarge images beyond their original size

    Returns:
    - resized PIL.Image
    """
    target_w = None
    target_h = None

    if isinstance(resolution, str):
        res_l = resolution.lower()
        # Accept float strings for single number (e.g., '1598.0')
        try:
            as_float = float(res_l)
            if as_float.is_integer():
                print("Single number resolution provided (float), preserving aspect ratio.")
                target_w = int(round(as_float))
            else:
                # Not an integer, treat as error
                raise ValueError
        except ValueError:
            # Not a single float, try WxH parsing
            try:
                # Accept float values in WxH (e.g., '1598.0x600.0')
                for delim in ("x", ",", " ", "*", "-"):
                    if delim in res_l:
                        parts = res_l.split(delim)
                        break
                else:
                    if "(" in res_l and ")" in res_l:
                        res_l = res_l.replace("(", "").replace(")", "")
                        parts = res_l.split(",")
                    else:
                        raise ValueError("Delimiter not found in resolution string.")

                if len(parts) != 2:
                    raise ValueError("Resolution string must contain exactly one width and one height.")
                # Accept float values, round to int
                target_w, target_h = (int(round(float(p))) for p in parts)
            except Exception as e:
                raise ValueError(f"Failed to parse resolution string '{resolution}'. Expected format 'WxH' or similar.") from e

    elif isinstance(resolution, tuple) and len(resolution) == 2:
        target_w, target_h = map(int, resolution)
        
    elif isinstance(resolution, int):
        target_w = resolution

    else:
        raise ValueError("resolution must be a 'WIDTHxHEIGHT' string, a (width, height) tuple, or an integer width.")
    
    # 2. Calculate missing dimension if aspect ratio must be preserved (only W provided)
    if target_w is not None and target_h is None:
        # This covers:
        # a) Input was an integer (handled in elif resolution is int)
        # b) Input was a digit string (handled in if res_l.isdigit())
        
        # Ensure we don't divide by zero if image width is zero (shouldn't happen but good practice)
        if img.width == 0:
            raise ValueError("Image width is zero, cannot calculate aspect ratio.")
            
        # Calculate height based on original aspect ratio
        calculated_h = int(target_w * img.height / img.width)
        target_h = calculated_h
        
    # Final check to ensure dimensions were determined
    if target_w is None or target_h is None:
        # This should only happen if the initial parsing failed catastrophically
        raise RuntimeError("Target resolution dimensions could not be determined.")

    if target_w <= 0 or target_h <= 0:
        raise ValueError("Target width and height must be positive integers")

    src_w, src_h = img.size

    # If no upscaling allowed clamp target to source
    if not upscale:
        target_w = min(target_w, src_w)
        target_h = min(target_h, src_h)

    resample = Image.Resampling.LANCZOS

    fit_mode = fit_mode.lower()
    if fit_mode == "stretch":
        return img.resize((target_w, target_h), resample)

    # compute scale factors
    scale_w = target_w / src_w
    scale_h = target_h / src_h

    if fit_mode == "fit":
        scale = min(scale_w, scale_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        return img.resize((new_w, new_h), resample)

    if fit_mode == "crop":
        # scale to cover target, then center-crop
        scale = max(scale_w, scale_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = img.resize((new_w, new_h), resample)

        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        return resized.crop((left, top, right, bottom))

    raise ValueError("fit_mode must be one of: 'fit', 'crop', 'stretch'")

def palette_pass(
    colors: list,
    swatch_size: int = 64,
    columns: int = 0,
    save_to_slots: bool = False,
    save_path: Optional[str] = None,
) -> Image.Image:
    """Create a palette image from a list of colors (hex or RGB tuples).

    If `save_to_slots` is True and `save_path` is provided, save the
    generated palette image to disk (e.g., into `saved/`). This function
    is a simple backend pass that UI code can call. It does not itself
    implement UI for editing the list; it only generates the image.
    """
    from palette_generator import generate_palette_image, save_palette

    # Normalize swatch_size arg
    if isinstance(swatch_size, int):
        sw = (swatch_size, swatch_size)
    else:
        sw = tuple(swatch_size)

    img = generate_palette_image(colors, swatch_size=sw, columns=columns)
    if save_to_slots and save_path:
        save_palette(colors, save_path, swatch_size=sw, columns=columns)
    return img

class edge_detection_types(Enum):
    SOBEL = 0
    CANNY = 1
    DOG = 2 

def find_edges(
    img: Image.Image,
    threshold: float = 0.1,
    low_threshold: Optional[float] = None,
    high_threshold: Optional[float] = None,
    type: edge_detection_types | str = edge_detection_types.SOBEL,
    progress: Optional[callable] = None,
) -> Image.Image:
    """
    Detect edges in the image using one of the following methods:
    - Sobel
    - Canny
    - Difference of Gaussians (DoG)

    Args:
        img: Input PIL Image.
        threshold: Base threshold used by some detectors (default 0.1).
            - Range: 0.0 .. 1.0 (float). Values outside this range will be clipped
              by internal logic when used for 0..255 scaling for Canny.
        low_threshold: Low hysteresis threshold for Canny (optional).
            - Range: 0.0 .. 255.0 (float) or None. If None, computed as threshold * 255.
        high_threshold: High hysteresis threshold for Canny (optional).
            - Range: 0.0 .. 255.0 (float) or None. If None, computed as min(255, low_threshold * 2).
        type: Edge detection method to use. Accepts one of:
            - edge_detection_types.SOBEL
            - edge_detection_types.CANNY
            - edge_detection_types.DOG
            (or the equivalent enum value). Default should be provided by caller.
    Returns:
        PIL.Image: Output image with edges detected (mode 'L' grayscale).
    Notes:
        - All numeric ranges above are the valid/expected ranges. Inputs will be
          clipped/coerced where appropriate by the implementation.
        - For Canny, `threshold` is interpreted in 0..1 and scaled to 0..255 if low/high
          are not supplied.
    """
    if img is None:
        raise ValueError("Image must be provided")
    
    if isinstance(type, str):
        type = type.upper()
        if type == "SOBEL":
            type = edge_detection_types.SOBEL
        elif type == "CANNY":
            type = edge_detection_types.CANNY
        elif type == "DOG":
            type = edge_detection_types.DOG
        else:
            raise ValueError(f"Unknown edge detection type: {type}")
    
    img = img.convert("L")  # Convert to grayscale
    img_np = np.array(img, dtype=np.float32)

    # Helper: 2D convolution with small kernel using numpy (reflect padding)
    def _convolve2d(img_arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(img_arr, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        out = np.empty_like(img_arr, dtype=np.float32)
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                window = padded[i:i + kh, j:j + kw]
                out[i, j] = np.sum(window * kernel)
        return out

    # Helper: 1D Gaussian kernel
    def _gaussian_kernel1d(radius: int, sigma: float) -> np.ndarray:
        if radius <= 0:
            return np.array([1.0], dtype=np.float32)
        ax = np.arange(-radius, radius + 1, dtype=np.float32)
        kern = np.exp(-(ax * ax) / (2.0 * sigma * sigma))
        kern /= kern.sum()
        return kern

    # Separable convolution using 1D kernel
    def _separable_conv(img_arr: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
        radius = len(kernel_1d) // 2
        # horizontal pass
        padded_h = np.pad(img_arr, ((0, 0), (radius, radius)), mode='reflect')
        temp = np.empty_like(img_arr, dtype=np.float32)
        for y in range(img_arr.shape[0]):
            for x in range(img_arr.shape[1]):
                temp[y, x] = np.sum(padded_h[y, x:x + 2 * radius + 1] * kernel_1d)
        # vertical pass
        padded_v = np.pad(temp, ((radius, radius), (0, 0)), mode='reflect')
        out = np.empty_like(img_arr, dtype=np.float32)
        for y in range(img_arr.shape[0]):
            for x in range(img_arr.shape[1]):
                out[y, x] = np.sum(padded_v[y:y + 2 * radius + 1, x] * kernel_1d)
        return out

    if progress:
        try:
            progress({'percent': 0, 'message': 'Starting edge detection'})
        except Exception:
            pass

    if type == edge_detection_types.SOBEL:
        # Sobel kernels
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        gx = _convolve2d(img_np, kx)
        gy = _convolve2d(img_np, ky)
        edges = np.hypot(gx, gy)
        maxv = edges.max() if edges.max() != 0 else 1.0
        edges = (edges / maxv * 255.0).astype(np.uint8)
        if progress:
            try:
                progress({'percent': 100, 'message': 'Sobel edge detection complete'})
            except Exception:
                pass
        return Image.fromarray(edges, mode='L')

    elif type == edge_detection_types.CANNY:
        # Basic Canny-like pipeline implemented with numpy:
        # 1) Gaussian smoothing, 2) gradient (Sobel), 3) non-maximum suppression,
        # 4) hysteresis thresholding.
        sigma = 1.0
        radius = max(1, int(3.0 * sigma))
        gk = _gaussian_kernel1d(radius, sigma)
        smooth = _separable_conv(img_np, gk)
        if progress:
            try:
                progress({'percent': 10, 'message': 'Smoothed image'})
            except Exception:
                pass

        # gradients
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        gx = _convolve2d(smooth, kx)
        gy = _convolve2d(smooth, ky)
        if progress:
            try:
                progress({'percent': 30, 'message': 'Computed gradients'})
            except Exception:
                pass
        mag = np.hypot(gx, gy)
        # Normalize magnitude to 0..255 for thresholding
        mag_max = mag.max() if mag.max() != 0 else 1.0
        mag_n = (mag / mag_max) * 255.0

        # angles in degrees 0..180
        ang = (np.arctan2(gy, gx) * (180.0 / math.pi)) % 180.0

        # Non-maximum suppression (quantize directions to 4 sectors)
        H, W = mag.shape
        nms = np.zeros_like(mag_n, dtype=np.float32)
        for y in range(1, H - 1):
            # report NMS progress periodically
            if progress and (y % max(1, (H // 25)) == 0):
                try:
                    pct = 30 + int(((y - 1) / max(1, (H - 2))) * 50)
                    progress({'percent': pct, 'message': f'NMS {y}/{H}'})
                except Exception:
                    pass
            for x in range(1, W - 1):
                a = ang[y, x]
                m = mag_n[y, x]
                # determine neighbors to compare
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    n1 = mag_n[y, x - 1]; n2 = mag_n[y, x + 1]
                elif 22.5 <= a < 67.5:
                    n1 = mag_n[y - 1, x + 1]; n2 = mag_n[y + 1, x - 1]
                elif 67.5 <= a < 112.5:
                    n1 = mag_n[y - 1, x]; n2 = mag_n[y + 1, x]
                else:  # 112.5 .. 157.5
                    n1 = mag_n[y - 1, x - 1]; n2 = mag_n[y + 1, x + 1]

                if m >= n1 and m >= n2:
                    nms[y, x] = m
                else:
                    nms[y, x] = 0.0

        # hysteresis thresholds
        if low_threshold is None or high_threshold is None:
            low_threshold = threshold * 255.0
            high_threshold = min(255.0, low_threshold * 2.0)
        low_t = float(np.clip(low_threshold, 0.0, 255.0))
        high_t = float(np.clip(high_threshold, 0.0, 255.0))

        strong = (nms >= high_t)
        weak = ((nms >= low_t) & (nms < high_t))

        edges_bool = np.zeros_like(nms, dtype=bool)
        # initialize with strong edges
        sy, sx = np.where(strong)
        stack = list(zip(sy.tolist(), sx.tolist()))
        edges_bool[strong] = True

        # 8-connected hysteresis: promote weak neighbors connected to strong
        while stack:
            cy, cx = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= H or nx < 0 or nx >= W:
                        continue
                    if weak[ny, nx] and not edges_bool[ny, nx]:
                        edges_bool[ny, nx] = True
                        stack.append((ny, nx))

        if progress:
            try:
                progress({'percent': 85, 'message': 'NMS complete, performing hysteresis'})
            except Exception:
                pass

        edges_uint8 = (edges_bool.astype(np.uint8) * 255)
        if progress:
            try:
                progress({'percent': 100, 'message': 'Canny edge detection complete'})
            except Exception:

                pass

        return Image.fromarray(edges_uint8, mode='L')

    elif type == edge_detection_types.DOG:
        # Difference of Gaussians: blur with two sigmas and subtract
        sigma1 = 1.0
        sigma2 = 2.0
        r1 = max(1, int(3.0 * sigma1))
        r2 = max(1, int(3.0 * sigma2))
        k1 = _gaussian_kernel1d(r1, sigma1)
        k2 = _gaussian_kernel1d(r2, sigma2)
        b1 = _separable_conv(img_np, k1)
        b2 = _separable_conv(img_np, k2)
        dog = b1 - b2
        m = np.abs(dog).max() if np.abs(dog).max() != 0 else 1.0
        edges = (np.abs(dog) / m * 255.0).astype(np.uint8)
        return Image.fromarray(edges, mode='L')

    else:
        raise ValueError(f"Unknown edge detection type: {type}")