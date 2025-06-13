# passes.py

from PIL import Image
import converters, math, random, threading
from tqdm import tqdm
from collections import defaultdict


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