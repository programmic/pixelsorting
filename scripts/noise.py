"""Library of noise generation functions for pixel sorting."""

import numpy as np

def __gaussian_noise(img_arr: np.ndarray, amount: float) -> np.ndarray:
    mean = 0.0
    sigma = amount * 255.0
    gauss = np.random.normal(mean, sigma, img_arr.shape).astype(np.float32)
    noisy = img_arr + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def __salt_and_pepper_noise(img_arr: np.ndarray, amount: float) -> np.ndarray:
    noisy = img_arr.copy()
    num_salt = np.ceil(amount * img_arr.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * img_arr.size * 0.5).astype(int)

    # Salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in img_arr.shape]
    noisy[tuple(coords)] = 255

    # Pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img_arr.shape]
    noisy[tuple(coords)] = 0

    return noisy.astype(np.uint8)

def __shot_noise(img_arr: np.ndarray, amount: float) -> np.ndarray:
    noisy = img_arr.copy()
    vals = 255 * np.random.poisson(img_arr / 255.0 * amount)
    noisy = noisy + vals
    return np.clip(noisy, 0, 255).astype(np.uint8)

def __uniform_noise(img_arr: np.ndarray, amount: float) -> np.ndarray:
    scale = amount * 255.0
    uniform = np.random.uniform(-scale, scale, img_arr.shape).astype(np.float32)
    noisy = img_arr + uniform
    return np.clip(noisy, 0, 255).astype(np.uint8)

def __fifty_percent_noise(img_arr: np.ndarray, amount: float) -> np.ndarray:
    noisy = img_arr.copy()
    prob = amount
    random_matrix = np.random.rand(*img_arr.shape)
    noisy[random_matrix < (prob / 2)] = 0
    noisy[random_matrix > (1 - prob / 2)] = 255
    return noisy.astype(np.uint8)

def __blue_noise(img_arr: np.ndarray, amount: float) -> np.ndarray:
    scale = amount * 255.0
    blue = np.random.normal(0, scale, img_arr.shape).astype(np.float32)
    noisy = img_arr + blue
    return np.clip(noisy, 0, 255).astype(np.uint8)

def __perlin_noise(img_arr: np.ndarray, amount: float, size: float) -> np.ndarray:
    # generate Perlin noise without external libraries
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    def gradient(h, x, y):
        vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
        g = vectors[h%4]
        return g[:, :, 0]*x + g[:, :, 1]*y
    def perlin(x, y):
        xi = x.astype(int)
        yi = y.astype(int)
        xf = x - xi
        yf = y - yi
        u = f(xf)
        v = f(yf)

        n00 = gradient(p[p[xi]+yi], xf, yf)
        n01 = gradient(p[p[xi]+yi+1], xf, yf-1)
        n10 = gradient(p[p[xi+1]+yi], xf-1, yf)
        n11 = gradient(p[p[xi+1]+yi+1], xf-1, yf-1)

        x1 = (1 - u) * n00 + u * n10
        x2 = (1 - u) * n01 + u * n11
        return (1 - v) * x1 + v * x2
    lin_x = np.linspace(0, size, img_arr.shape[1], endpoint=False)
    lin_y = np.linspace(0, size, img_arr.shape[0], endpoint=False)
    x, y = np.meshgrid(lin_x, lin_y)
    permutation = np.arange(256, dtype=int)
    np.random.shuffle(permutation)
    p = np.stack([permutation, permutation]).flatten()
    noise = perlin(x, y)
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255.0
    noisy = img_arr + (noise * amount).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)
    
def __blue_noise(img_arr: np.ndarray, amount: float) -> np.ndarray:
    scale = amount * 255.0
    blue = np.random.normal(0, scale, img_arr.shape).astype(np.float32)
    noisy = img_arr + blue
    return np.clip(noisy, 0, 255).astype(np.uint8)

def __pink_noise(img_arr: np.ndarray, amount: float) -> np.ndarray:
    scale = amount * 255.0
    pink = np.cumsum(np.random.normal(0, scale, img_arr.shape), axis=0)
    noisy = img_arr + pink
    return np.clip(noisy, 0, 255).astype(np.uint8)

def __white_noise(img_arr: np.ndarray, amount: float) -> np.ndarray:
    scale = amount * 255.0
    white = np.random.normal(0, scale, img_arr.shape).astype(np.float32)
    noisy = img_arr + white
    return np.clip(noisy, 0, 255).astype(np.uint8)
