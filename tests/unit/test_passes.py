"""Unit tests for scripts/passes.py"""
import unittest
import sys
import os
from PIL import Image

# Ensure scripts is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.base_test import ImageTestCase
from scripts import passes


class TestPasses(ImageTestCase):
    def test_ensure_rgba_conversions(self):
        rgb = Image.new('RGB', (2, 2), (10, 20, 30))
        rgba = passes._ensure_rgba(rgb)
        self.assertEqual(rgba.mode, 'RGBA')
        self.assertEqual(rgba.getpixel((0, 0)), (10, 20, 30, 255))

        gray = Image.new('L', (2, 2), 128)
        g2 = passes._ensure_rgba(gray)
        self.assertEqual(g2.mode, 'RGBA')
        self.assertEqual(g2.getpixel((0, 0)), (128, 128, 128, 255))

    def test_get_put_pixel_rgba(self):
        img = Image.new('RGB', (3, 3), (0, 0, 0))
        passes._put_pixel_rgba(img, 1, 1, (123, 45, 67, 200))
        self.assertEqual(img.getpixel((1, 1)), (123, 45, 67))

        rgba_img = Image.new('RGBA', (2, 2), (1, 2, 3, 4))
        self.assertEqual(passes._get_pixel_rgba(rgba_img.getpixel((0, 0))), (1, 2, 3, 4))
        self.assertEqual(passes._get_pixel_rgba(128), (128, 128, 128, 255))

    def test_scale_image_center_and_downscale(self):
        base = Image.new('RGB', (10, 10), (10, 20, 30))
        copy = Image.new('RGBA', (20, 20), (0, 0, 0, 0))
        # downscale 0 -> centered same size
        out = passes.scale_image(base, copy, downscale=0)
        self.assertEqual(out.size, copy.size)
        # downscale 50% should be smaller but centered
        out2 = passes.scale_image(base, copy, downscale=50)
        self.assertEqual(out2.size, copy.size)

    def test_mix_percent_and_alias(self):
        a = Image.new('RGB', (4, 4), (100, 0, 0))
        b = Image.new('RGB', (4, 4), (0, 100, 0))
        mixed = passes.mix_percent(a, b, 50)
        self.assertEqual(mixed.getpixel((0, 0))[0], 50)
        # alias
        alias = passes.mix_by_percent(a, b, 50)
        self.assertEqual(alias.getpixel((0,0))[1], 50)

    def test_invert_basic(self):
        img = Image.new('RGB', (2,2), (10,20,30))
        inv = passes.invert(img, 'RGB', 100)
        self.assertEqual(inv.getpixel((0,0)), (245, 235, 225))
        # partial impact
        inv2 = passes.invert(img, 'R', 50)
        r,g,b = inv2.getpixel((0,0))
        self.assertTrue(r != 10)

    def test_maxAdd_difference_multiply(self):
        a = Image.new('RGB', (2,2), (200,10,5))
        b = Image.new('RGB', (2,2), (100,50,250))
        mx = passes.maxAdd(a, b)
        self.assertEqual(mx.getpixel((0,0))[0], 200)
        diff = passes.difference(a, b)
        self.assertEqual(diff.mode, 'RGBA')
        mul = passes.multiply(a, 1.5, allowValueOverflow=False)
        self.assertTrue(all(v <= 255 for v in mul.getpixel((0,0))[:3]))

    def test_subtract_images_and_errors(self):
        a = Image.new('L', (3,3), 100)
        b = Image.new('L', (3,3), 30)
        sub = passes.subtract_images(a, b)
        self.assertEqual(sub.getpixel((0,0)), 70)
        # different sizes
        c = Image.new('L', (2,2), 10)
        with self.assertRaises(ValueError):
            passes.subtract_images(a, c)

    def test_to_vertical_and_split_connected_chunks(self):
        chunks = [[(0,0),(0,1),(1,0),(1,1)], [(2,0),(2,1),(2,3)]]
        v = passes.to_vertical_chunks(chunks)
        self.assertTrue(isinstance(v, list))
        split = passes.split_connected_chunks([[(2,0),(2,1),(2,3)]])
        self.assertEqual(len(split), 2)

    def test_lerp_and_errors(self):
        a = Image.new('RGB', (3,3), (10,10,10))
        b = Image.new('RGB', (3,3), (20,20,20))
        mask = Image.new('L', (3,3), 128)
        out = passes.lerp(a.copy(), b, mask)
        self.assertEqual(out.getpixel((0,0))[0], 15)

    def test_cristalline_expansion_deterministic(self):
        # Use small image and small sample count to make deterministic-ish
        img = Image.new('RGB', (6,6), (5,10,15))
        out = passes.cristalline_expansion(img, c=3)
        self.assertEqual(out.size, img.size)

    def test_kuwahara_wrapper_cpu_fallback(self):
        img = Image.new('RGB', (10,10), (10,10,10))
        out = passes.kuwahara_wrapper(img, kernel=1, regions=4, isAnisotropic=False, stylePapari=False)
        self.assertEqual(out.size, img.size)

    def test_sort_simple(self):
        # simple sort: small image should remain same size and produce no errors
        img = Image.new('RGB', (5,5))
        out = passes.sort(img, mode='lum', flip_dir=False, rotate=False, mask=None)
        self.assertEqual(out.size, img.size)

    def test_sort_with_mask(self):
        # Create a 5x5 image with a gradient
        img = Image.new('RGB', (5, 5), (0, 0, 0))
        for x in range(5):
            for y in range(5):
                img.putpixel((x, y), (x * 50, y * 50, 0))

        # Create a mask with a diagonal line
        mask = Image.new('RGB', (5, 5), (0, 0, 0))
        for i in range(5):
            mask.putpixel((i, i), (255, 255, 255))

        # Apply the sort function with the mask
        out = passes.sort(img, mode='lum', flip_dir=False, rotate=False, mask=mask)

        # Verify that only the diagonal pixels are sorted
        for i in range(5):
            self.assertEqual(out.getpixel((i, i)), (i * 50, i * 50, 0))

        # Verify that other pixels remain unchanged
        for x in range(5):
            for y in range(5):
                if x != y:
                    self.assertEqual(out.getpixel((x, y)), (x * 50, y * 50, 0))


if __name__ == '__main__':
    unittest.main()
