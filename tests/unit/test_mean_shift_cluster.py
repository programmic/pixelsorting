"""Unit tests for meanShiftCluster behavior (flat regions toggle)."""
import unittest
import sys
import os
from PIL import Image

# Ensure scripts is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.base_test import ImageTestCase
from scripts import passes


class TestMeanShiftCluster(ImageTestCase):
    def test_flat_regions_reduces_unique_colors(self):
        # Create a 1x10 gradient image
        img = Image.new('RGB', (10, 1))
        for x in range(10):
            v = int(x * 25)
            img.putpixel((x, 0), (v, v, v))

        # Run with gradient mode (old behavior)
        out_grad = passes.meanShiftCluster(img, spatial_radius=2, color_radius=15, max_iter=10, flat_regions=False)
        # Run with flat regions enabled (new behavior)
        out_flat = passes.meanShiftCluster(img, spatial_radius=2, color_radius=50, max_iter=10, flat_regions=True)

        # Count unique colors
        def unique_colors(image):
            pixels = [image.getpixel((x, 0)) for x in range(image.size[0])]
            return len(set(pixels))

        unique_grad = unique_colors(out_grad)
        unique_flat = unique_colors(out_flat)

        # Flat regions should not increase the number of unique colors and
        # generally will reduce it when clustering radius is larger.
        self.assertLessEqual(unique_flat, unique_grad)
        self.assertTrue(unique_flat >= 1)

    def test_uniform_regions_stay_uniform(self):
        # Left half one color, right half another
        img = Image.new('RGB', (6, 1))
        for x in range(6):
            if x < 3:
                img.putpixel((x, 0), (10, 10, 10))
            else:
                img.putpixel((x, 0), (200, 200, 200))

        out_flat = passes.meanShiftCluster(img, spatial_radius=1, color_radius=10, max_iter=20, flat_regions=True)

        # All pixels in left half should be identical; same for right half
        left_colors = {out_flat.getpixel((x, 0)) for x in range(3)}
        right_colors = {out_flat.getpixel((x, 0)) for x in range(3, 6)}
        self.assertEqual(len(left_colors), 1)
        self.assertEqual(len(right_colors), 1)


if __name__ == '__main__':
    unittest.main()
