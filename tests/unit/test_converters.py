"""Unit tests for converters.py module."""
import unittest
import sys
import os

# Add the scripts directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import scripts.converters as converters
from tests.base_test import BaseTestCase


class TestConverters(BaseTestCase):
    """Test suite for color space conversion functions."""
    
    def test_get_luminance_rgb(self):
        """Test luminance calculation for RGB values."""
        # Test pure colors
        self.assertAlmostEqual(converters.get_luminance((255, 0, 0)), 54.213, places=1)
        self.assertAlmostEqual(converters.get_luminance((0, 255, 0)), 182.376, places=1)
        self.assertAlmostEqual(converters.get_luminance((0, 0, 255)), 18.425, places=1)
        
        # Test grayscale
        self.assertAlmostEqual(converters.get_luminance((128, 128, 128)), 128.0, places=1)
        self.assertAlmostEqual(converters.get_luminance((0, 0, 0)), 0.0, places=1)
        self.assertAlmostEqual(converters.get_luminance((255, 255, 255)), 255.0, places=1)
    
    def test_get_hue(self):
        """Test hue calculation for RGB values."""
        # Test pure colors
        self.assertEqual(converters.get_hue((255, 0, 0)), 0)  # Red
        self.assertEqual(converters.get_hue((0, 255, 0)), 120)  # Green
        self.assertEqual(converters.get_hue((0, 0, 255)), 240)  # Blue
        
        # Test edge cases
        hue = converters.get_hue((128, 64, 200))
        self.assertGreaterEqual(hue, 0)
        self.assertLessEqual(hue, 360)
    
    def test_get_rgb_components(self):
        """Test RGB component extraction."""
        # Test red component
        self.assertEqual(converters.get_r((255, 100, 50)), 255)
        self.assertEqual(converters.get_r((0, 200, 100)), 0)
        
        # Test green component
        self.assertEqual(converters.get_g((100, 255, 50)), 255)
        self.assertEqual(converters.get_g((200, 0, 100)), 0)
        
        # Test blue component
        self.assertEqual(converters.get_b((50, 100, 255)), 255)
        self.assertEqual(converters.get_b((100, 200, 0)), 0)
    
    def test_convert_function(self):
        """Test the main convert function with different modes."""
        pixel = (100, 150, 200)
        
        # Test all supported modes
        result = converters.convert(pixel, "lum")
        self.assertIsInstance(result, (int, float))
        
        result = converters.convert(pixel, "hue")
        self.assertIsInstance(result, int)
        
        self.assertEqual(converters.convert(pixel, "r"), 100)
        self.assertEqual(converters.convert(pixel, "g"), 150)
        self.assertEqual(converters.convert(pixel, "b"), 200)
    
    def test_convert_invalid_mode(self):
        """Test convert function with invalid mode."""
        with self.assertRaises(Exception):
            converters.convert((100, 150, 200), "invalid_mode")
    
    def test_get_luminance_invalid_input(self):
        """Test get_luminance with invalid input."""
        with self.assertRaises(Exception):
            converters.get_luminance((100, 150))  # Missing blue component
    
    def test_rotate_coords(self):
        """Test coordinate rotation functions."""
        coords = [(0, 0), (10, 5), (20, 15)]
        img_size = (100, 50)
        
        # Test 90 degree rotation
        rotated_90 = converters.rotate_coords(coords, img_size, 90)
        expected_90 = [(0, 99), (5, 89), (15, 79)]
        self.assertEqual(rotated_90, expected_90)
        
        # Test -90 degree rotation
        rotated_neg90 = converters.rotate_coords(coords, img_size, -90)
        expected_neg90 = [(49, 0), (44, 10), (34, 20)]
        self.assertEqual(rotated_neg90, expected_neg90)
        
        # Test 0 degree rotation (no change)
        rotated_0 = converters.rotate_coords(coords, img_size, 0)
        self.assertEqual(rotated_0, coords)
