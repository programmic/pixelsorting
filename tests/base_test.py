"""Base test classes and utilities for unittest framework."""
import unittest
import tempfile
import os
from PIL import Image
import numpy as np


class BaseTestCase(unittest.TestCase):
    """Base test case class with common test fixtures."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        super().setUpClass()
        
    def setUp(self):
        """Set up test fixtures before each test method."""
        super().setUp()
        self._temp_dir = None
        
    def tearDown(self):
        """Clean up after each test method."""
        super().tearDown()
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
    
    @property
    def temp_directory(self):
        """Get a temporary directory for testing."""
        if not self._temp_dir:
            self._temp_dir = tempfile.mkdtemp()
        return self._temp_dir
    
    def create_sample_rgb_image(self, width=100, height=100, color='red'):
        """Create a sample RGB image for testing."""
        return Image.new('RGB', (width, height), color=color)
    
    def create_sample_rgba_image(self, width=100, height=100, color=(255, 0, 0, 255)):
        """Create a sample RGBA image for testing."""
        return Image.new('RGBA', (width, height), color=color)
    
    def create_sample_grayscale_image(self, width=100, height=100, color=128):
        """Create a sample grayscale image for testing."""
        return Image.new('L', (width, height), color=color)
    
    def create_sample_image_with_pattern(self, width=50, height=50):
        """Create a sample image with a specific pattern for testing."""
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        for i in range(width):
            for j in range(height):
                pixels[i, j] = (i * 5, j * 5, (i + j) * 2)
        return img
    
    def create_sample_numpy_array(self, shape=(100, 100, 3), dtype=np.uint8):
        """Create a sample numpy array for testing."""
        return np.random.randint(0, 256, shape, dtype=dtype)


class ImageTestCase(BaseTestCase):
    """Test case class specifically for image-related tests."""
    
    def setUp(self):
        """Set up image test fixtures."""
        super().setUp()
        self.sample_rgb_image = self.create_sample_rgb_image()
        self.sample_rgba_image = self.create_sample_rgba_image()
        self.sample_grayscale_image = self.create_sample_grayscale_image()
        self.sample_image_with_pattern = self.create_sample_image_with_pattern()
        self.sample_numpy_array = self.create_sample_numpy_array()
