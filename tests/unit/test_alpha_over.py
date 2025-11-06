#!/usr/bin/env python3
"""
Test script to verify the alpha_over function fix
"""

from PIL import Image
import numpy as np
from passes import alpha_over

def create_test_images():
    """Create test images for alpha over testing"""
    # Create a simple background image (red)
    bg = Image.new('RGBA', (100, 100), (255, 0, 0, 255))
    
    # Create a foreground image with transparency (green circle)
    fg = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    pixels = fg.load()
    for x in range(100):
        for y in range(100):
            if (x-50)**2 + (y-50)**2 < 400:  # Circle with radius 20
                pixels[x, y] = (0, 255, 0, 128)  # Semi-transparent green
    
    return bg, fg

def test_alpha_over_basic():
    """Test basic alpha over functionality"""
    print("Testing basic alpha over functionality...")
    
    bg, fg = create_test_images()
    
    try:
        # Test the fixed alpha_over function
        result = alpha_over(bg, fg)
        
        # Verify the result
        assert result.mode == 'RGBA', f"Expected RGBA mode, got {result.mode}"
        assert result.size == bg.size, f"Size mismatch: {result.size} vs {bg.size}"
        
        # Check that we have some blended pixels (not just background)
        center_pixel = result.getpixel((50, 50))
        print(f"Center pixel: {center_pixel}")
        
        # Should be a blend of red and green due to alpha
        assert center_pixel[0] < 255, "Red channel should be reduced due to blending"
        assert center_pixel[1] > 0, "Green channel should be present"
        
        print("✓ Basic alpha over test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic alpha over test failed: {e}")
        return False


def test_size_mismatch():
    """Test handling of size mismatches"""
    print("Testing size mismatch handling...")
    
    bg = Image.new('RGBA', (100, 100), (255, 0, 0, 255))
    fg = Image.new('RGBA', (50, 50), (0, 255, 0, 128))  # Different size
    
    try:
        result = alpha_over(bg, fg)
        assert result.size == bg.size, "Result should match background size"
        print("✓ Size mismatch test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Size mismatch test failed: {e}")
        return False

def test_different_modes():
    """Test handling of different image modes"""
    print("Testing different image modes...")
    
    # Test RGB + RGBA
    bg = Image.new('RGB', (100, 100), (255, 0, 0))
    fg = Image.new('RGBA', (100, 100), (0, 255, 0, 128))
    
    try:
        result = alpha_over(bg, fg)
        assert result.mode == 'RGBA', "Result should be RGBA"
        print("✓ RGB + RGBA mode test passed!")
        return True
        
    except Exception as e:
        print(f"✗ RGB + RGBA mode test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running alpha over fix tests...")
    print("=" * 50)
    
    tests = [
        test_alpha_over_basic,
        test_size_mismatch,
        test_different_modes
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The alpha over fix is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
