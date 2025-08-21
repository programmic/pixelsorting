#!/usr/bin/env python3
"""
Test runner script for the unittest framework.
This script provides a unified way to run all tests in the project.
"""

import unittest
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Discover and run all tests
if __name__ == '__main__':
    # Discover tests in the tests directory
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run the tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
