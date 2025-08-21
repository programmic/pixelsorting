import os

def get_output_dir():
    """Get the absolute path to the output directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "printouts")
