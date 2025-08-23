"""
Fix for "image index out of range" error in Alpha over pass
This module provides enhanced validation and error handling for the Alpha over render pass
"""

from PIL import Image
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def validate_alpha_over_inputs(inputs: List[Optional[Image.Image]]) -> List[Image.Image]:
    """
    Validate inputs for Alpha over operation
    
    Args:
        inputs: List of input images (should be exactly 2)
    
    Returns:
        List of validated images
    
    Raises:
        ValueError: If inputs are invalid
    """
    if not inputs or len(inputs) != 2:
        raise ValueError("Alpha over requires exactly 2 input images")
    
    validated_inputs = []
    for i, img in enumerate(inputs):
        if img is None:
            raise ValueError(f"Input image {i+1} is None")
        
        if not isinstance(img, Image.Image):
            raise ValueError(f"Input {i+1} is not a PIL Image")
        
        validated_inputs.append(img)
    
    return validated_inputs

def ensure_rgba_mode(img: Image.Image) -> Image.Image:
    """Ensure image is in RGBA mode"""
    if img.mode == 'RGBA':
        return img
    elif img.mode == 'RGB':
        return img.convert('RGBA')
    elif img.mode == 'L':
        rgb = img.convert('RGB')
        return rgb.convert('RGBA')
    else:
        return img.convert('RGBA')

def safe_alpha_over(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """
    Safe Alpha over operation with comprehensive validation
    
    Args:
        img1: Background image
        img2: Foreground image
    
    Returns:
        Composited image in RGBA mode
    
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If compositing fails
    """
    try:
        # Validate inputs
        if img1 is None or img2 is None:
            raise ValueError("Both images must be provided")
        
        # Ensure both images are in RGBA mode
        img1_rgba = ensure_rgba_mode(img1)
        img2_rgba = ensure_rgba_mode(img2)
        
        # Handle size mismatches
        if img1_rgba.size != img2_rgba.size:
            logger.warning(f"Image size mismatch: {img1_rgba.size} vs {img2_rgba.size}")
            img2_rgba = img2_rgba.resize(img1_rgba.size, Image.Resampling.LANCZOS)
        
        # Perform alpha compositing using PIL's built-in method
        # The alpha_composite method composites img2 over img1 using alpha transparency
        result = Image.alpha_composite(img1_rgba, img2_rgba)
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Alpha over operation failed: {str(e)}")

def debug_slot_contents(slot_table, slot_name: str) -> str:
    """Debug information for a slot"""
    try:
        img = slot_table.get_image(slot_name)
        if img is None:
            return f"Slot '{slot_name}': No image"
        else:
            return f"Slot '{slot_name}': {img.size} {img.mode}"
    except Exception as e:
        return f"Slot '{slot_name}': Error - {str(e)}"

def validate_render_pass_setup(render_pass_widget, slot_table) -> List[str]:
    """
    Validate a render pass setup
    
    Args:
        render_pass_widget: The render pass widget
        slot_table: The slot table
    
    Returns:
        List of validation issues
    """
    issues = []
    
    if render_pass_widget.renderpass_type != "Alpha Over":
        return issues
    
    # Check input slots
    inputs = render_pass_widget.selectedInputs
    if len(inputs) != 2:
        issues.append(f"Alpha over requires 2 inputs, got {len(inputs)}")
    
    for i, slot in enumerate(inputs):
        if slot is None:
            issues.append(f"Input {i+1} is not set")
        else:
            img = slot_table.get_image(slot)
            if img is None:
                issues.append(f"Input {i+1} (slot '{slot}') has no image")
    
    # Check output slot
    if render_pass_widget.selectedOutput is None:
        issues.append("Output slot is not set")
    
    return issues
