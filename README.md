# PixelSorting V2

## Description
This project provides an image processing application with a modular render pass system for applying various effects to images.

## Setup Instructions
1. Run `setupProject.py` to set up the environment.
2. Activate the virtual environment:
   - On Windows: `.\.venv\Scripts\activate`
   - On macOS/Linux: `source .venv/bin/activate`
3. Run the project using `python scripts/main.py`.

## Adding New Render Passes

To add a new render pass to the application, you need to make changes in three places:

### 1. Create the Image Processing Function (`scripts/passes.py`)

Add your image processing function to `passes.py`. The function should:
- Take a PIL Image as the first parameter
- Return a PIL Image
- Include type hints
- Have parameters that match the settings you'll define in `renderPasses.json`

Example:
```python
def my_effect(img: Image.Image, intensity: float) -> Image.Image:
    """
    Apply my effect to an image.
    
    Args:
        img: Input image
        intensity: Effect strength (0-100)
    
    Returns:
        Processed image
    """
    out = img.copy()
    # Your image processing code here
    return out
```

### 2. Use the Settings Generator Tool

After creating your function in `passes.py`, use the settings generator tool to automatically configure it:

```bash
python scripts/createSettingsTool.py
```

The tool will:
1. Read your function from `passes.py`
2. Guide you through setting up UI parameters
3. Automatically update `renderPasses.json` with your configuration
4. Update function mappings in `renderHook.py`

Example session:
```
Enter the name of your function in passes.py: my_effect
Enter the UI name for your render pass: My Effect
Enter UI label for parameter 'intensity' (or press Enter to use same name): Intensity
Enter minimum value for Intensity: 0
Enter maximum value for Intensity: 100
Enter default value for Intensity: 50
```

The tool supports all parameter types:
- `int`/`float`: Automatically creates sliders with min/max values
- `bool`: Creates toggle switches
- `str`: Creates radio buttons or dropdowns
- Custom parameter names: Maps UI labels to function parameters

The tool will automatically:
1. Create the proper JSON configuration in `renderPasses.json`
2. Add function mappings to `renderHook.py`
3. Set up any needed parameter name mappings


### Important Notes

1. **Number of Inputs**: The system automatically determines if your pass needs one or two input images based on your UI name in `renderPasses.json`. Passes that need two inputs should have names like "Mix By Percent", "Mix Screen", or "Subtract".

2. **Parameter Types**: The render system automatically converts parameter types based on your function's type hints:
   - `int`: Values are converted to integers
   - `float`: Values are converted to floating-point numbers
   - `bool`: "true", "1", "yes" are converted to True
   - `str`: Values are kept as strings

3. **Masks**: If your function needs to support masking, add a `mask: Image.Image` parameter to your function. The system will automatically handle mask slot selection in the UI.

4. **Error Handling**: Your function should:
   - Validate input parameters
   - Return a valid PIL Image
   - Handle edge cases gracefully
   - Raise clear error messages if something goes wrong

5. **Performance**: For computationally intensive operations:
   - Consider using NumPy for bulk operations
   - Use the `@timing` decorator for performance monitoring
   - Provide progress feedback with `tqdm` for long operations

## Troubleshooting

### Common Errors with createSettingsTool.py

1. **"'Attribute' object has no attribute 'id'"**
   ```
   Error extracting function info: 'Attribute' object has no attribute 'id'
   ```
   This error occurs when your function's type hints aren't using simple types. To fix:
   
   Wrong:
   ```python
   def subtract_images(img1: PIL.Image.Image, img2: PIL.Image.Image) -> PIL.Image.Image:
   ```
   
   Correct:
   ```python
   from PIL import Image
   def subtract_images(img1: Image.Image, img2: Image.Image) -> Image.Image:
   ```

2. **"Error: Couldn't find function in passes.py"**
   - Make sure the function name exactly matches (case-sensitive)
   - Verify the function is in `scripts/passes.py`
   - Check for any syntax errors in your function that might prevent parsing

3. **Type Hint Requirements**
   The tool supports these type hints:
   ```python
   img: Image.Image      # For image parameters
   value: int           # For integer sliders
   value: float         # For float sliders
   value: bool          # For switches
   value: str          # For dropdowns/radios
   mask: Image.Image    # For mask support
   ```
   
4. **Two-Input Functions**
   For functions that process two images:
   - First parameter should be `img1: Image.Image`
   - Second parameter should be `img2: Image.Image`
   - Give the pass a name containing "Mix" or "Subtract"
   
   Example:
   ```python
   def subtract_images(img1: Image.Image, img2: Image.Image, strength: float = 1.0) -> Image.Image:
   ```
