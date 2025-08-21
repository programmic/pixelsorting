from typing import List, Dict, Any, Optional, Tuple, Callable
import json
import sys
import os
import inspect
import ast
from pathlib import Path
import re

# Function validation rules configuration
FUNCTION_RULES = {
    # Rules for functions containing specific keywords
    "name_based_rules": [
        {
            "pattern": r"subtract",  # regex pattern to match in function name
            "case_sensitive": False,
            "requirements": {
                "exact_image_inputs": 2,
                "error_message": "Functions with 'subtract' in the name must have exactly 2 image inputs"
            }
        },
        {
            "pattern": r"mix",
            "case_sensitive": False,
            "requirements": {
                "exact_image_inputs": 2,
                "error_message": "Functions with 'mix' in the name must have exactly 2 image inputs"
            }
        },
        {
            "pattern": r"mask",
            "case_sensitive": False,
            "requirements": {
                "requires_mask": True,
                "error_message": "Functions with 'mask' in the name must accept a mask parameter"
            }
        }
    ],
    # General validation rules
    "general_rules": {
        "max_image_inputs": 2,
        "min_image_inputs": 1,
        "allowed_types": ["int", "float", "bool", "str", "Image.Image"],
        "required_return_type": "Image.Image"
    }
}

def validate_function_rules(func_name: str, num_image_inputs: int, has_mask: bool) -> Tuple[bool, str]:
    """
    Validates a function against the defined rules.
    Returns (is_valid, error_message).
    """
    # Check general rules
    general = FUNCTION_RULES["general_rules"]
    if num_image_inputs > general["max_image_inputs"]:
        return False, f"Functions can have at most {general['max_image_inputs']} image inputs"
    if num_image_inputs < general["min_image_inputs"]:
        return False, f"Functions must have at least {general['min_image_inputs']} image input"

    # Check name-based rules
    for rule in FUNCTION_RULES["name_based_rules"]:
        pattern = rule["pattern"]
        if rule["case_sensitive"]:
            match = re.search(pattern, func_name)
        else:
            match = re.search(pattern, func_name, re.IGNORECASE)
            
        if match:
            req = rule["requirements"]
            if "exact_image_inputs" in req and num_image_inputs != req["exact_image_inputs"]:
                return False, req["error_message"]
            if req.get("requires_mask", False) and not has_mask:
                return False, req["error_message"]

    return True, ""

def update_renderhook_maps(func_name: str, ui_name: str, param_mappings: Dict[str, str]) -> None:
    """Updates the function and parameter mappings in renderHook.py"""
    render_hook_path = Path(__file__).parent / "renderHook.py"
    
    with open(render_hook_path, 'r') as f:
        content = f.read()
        
    # Parse the Python code into an AST
    tree = ast.parse(content)
    
    # Find the dictionaries we need to update
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == 'func_name_map':
                        # Add to func_name_map
                        map_str = f'    "{ui_name}": "{func_name}",\n'
                        content = content.replace('func_name_map = {', f'func_name_map = {{\n{map_str}')
                    elif target.id == 'setting_name_map' and param_mappings:
                        # Add parameter mappings
                        for ui_param, func_param in param_mappings.items():
                            map_str = f'    "{ui_param}": "{func_param}",\n'
                            content = content.replace('setting_name_map = {', f'setting_name_map = {{\n{map_str}')
    
    with open(render_hook_path, 'w') as f:
        f.write(content)

def update_json_config(ui_name: str, settings: List[Dict[str, Any]], num_inputs: int) -> None:
    """Updates renderPasses.json with new pass configuration"""
    json_path = Path(__file__).parent.parent / 'renderPasses.json'
    
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}
    
    # Add category with number of inputs as first setting
    category = ui_name.lower().replace(" ", "_")
    settings.insert(0, {
        "kategory": category,
        "num_inputs": num_inputs
    })
    
    config[ui_name] = settings
    
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)

def validate_type_hint(annotation: ast.AST) -> Optional[str]:
    """
    Validates and extracts the type hint information.
    Returns the simple type name or None if invalid.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Attribute):
        # Handle complex types like Image.Image
        if isinstance(annotation.value, ast.Name) and annotation.value.id == 'Image' and annotation.attr == 'Image':
            return 'Image.Image'
        else:
            print(f"\nError: Complex type hint detected. Found: '{ast.unparse(annotation)}'")
            print("Please use simple types or 'Image.Image' for PIL images.")
            print("Example: def my_func(img: Image.Image, value: int) -> Image.Image:")
            return None
    elif annotation is None:
        print("\nError: Missing type hint.")
        print("All parameters must have type hints.")
        print("Example: def my_func(value: int) -> Image.Image:")
        return None
    else:
        print(f"\nError: Unsupported type hint: {ast.unparse(annotation)}")
        print("Supported types: Image.Image, int, float, bool, str")
        return None

def extract_function_info(func_path: str, func_name: str) -> Optional[Dict[str, Any]]:
    """Extracts parameter information from a function in passes.py"""
    try:
        with open(func_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # First, check if the file can be parsed
        if not tree:
            print("\nError: Could not parse passes.py")
            print("Check for syntax errors in the file.")
            return None
            
        # Check for Image import
        has_image_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'PIL':
                for name in node.names:
                    if name.name == 'Image':
                        has_image_import = True
                        break
            elif isinstance(node, ast.Import):
                for name in node.names:
                    if name.name == 'PIL':
                        has_image_import = True
                        break
        
        if not has_image_import:
            print("\nWarning: No 'PIL.Image' or 'Image' import found.")
            print("Add 'from PIL import Image' at the top of passes.py")
        
        function_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                function_found = True
                # Check return annotation
                if not node.returns:
                    print("\nError: Missing return type hint.")
                    print(f"Add return type: def {func_name}(...) -> Image.Image:")
                    return None
                
                return_type = validate_type_hint(node.returns)
                if return_type != 'Image.Image':
                    print("\nError: Function must return Image.Image")
                    print(f"Change: def {func_name}(...) -> Image.Image:")
                    return None
                
                params = []
                image_params = []
                has_mask = False
                
                for arg in node.args.args:
                    # Check for missing parameter names
                    if not arg.arg:
                        print("\nError: Found parameter without name")
                        print(f"In function: {func_name}")
                        return None
                    
                    type_hint = validate_type_hint(arg.annotation)
                    if not type_hint:
                        return None
                        
                    # Handle image parameters
                    if type_hint == 'Image.Image':
                        if arg.arg == 'mask':
                            has_mask = True
                        else:
                            image_params.append(arg.arg)
                        continue
                        
                    # Skip self parameter
                    if arg.arg == 'self':
                        continue
                        
                    # Validate other parameters
                    if type_hint not in ['int', 'float', 'bool', 'str']:
                        print(f"\nError: Parameter '{arg.arg}' has unsupported type")
                        print("Supported types: int, float, bool, str")
                        print(f"Found: {ast.unparse(arg.annotation) if arg.annotation else 'no type hint'}")
                        return None
                        
                    params.append({
                        'name': arg.arg,
                        'annotation': type_hint
                    })
                
                if not image_params:
                    print("\nError: Function must have at least one image parameter")
                    print("Add: img: Image.Image as first parameter")
                    return None
                
                # Validate function against rules
                num_inputs = len(image_params)
                is_valid, error_message = validate_function_rules(func_name, num_inputs, has_mask)
                
                if not is_valid:
                    print(f"\nError: {error_message}")
                    print("Current image parameters:", ', '.join(image_params))
                    print(f"Current number of image inputs: {num_inputs}")
                    if has_mask:
                        print("Has mask parameter: Yes")
                    return None
                    
                return {
                    'params': params,
                    'num_inputs': num_inputs,
                    'has_mask': has_mask,
                    'image_param_names': image_params
                }
        
        if not function_found:
            print(f"\nError: Function '{func_name}' not found in passes.py")
            print("Check for:")
            print("1. Correct function name (case sensitive)")
            print("2. Function is defined in passes.py")
            print("3. No typos in the function name")
            return None
            
    except SyntaxError as e:
        print(f"\nSyntax error in passes.py: {e}")
        print(f"Line {e.lineno}, Column {e.offset}: {e.text}")
        return None
    except Exception as e:
        print(f"\nError analyzing function: {e}")
        print("Make sure your function:")
        print("1. Has proper type hints")
        print("2. Uses supported parameter types")
        print("3. Has correct syntax")
        return None

def create_setting(config_type: str) -> Dict[str, Any]:
    """Creates a single setting configuration based on type."""
    name = input("Enter setting name (label): ").strip()
    
    config = {"label": name, "type": config_type}
    
    if config_type in ["radio", "dropdown"]:
        options = input("Enter comma-separated options (e.g., Gaussian,Box,Median): ").strip()
        config["options"] = [opt.strip() for opt in options.split(",")]
        
        # Print options with numbers for easier selection
        print("\nAvailable options:")
        for i, opt in enumerate(config["options"], 1):
            print(f"{i}. {opt}")
        
        while True:
            default_input = input(f"Enter default option (1-{len(config['options'])}): ").strip()
            try:
                default_idx = int(default_input) - 1
                if 0 <= default_idx < len(config["options"]):
                    config["default"] = config["options"][default_idx]
                    break
                print(f"Please enter a number between 1 and {len(config['options'])}")
            except ValueError:
                # Allow direct option input as fallback
                if default_input in config["options"]:
                    config["default"] = default_input
                    break
                print("Please enter a valid number or exact option name")
    
    elif config_type in ["slider", "multislider"]:
        config["min"] = float(input("Enter minimum value: "))
        config["max"] = float(input("Enter maximum value: "))
        
        if config_type == "multislider":
            default_values = input("Enter comma-separated default values (e.g., 1,0.5,0): ").strip()
            config["default"] = [float(v) for v in default_values.split(",")]
        else:
            config["default"] = float(input("Enter default value: "))
    
    elif config_type == "switch":
        while True:
            default_str = input("Enable by default? (y/n): ").strip().lower()
            if default_str in ['y', 'n']:
                config["default"] = default_str == 'y'
                break
            print("Please enter 'y' for yes or 'n' for no")
    
    return config

def create_renderpass_settings() -> List[Dict[str, Any]]:
    """Creates a complete settings configuration for a render pass type."""
    settings = []
    
    print("\nAvailable setting types:")
    print("- radio: Group of radio buttons (mutually exclusive options)")
    print("- dropdown: Dropdown select box")
    print("- slider: Single value slider")
    print("- multislider: Multiple value slider")
    print("- switch: On/Off toggle switch")
    
    while True:
        print("\nCurrent settings:")
        for i, setting in enumerate(settings, 1):
            value_info = ""
            if setting["type"] in ["radio", "dropdown"]:
                value_info = f" - Options: {', '.join(setting['options'])}"
            elif setting["type"] == "switch":
                value_info = f" - Default: {'On' if setting['default'] else 'Off'}"
            print(f"{i}. {setting['label']} ({setting['type']}){value_info}")
            
        print("\nAdd a new setting")
        print("1. Add radio button group (mutually exclusive options)")
        print("2. Add dropdown select box")
        print("3. Add slider")
        print("4. Add multislider")
        print("5. Add toggle switch (on/off)")
        print("6. Finish")
        
        choice = input("Select option (1-8): ").strip()
        
        if choice == "1":
            settings.append(create_setting("radio"))
        elif choice == "2":
            settings.append(create_setting("dropdown"))
        elif choice == "3":
            settings.append(create_setting("slider"))
        elif choice == "4":
            settings.append(create_setting("multislider"))
        elif choice == "5":
            settings.append(create_setting("switch"))
        elif choice == "6":
            break
        else:
            print("Invalid choice, please try again.")
    
    return settings

def generate_code_block(renderpass_type: str, settings: List[Dict[str, Any]]) -> str:
    """Generates Python code block for the settings."""
    code = f"if renderpass_type == \"{renderpass_type}\":\n"
    code += "    return [\n"
    
    for setting in settings:
        formatted_setting = "        {\n"
        for key, value in setting.items():
            if isinstance(value, str):
                formatted_setting += f"            \"{key}\": \"{value}\",\n"
            else:
                formatted_setting += f"            \"{key}\": {value},\n"
        formatted_setting = formatted_setting.rstrip(",\n") + "\n        },\n"
        code += formatted_setting
    
    code += "    ]\n"
    return code

def validate_ui_name(name: str) -> bool:
    """Validates the UI name for the render pass"""
    if not name:
        print("\nError: UI name cannot be empty")
        return False
    if not name[0].isupper():
        print("\nError: UI name should start with an uppercase letter")
        print("Example: 'My Effect' instead of 'my effect'")
        return False
    if any(c in name for c in "!@#$%^&*()+={}[]|\;:\"'<>,.?/"):
        print("\nError: UI name contains invalid characters")
        print("Use only letters, numbers, and spaces")
        return False
    return True

def validate_numeric_input(value_str: str, param_name: str, is_min: bool = False) -> Optional[float]:
    """Validates numeric input for sliders"""
    try:
        value = float(value_str)
        if is_min and value < 0:
            print(f"\nWarning: Consider using a non-negative minimum value for {param_name}")
        return value
    except ValueError:
        print(f"\nError: Invalid number format for {param_name}")
        print("Please enter a valid number (e.g., 0, 1.5, -10)")
        return None

def main():
    print("="*50)
    print("Render Pass Settings Generator")
    print("="*50)
    print("\nThis tool will help you configure your render pass for the application.")
    print("Make sure your function is properly defined in passes.py first.")
    
    # Get function information
    func_name = input("\nEnter the name of your function in passes.py: ").strip()
    if not func_name:
        print("\nError: Function name cannot be empty")
        return
    
    passes_path = Path(__file__).parent / "passes.py"
    if not passes_path.exists():
        print("\nError: passes.py not found!")
        print(f"Expected location: {passes_path}")
        return
    
    print("\nAnalyzing function...")
    func_info = extract_function_info(str(passes_path), func_name)
    if not func_info:
        return  # Error message already printed by extract_function_info
        
    # Show function analysis results
    print("\nFunction Analysis:")
    print(f"Number of image inputs: {func_info['num_inputs']}")
    print(f"Image parameters: {', '.join(func_info['image_param_names'])}")
    if func_info['has_mask']:
        print("Supports masking: Yes")
    print(f"Additional parameters: {len(func_info['params'])}")
    
    # For two-input functions, ensure the UI name reflects this
    if func_info['num_inputs'] == 2:
        print("\nNote: This function requires two image inputs.")
        print("The UI name should contain 'Mix' or 'Subtract' to indicate this.")
        print("Examples: 'Mix By Value', 'Subtract Images', 'Mix Overlay'")
    
    # Get UI name for the render pass
    ui_name = input("\nEnter the UI name for your render pass (e.g., 'My Effect'): ").strip()
    
    # Create parameter mappings
    param_mappings = {}
    for param in func_info['params']:
        ui_name_param = input(f"\nEnter UI label for parameter '{param['name']}' (or press Enter to use same name): ").strip()
        if ui_name_param and ui_name_param != param['name']:
            param_mappings[ui_name_param] = param['name']
    
    # Create settings based on parameters
    settings = []
    for param in func_info['params']:
        ui_label = next((k for k, v in param_mappings.items() if v == param['name']), param['name'])
        print(f"\nConfiguring parameter: {ui_label}")
        
        if param['annotation'] == 'bool':
            print("Type: Boolean switch")
            default_str = input("Enable by default? (y/n): ").strip().lower()
            if default_str not in ['y', 'n']:
                print("Invalid input. Using default value: false")
                default_str = 'n'
            settings.append({
                "label": ui_label,
                "type": "switch",
                "default": default_str == 'y'
            })
            
        elif param['annotation'] in ['int', 'float']:
            print(f"Type: {'Integer' if param['annotation'] == 'int' else 'Float'} slider")
            
            while True:
                min_val = validate_numeric_input(
                    input(f"Enter minimum value for {ui_label}: "),
                    ui_label,
                    is_min=True
                )
                if min_val is not None:
                    break
            
            while True:
                max_val = validate_numeric_input(
                    input(f"Enter maximum value for {ui_label}: "),
                    ui_label
                )
                if max_val is not None and max_val > min_val:
                    break
                print(f"Maximum value must be greater than minimum ({min_val})")
            
            while True:
                default = validate_numeric_input(
                    input(f"Enter default value for {ui_label}: "),
                    ui_label
                )
                if default is not None and min_val <= default <= max_val:
                    break
                print(f"Default must be between {min_val} and {max_val}")
            
            settings.append({
                "label": ui_label,
                "type": "slider",
                "min": min_val,
                "max": max_val,
                "default": default,
                "integer": param['annotation'] == 'int'
            })
            
        elif param['annotation'] == 'str':
            print("Type: String selection")
            while True:
                options_str = input(f"Enter comma-separated options for {ui_label}: ").strip()
                if not options_str:
                    print("Error: Must provide at least one option")
                    continue
                    
                options = [opt.strip() for opt in options_str.split(',')]
                options = [opt for opt in options if opt]  # Remove empty options
                
                if not options:
                    print("Error: No valid options provided")
                    continue
                    
                if len(set(options)) != len(options):
                    print("Error: Duplicate options found")
                    continue
                    
                break
                
            print("\nAvailable options:")
            for i, opt in enumerate(options, 1):
                print(f"{i}. {opt}")
                
            while True:
                default_str = input(f"Enter default option (1-{len(options)}): ").strip()
                try:
                    default_idx = int(default_str) - 1
                    if 0 <= default_idx < len(options):
                        break
                    print(f"Please enter a number between 1 and {len(options)}")
                except ValueError:
                    print("Please enter a valid number")
    
    # Update renderPasses.json
    print("\nUpdating renderPasses.json...")
    
    # Create the full settings configuration
    full_settings = settings.copy()
    
    # Add input configuration based on number of inputs
    input_config = {
        "inputs_required": func_info['num_inputs']
    }
    if func_info['num_inputs'] == 2:
        if not any(word in ui_name.lower() for word in ['mix', 'subtract']):
            print("\nWarning: Two-input function name should contain 'Mix' or 'Subtract'")
            print("This helps users understand that two images are needed.")
            if input("Would you like to change the UI name? (y/n): ").lower() == 'y':
                new_name = input("Enter new UI name: ").strip()
                if new_name:
                    ui_name = new_name
    
    # Add mask configuration if supported
    if func_info['has_mask']:
        full_settings.append({
            "label": "Mask Settings",
            "type": "mask_config",
            "default": None
        })
    
    update_json_config(ui_name, full_settings, func_info['num_inputs'])
    
    # Update renderHook.py
    print("Updating renderHook.py...")
    update_renderhook_maps(func_name, ui_name, param_mappings)
    
    print("\n" + "="*50)
    print("Configuration Complete!")
    print("="*50)
    print("\nThe following changes have been made:")
    
    print("\n1. Added to renderPasses.json:")
    print(f"   - Pass name: '{ui_name}'")
    print(f"   - Number of inputs: {func_info['num_inputs']}")
    if func_info['has_mask']:
        print("   - Mask support: Enabled")
    if settings:
        print("   - Parameters:")
        for setting in settings:
            print(f"     • {setting['label']} ({setting['type']})")
    
    print("\n2. Updated renderHook.py:")
    print(f"   - Added function mapping: {ui_name} -> {func_name}")
    if param_mappings:
        print("   - Added parameter mappings:")
        for ui_param, func_param in param_mappings.items():
            print(f"     • {ui_param} -> {func_param}")
    
    print("\nYou can now use your new render pass in the application!")
    print("The pass will appear in the render pass dropdown menu.")
    print("="*50)

if __name__ == "__main__":
    main()
