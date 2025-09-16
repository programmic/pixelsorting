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
            # Only match 'mask' if function name does NOT contain both 'generate' and 'mask' in any order
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

def validate_function_rules(func_name: str, num_image_inputs: int, has_mask: bool, param_names: List[str]) -> Tuple[bool, str]:
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
        # Skip mask rule for any function name containing both 'generate' and 'mask' (any order, any chars between)
        lower_name = func_name.lower()
        if "mask" in pattern and ("generate" in lower_name and "mask" in lower_name):
            continue
        if match:
            req = rule["requirements"]
            if req.get("requires_mask", False):
                if not any(p.lower() == "mask" for p in param_names):
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
    
    # Remove existing pass if it already exists
    if ui_name in config:
        print(f"Removing existing pass '{ui_name}' to replace with new configuration...")
        del config[ui_name]

    # New format: top-level dict for each pass
    # Prompt for function alias if not provided
    func_alias = ui_name.lower().replace(" ", "_")
    original_func_name = settings[-1].get("original_func_name") if "original_func_name" in settings[-1] else ui_name
    pass_dict = {
        "original_func_name": original_func_name,
        "num_inputs": num_inputs,
        "function_alias": func_alias,
        "settings": []
    }
    # Each subsetting: name, unique alias, requirements
    for s in settings:
        subsetting = {
            "name": s.get("label"),
            "alias": s.get("alias", s.get("label")),
            "type": s.get("type"),
            "default": s.get("default")
        }
        # Add slider/range options
        for key in ["min", "max", "integer", "options"]:
            if key in s:
                subsetting[key] = s[key]
        # Add requirements (can be bool or logic string)
        if "requires" in s:
            subsetting["requirements"] = s["requires"]
        pass_dict["settings"].append(subsetting)

    config[ui_name] = pass_dict

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
    elif isinstance(annotation, ast.Subscript):
        # Handle Optional[Image.Image]
        if (
            isinstance(annotation.value, ast.Name) and annotation.value.id == 'Optional' and
            ((isinstance(annotation.slice, ast.Name) and annotation.slice.id == 'Image.Image') or
             (isinstance(annotation.slice, ast.Attribute) and isinstance(annotation.slice.value, ast.Name) and annotation.slice.value.id == 'Image' and annotation.slice.attr == 'Image'))
        ):
            return 'Image.Image'
        # Handle Optional[int], Optional[float], etc.
        if isinstance(annotation.value, ast.Name) and annotation.value.id == 'Optional':
            if isinstance(annotation.slice, ast.Name):
                return annotation.slice.id
        print(f"\nError: Unsupported type hint: {ast.unparse(annotation)}")
        print("Supported types: Image.Image, int, float, bool, str, Optional[Image.Image]")
        return None
    elif annotation is None:
        print("\nError: Missing type hint.")
        print("All parameters must have type hints.")
        print("Example: def my_func(value: int) -> Image.Image:")
        return None
    else:
        print(f"\nError: Unsupported type hint: {ast.unparse(annotation)}")
        print("Supported types: Image.Image, int, float, bool, str, Optional[Image.Image]")
        return None

def extract_function_info(func_path: str, func_name: str) -> Optional[Dict[str, Any]]:
    """Extracts parameter information from a function in passes.py"""
    try:
        with open(func_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # First, check if the file can be parsed
        if not tree:
            print("\n\033[91m==============================\033[0m")
            print("\033[91mError: Could not parse passes.py\033[0m")
            print("\033[91mCheck for syntax errors in the file.\033[0m")
            print("\033[91m==============================\033[0m\n")
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
            print("\n\033[93m------------------------------\033[0m")
            print("\033[93mWarning: No 'PIL.Image' or 'Image' import found.\033[0m")
            print("\033[93mAdd 'from PIL import Image' at the top of passes.py\033[0m")
            print("\033[93m------------------------------\033[0m\n")
        
        function_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                function_found = True
                # Check return annotation
                if not node.returns:
                    print("\n\033[91m==============================\033[0m")
                    print("\033[91mError: Missing return type hint.\033[0m")
                    print(f"\033[91mAdd return type: def {func_name}(...) -> Image.Image:\033[0m")
                    print("\033[91m==============================\033[0m\n")
                    return None

                return_type = validate_type_hint(node.returns)
                if return_type != 'Image.Image':
                    print("\n\033[91m==============================\033[0m")
                    print("\033[91mError: Function must return Image.Image\033[0m")
                    print(f"\033[91mChange: def {func_name}(...) -> Image.Image:\033[0m")
                    print("\033[91m==============================\033[0m\n")
                    return None
                
                params = []
                image_params = []
                has_mask = False
                
                for arg in node.args.args:
                    # Check for missing parameter names
                    if not arg.arg:
                        print("\n\033[91m==============================\033[0m")
                        print("\033[91mError: Found parameter without name\033[0m")
                        print(f"\033[91mIn function: {func_name}\033[0m")
                        print("\033[91m==============================\033[0m\n")
                        return None
                    
                    type_hint = validate_type_hint(arg.annotation)
                    if not type_hint:
                        print("\033[91m==============================\033[0m")
                        print(f"\033[91mError: Invalid or missing type hint for parameter '{arg.arg}' in function '{func_name}'.\033[0m")
                        print("\033[91m==============================\033[0m\n")
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
                        print("\n\033[91m==============================\033[0m")
                        print(f"\033[91mError: Parameter '{arg.arg}' has unsupported type\033[0m")
                        print("\033[91mSupported types: int, float, bool, str\033[0m")
                        print(f"\033[91mFound: {ast.unparse(arg.annotation) if arg.annotation else 'no type hint'}\033[0m")
                        print("\033[91m==============================\033[0m\n")
                        return None
                        
                    params.append({
                        'name': arg.arg,
                        'annotation': type_hint
                    })
                
                if not image_params:
                    print("\n\033[91m==============================\033[0m")
                    print("\033[91mError: Function must have at least one image parameter\033[0m")
                    print("\033[91mAdd: img: Image.Image as first parameter\033[0m")
                    print("\033[91m==============================\033[0m\n")
                    return None
                
                # Validate function against rules
                num_inputs = len(image_params)
                is_valid, error_message = validate_function_rules(func_name, num_inputs, has_mask, [p.arg for p in node.args.args])
                
                if not is_valid:
                    print("\n\033[91m==============================\033[0m")
                    print(f"\033[91mError: {error_message}\033[0m")
                    print(f"\033[91mCurrent image parameters: {', '.join(image_params)}\033[0m")
                    print(f"\033[91mCurrent number of image inputs: {num_inputs}\033[0m")
                    if has_mask:
                        print("\033[91mHas mask parameter: Yes\033[0m")
                    print("\033[91m==============================\033[0m\n")
                    return None
                    
                return {
                    'params': params,
                    'num_inputs': num_inputs,
                    'has_mask': has_mask,
                    'image_param_names': image_params
                }
        
        if not function_found:
            print("\n\033[91m==============================\033[0m")
            print(f"\033[91mError: Function '{func_name}' not found in passes.py\033[0m")
            print("\033[91mCheck for:\033[0m")
            print("\033[91m1. Correct function name (case sensitive)\033[0m")
            print("\033[91m2. Function is defined in passes.py\033[0m")
            print("\033[91m3. No typos in the function name\033[0m")
            print("\033[91m==============================\033[0m\n")
            return None
            
    except SyntaxError as e:
        print("\n\033[91m==============================\033[0m")
        print(f"\033[91mSyntax error in passes.py: {e}\033[0m")
        print(f"\033[91mLine {e.lineno}, Column {e.offset}: {e.text}\033[0m")
        print("\033[91m==============================\033[0m\n")
        return None
    except Exception as e:
        print("\n\033[91m==============================\033[0m")
        print(f"\033[91mError analyzing function: {e}\033[0m")
        print("\033[91mMake sure your function:\033[0m")
        print("\033[91m1. Has proper type hints\033[0m")
        print("\033[91m2. Uses supported parameter types\033[0m")
        print("\033[91m3. Has correct syntax\033[0m")
        print("\033[91m==============================\033[0m\n")
        return None

def create_setting(param):
    annotation = param['annotation']
    name = param['name']
    ui_label = input(f"UI label for parameter '{name}' (or press Enter to use same name): ").strip() or name
    alias = input(f"Alias for parameter '{name}' (or press Enter to use same as label): ").strip() or ui_label

    # Suggest widget type based on annotation
    suggested_type = None
    if annotation == 'bool':
        suggested_type = 'switch'
    elif annotation == 'int':
        suggested_type = 'slider'
    elif annotation == 'float':
        suggested_type = 'multislider'
    elif annotation == 'str':
        # Let user choose between dropdown or radio
        suggested_type = 'dropdown'

    print(f"Suggested widget type for '{ui_label}' ({annotation}): {suggested_type}")
    custom_type = input(f"Enter widget type to use (press Enter to accept suggestion): ").strip().lower()
    widget_type = custom_type if custom_type else suggested_type

    if widget_type == 'switch':
        default = input(f"Default value for {ui_label} (y/n, default n): ").strip().lower() == 'y'
        requirements = {}
        add_dep = input(f"Do you want to add requirements for this switch? (y/n): ").strip().lower() == 'y'
        while add_dep:
            dep_label = input("  Enter label or alias of the control to require: ").strip()
            logic = input("  Requirement logic (e.g. true, false, kernel>15): ").strip()
            if dep_label and logic:
                requirements[dep_label] = logic
            add_dep = input("  Add another requirement? (y/n): ").strip().lower() == 'y'
        setting = {
            "label": ui_label,
            "alias": alias,
            "type": "switch",
            "default": default
        }
        if requirements:
            setting["requires"] = requirements
        return setting
    elif widget_type in ['slider', 'multislider']:
        min_val = float(input(f"Min value for {ui_label}: "))
        max_val = float(input(f"Max value for {ui_label}: "))
        if widget_type == 'multislider':
            default_min = float(input(f"Default min value for {ui_label}: "))
            default_max = float(input(f"Default max value for {ui_label}: "))
            default = [default_min, default_max]
        else:
            default = float(input(f"Default value for {ui_label}: "))
        requires = {}
        add_dep = input(f"Does this control depend on another control? (y/n): ").strip().lower() == 'y'
        while add_dep:
            dep_label = input("  Enter label or alias of required control: ").strip()
            dep_val = input("  Required value for that control (y/n): ").strip().lower() == 'y'
            if dep_label:
                requires[dep_label] = dep_val
            add_dep = input("  Add another dependency? (y/n): ").strip().lower() == 'y'
        setting = {
            "label": ui_label,
            "alias": alias,
            "type": widget_type,
            "min": min_val,
            "max": max_val,
            "default": default,
            "integer": annotation == 'int'
        }
        if requires:
            setting["requires"] = requires
        return setting
    elif widget_type in ['dropdown', 'radio']:
        options = [opt.strip() for opt in input(f"Comma-separated options for {ui_label}: ").split(',') if opt.strip()]
        default = options[0] if options else ''
        requires = {}
        add_dep = input(f"Does this control depend on another control? (y/n): ").strip().lower() == 'y'
        while add_dep:
            dep_label = input("  Enter label or alias of required control: ").strip()
            dep_val = input("  Required value for that control (y/n): ").strip().lower() == 'y'
            if dep_label:
                requires[dep_label] = dep_val
            add_dep = input("  Add another dependency? (y/n): ").strip().lower() == 'y'
        setting = {
            "label": ui_label,
            "alias": alias,
            "type": widget_type,
            "options": options,
            "default": default
        }
        if requires:
            setting["requires"] = requires
        return setting
    elif widget_type == 'text':
        default = input(f"Default text for {ui_label} (press Enter for none): ").strip()
        requires = {}
        add_dep = input(f"Does this control depend on another control? (y/n): ").strip().lower() == 'y'
        while add_dep:
            dep_label = input("  Enter label or alias of required control: ").strip()
            dep_val = input("  Required value for that control (y/n): ").strip().lower() == 'y'
            if dep_label:
                requires[dep_label] = dep_val
            add_dep = input("  Add another dependency? (y/n): ").strip().lower() == 'y'
        setting = {
            "label": ui_label,
            "alias": alias,
            "type": "text",
            "default": default
        }
        if requires:
            setting["requires"] = requires
        return setting
    elif widget_type == 'image_input':
        available_slots = input(f"Available slots for {ui_label} (comma-separated, or leave blank): ").strip()
        slots = [s.strip() for s in available_slots.split(',') if s.strip()] if available_slots else []
        setting = {
            "label": ui_label,
            "alias": alias,
            "type": "image_input"
        }
        if slots:
            setting["availableSlots"] = slots
        return setting
    return None

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
        ui_label = input(f"UI label for parameter '{param['name']}' (or press Enter to use same name): ").strip()
        if ui_label and ui_label != param['name']:
            param_mappings[ui_label] = param['name']

    # Create settings based on parameters
    settings = []
    i = 0
    params = func_info['params']
    while i < len(params):
        param = params[i]
        ui_label = next((k for k, v in param_mappings.items() if v == param['name']), param['name'])
        annotation = param['annotation']

        # Check for consecutive min/max number inputs
        if (
            i + 1 < len(params)
            and annotation in ['int', 'float']
            and params[i + 1]['annotation'] == annotation
            and 'min' in param['name'].lower()
            and 'max' in params[i + 1]['name'].lower()
        ):
            print(f"\nDetected consecutive '{annotation}' parameters: '{param['name']}' and '{params[i+1]['name']}'")
            use_multislider = input("Would you like to use a SuperQT multislider for these? (y/n): ").strip().lower() == 'y'
            if use_multislider:
                min_label = next((k for k, v in param_mappings.items() if v == param['name']), param['name'])
                max_label = next((k for k, v in param_mappings.items() if v == params[i+1]['name']), params[i+1]['name'])
                min_val = validate_numeric_input(input(f"Enter minimum value for {min_label}: "), min_label, is_min=True)
                max_val = validate_numeric_input(input(f"Enter maximum value for {max_label}: "), max_label)
                default_min = validate_numeric_input(input(f"Enter default min value: "), min_label)
                default_max = validate_numeric_input(input(f"Enter default max value: "), max_label)
                while (default_min is not None and default_max is not None) and (default_min < min_val or default_max > max_val or default_min > default_max):
                    print(f"Defaults must be within [{min_val}, {max_val}] and min <= max")
                    default_min = validate_numeric_input(input(f"Enter default min value: "), min_label)
                    default_max = validate_numeric_input(input(f"Enter default max value: "), max_label)
                settings.append({
                    "label": f"{min_label} / {max_label}",
                    "type": "multislider",
                    "min": min_val,
                    "max": max_val,
                    "default": [default_min, default_max],
                    "integer": annotation == 'int'
                })
                i += 2
                continue
        # ...existing code for bool, int, float, str...
        print(f"\nConfiguring parameter: {ui_label}")
        if annotation == 'bool':
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
        elif annotation in ['int', 'float']:
            print(f"Type: {'Integer' if annotation == 'int' else 'Float'} slider")
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
                "integer": annotation == 'int'
            })
        elif annotation == 'str':
            print("Type: String selection")
            while True:
                options_str = input(f"Enter comma-separated options for {ui_label}: ").strip()
                if not options_str:
                    print("Error: Must provide at least one option")
                    continue
                options = [opt.strip() for opt in options_str.split(',')]
                options = [opt for opt in options if opt]
                if not options:
                    print("Error: No valid options provided")
                    continue
                if len(set(options)) != len(options):
                    print("Error: Duplicate options found")
                    continue
                break
            print("\nAvailable options:")
            for idx, opt in enumerate(options, 1):
                print(f"{idx}. {opt}")
            while True:
                default_str = input(f"Enter default option (1-{len(options)}): ").strip()
                try:
                    default_idx = int(default_str) - 1
                    if 0 <= default_idx < len(options):
                        break
                    print(f"Please enter a number between 1 and {len(options)}")
                except ValueError:
                    print("Please enter a valid number")
        # ...existing code...
        i += 1

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
    
    print("\nYou can now use your new render pass in1 the application!")
    print("The pass will appear in the render pass dropdown menu.")
    print("="*50)



def ensure_valid_json(json_path):
    """Ensures the file at json_path contains valid JSON. Creates '{}' if empty or invalid."""
    import json
    from pathlib import Path
    if not Path(json_path).exists():
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write('{}')
        return
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError('Empty file')
            json.loads(content)
    except Exception:
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write('{}')

def get_unimplemented_functions():
    """
    Returns a list of function names in passes.py not present in renderPasses.json and not containing #globalignore in their body
    """
    from pathlib import Path
    import ast
    passes_path = Path(__file__).parent / "passes.py"
    json_path = Path(__file__).parent.parent / "renderPasses.json"
    ensure_valid_json(json_path)
    try:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            config = {}
        # Build a set of all implemented function names (using original_func_name if present)
        implemented_names = set()
        for ui_name, settings in config.items():
            # Try to match function names from UI names (exact and normalized)
            implemented_names.add(ui_name)
            implemented_names.add(ui_name.lower().replace(' ', '_'))
            # Check for original_func_name in settings
            for entry in settings:
                if isinstance(entry, dict) and 'original_func_name' in entry and entry['original_func_name']:
                    implemented_names.add(entry['original_func_name'])
                    implemented_names.add(entry['original_func_name'].lower().replace(' ', '_'))
        with open(passes_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        all_funcs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                # Get function source code
                func_src = ast.get_source_segment(content, node)
                if func_src and '#globalignore' in func_src:
                    continue
                # Only show functions not present in config (by name or normalized name)
                if node.name not in implemented_names and node.name.lower().replace(' ', '_') not in implemented_names:
                    all_funcs.append(node.name)
        return all_funcs
    except Exception as e:
        print(f"Error occurred while getting unimplemented functions: {e}")

def interactive_menu():
    while True:
        while True:
            print("\n==== Render Pass Settings Menu ====")
            print("1. Configure missing functions")
            print("2. Configure any function by name")
            print("3. Exit")
            print("4. Quick tutorial (inputs, widgets, usage)")
            choice = input("Select option (1-4): ").strip()
            if choice == "1":
                missing = get_unimplemented_functions()
                if not missing:
                    print("No missing functions to configure.")
                    continue
                for func_name in missing:
                    print(f"\nConfiguring function: {func_name}")
                    try:
                        # The main configuration logic goes here
                        passes_path = str(Path(__file__).parent / "passes.py")
                        func_info = extract_function_info(passes_path, func_name)
                        if not func_info:
                            print(f"Skipping unsupported or invalid function '{func_name}'.")
                            continue
                        ui_name = input(f"Enter UI name for '{func_name}' (or press Enter to use function name, or type 'skip' to skip): ").strip()
                        if ui_name.lower() == 'skip':
                            print(f"Skipping function '{func_name}'.")
                            continue
                        if ui_name.lower() == 'disable':
                            passes_path_obj = Path(__file__).parent / "passes.py"
                            with open(passes_path_obj, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                            for idx, line in enumerate(lines):
                                if line.strip().startswith(f"def {func_name}(") and "#globalignore" not in line:
                                    lines[idx] = line.rstrip() + " #globalignore\n"
                                    break
                            with open(passes_path_obj, "w", encoding="utf-8") as f:
                                f.writelines(lines)
                            print(f"Function '{func_name}' disabled with #globalignore.")
                            continue
                        param_mappings = {}
                        param_labels = []
                        j = 0
                        params_len = len(func_info['params'])
                        while j < params_len:
                            param = func_info['params'][j]
                            ui_label = input(f"UI label for parameter '{param['name']}' (or press Enter to use same name): ").strip()
                            if ui_label.lower() == 'edit' and j > 0:
                                j -= 1
                                continue
                            if ui_label and ui_label != param['name']:
                                param_mappings[ui_label] = param['name']
                                param_labels.append(ui_label)
                            else:
                                param_labels.append(param['name'])
                            j += 1
                        settings = []
                        i = 0
                        params = func_info['params']
                        while i < len(params):
                            param = params[i]
                            ui_label = param_labels[i] if i < len(param_labels) else param['name']
                            annotation = param['annotation']
                            if (
                                i + 1 < len(params)
                                and annotation in ['int', 'float']
                                and params[i + 1]['annotation'] == annotation
                                and 'min' in param['name'].lower()
                                and 'max' in params[i + 1]['name'].lower()
                            ):
                                print(f"\nDetected consecutive '{annotation}' parameters: '{param['name']}' and '{params[i+1]['name']}'")
                                use_multislider = input("Would you like to use a multislider for these? (y/n): ").strip().lower()
                                if use_multislider == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                if use_multislider == 'y':
                                    min_label = param_labels[i] if i < len(param_labels) else param['name']
                                    max_label = param_labels[i+1] if i+1 < len(param_labels) else params[i+1]['name']
                                    min_val = validate_numeric_input(input(f"Enter minimum value for {min_label}: "), min_label, is_min=True)
                                    if str(min_val).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    max_val = validate_numeric_input(input(f"Enter maximum value for {max_label}: "), max_label)
                                    if str(max_val).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    while max_val is not None and min_val is not None and max_val <= min_val:
                                        print(f"Maximum value must be greater than minimum ({min_val})")
                                        max_val = validate_numeric_input(input(f"Enter maximum value for {max_label}: "), max_label)
                                        if str(max_val).lower() == 'edit' and i > 0:
                                            i -= 1
                                            continue
                                    default_min = validate_numeric_input(input(f"Enter default min value: "), min_label)
                                    if str(default_min).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    default_max = validate_numeric_input(input(f"Enter default max value: "), max_label)
                                    if str(default_max).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    while (default_min is not None and default_max is not None) and (default_min < min_val or default_max > max_val or default_min > default_max):
                                        print(f"Defaults must be within [{min_val}, {max_val}] and min <= max")
                                        default_min = validate_numeric_input(input(f"Enter default min value: "), min_label)
                                        if str(default_min).lower() == 'edit' and i > 0:
                                            i -= 1
                                            continue
                                        default_max = validate_numeric_input(input(f"Enter default max value: "), max_label)
                                        if str(default_max).lower() == 'edit' and i > 0:
                                            i -= 1
                                            continue
                                    settings.append({
                                        "label": f"{min_label} / {max_label}",
                                        "type": "multislider",
                                        "min": min_val,
                                        "max": max_val,
                                        "default": [default_min, default_max],
                                        "integer": annotation == 'int'
                                    })
                                    i += 2
                                    continue
                            print(f"\nConfiguring parameter: {ui_label}")
                            if annotation == 'bool':
                                print("Type: Boolean switch")
                                default_str = input("Enable by default? (y/n): ").strip().lower()
                                if default_str not in ['y', 'n']:
                                    print("Invalid input. Using default value: false")
                                    default_str = 'n'
                                # Prompt for switch dependencies
                                requires = {}
                                add_dep = input(f"Do you want to link this switch to another switch? (y/n): ").strip().lower() == 'y'
                                while add_dep:
                                    dep_label = input("  Enter label or alias of the switch to link to: ").strip()
                                    dep_val = input("  Required value for that switch (y/n): ").strip().lower() == 'y'
                                    if dep_label:
                                        requires[dep_label] = dep_val
                                    add_dep = input("  Link to another switch? (y/n): ").strip().lower() == 'y'
                                setting = {
                                    "label": ui_label,
                                    "type": "switch",
                                    "default": default_str == 'y'
                                }
                                if requires:
                                    setting["requires"] = requires
                                settings.append(setting)
                            elif annotation in ['int', 'float']:
                                print(f"Type: {'Integer' if annotation == 'int' else 'Float'} slider")
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
                                    "integer": annotation == 'int'
                                })
                            elif annotation == 'str':
                                print("Type: String selection")
                                while True:
                                    options_str = input(f"Enter comma-separated options for {ui_label}: ").strip()
                                    if not options_str:
                                        print("Error: Must provide at least one option")
                                        continue
                                    options = [opt.strip() for opt in options_str.split(',')]
                                    options = [opt for opt in options if opt]
                                    if not options:
                                        print("Error: No valid options provided")
                                        continue
                                    if len(set(options)) != len(options):
                                        print("Error: Duplicate options found")
                                        continue
                                    break
                                print("\nAvailable options:")
                                for idx, opt in enumerate(options, 1):
                                    print(f"{idx}. {opt}")
                                while True:
                                    default_str = input(f"Enter default option (1-{len(options)}): ").strip()
                                    try:
                                        default_idx = int(default_str) - 1
                                        if 0 <= default_idx < len(options):
                                            break
                                        print(f"Please enter a number between 1 and {len(options)}")
                                    except ValueError:
                                        print("Please enter a valid number")
                            i += 1
                        update_json_config(ui_name, settings, func_info['num_inputs'])
                        update_renderhook_maps(func_name, ui_name, param_mappings)
                        print(f"\nFunction '{func_name}' configured successfully.")
                    except Exception as e:
                        print(f"Error configuring '{func_name}': {e}")
                        passes_path = str(Path(__file__).parent / "passes.py")
                        func_info = extract_function_info(passes_path, func_name)
                        if not func_info:
                            print(f"Skipping unsupported or invalid function '{func_name}'.")
                            continue
                        ui_name = input(f"Enter UI name for '{func_name}' (or press Enter to use function name, or type 'skip' to skip): ").strip()
                        if ui_name.lower() == 'skip':
                            print(f"Skipping function '{func_name}'.")
                            continue
                        if ui_name.lower() == 'disable':
                            passes_path_obj = Path(__file__).parent / "passes.py"
                            with open(passes_path_obj, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                            for idx, line in enumerate(lines):
                                if line.strip().startswith(f"def {func_name}(") and "#globalignore" not in line:
                                    lines[idx] = line.rstrip() + " #globalignore\n"
                                    break
                            with open(passes_path_obj, "w", encoding="utf-8") as f:
                                f.writelines(lines)
                            print(f"Function '{func_name}' disabled with #globalignore.")
                            continue
                        param_mappings = {}
                        param_labels = []
                        j = 0
                        params_len = len(func_info['params'])
                        while j < params_len:
                            param = func_info['params'][j]
                            ui_label = input(f"UI label for parameter '{param['name']}' (or press Enter to use same name): ").strip()
                            if ui_label.lower() == 'edit' and j > 0:
                                j -= 1
                                continue
                            if ui_label and ui_label != param['name']:
                                param_mappings[ui_label] = param['name']
                                param_labels.append(ui_label)
                            else:
                                param_labels.append(param['name'])
                            j += 1
                        settings = []
                        i = 0
                        params = func_info['params']
                        while i < len(params):
                            param = params[i]
                            ui_label = param_labels[i] if i < len(param_labels) else param['name']
                            annotation = param['annotation']
                            if (
                                i + 1 < len(params)
                                and annotation in ['int', 'float']
                                and params[i + 1]['annotation'] == annotation
                                and 'min' in param['name'].lower()
                                and 'max' in params[i + 1]['name'].lower()
                            ):
                                print(f"\nDetected consecutive '{annotation}' parameters: '{param['name']}' and '{params[i+1]['name']}'")
                                use_multislider = input("Would you like to use a multislider for these? (y/n): ").strip().lower()
                                if use_multislider == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                if use_multislider == 'y':
                                    min_label = param_labels[i] if i < len(param_labels) else param['name']
                                    max_label = param_labels[i+1] if i+1 < len(param_labels) else params[i+1]['name']
                                    min_val = validate_numeric_input(input(f"Enter minimum value for {min_label}: "), min_label, is_min=True)
                                    if str(min_val).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    max_val = validate_numeric_input(input(f"Enter maximum value for {max_label}: "), max_label)
                                    if str(max_val).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    while max_val is not None and min_val is not None and max_val <= min_val:
                                        print(f"Maximum value must be greater than minimum ({min_val})")
                                        max_val = validate_numeric_input(input(f"Enter maximum value for {max_label}: "), max_label)
                                        if str(max_val).lower() == 'edit' and i > 0:
                                            i -= 1
                                            continue
                                    default_min = validate_numeric_input(input(f"Enter default min value: "), min_label)
                                    if str(default_min).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    default_max = validate_numeric_input(input(f"Enter default max value: "), max_label)
                                    if str(default_max).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    while (default_min is not None and default_max is not None) and (default_min < min_val or default_max > max_val or default_min > default_max):
                                        print(f"Defaults must be within [{min_val}, {max_val}] and min <= max")
                                        default_min = validate_numeric_input(input(f"Enter default min value: "), min_label)
                                        if str(default_min).lower() == 'edit' and i > 0:
                                            i -= 1
                                            continue
                                        default_max = validate_numeric_input(input(f"Enter default max value: "), max_label)
                                        if str(default_max).lower() == 'edit' and i > 0:
                                            i -= 1
                                            continue
                                    settings.append({
                                        "label": f"{min_label} / {max_label}",
                                        "type": "multislider",
                                        "min": min_val,
                                        "max": max_val,
                                        "default": [default_min, default_max],
                                        "integer": annotation == 'int'
                                    })
                                    i += 2
                                    continue
                            print(f"\nConfiguring parameter: {ui_label}")
                            if annotation == 'bool':
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
                            elif annotation in ['int', 'float']:
                                print(f"Type: {'Integer' if annotation == 'int' else 'Float'} slider")
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
                                    "integer": annotation == 'int'
                                })
                            elif annotation == 'str':
                                print("Type: String selection")
                                while True:
                                    options_str = input(f"Enter comma-separated options for {ui_label}: ").strip()
                                    if not options_str:
                                        print("Error: Must provide at least one option")
                                        continue
                                    options = [opt.strip() for opt in options_str.split(',')]
                                    options = [opt for opt in options if opt]
                                    if not options:
                                        print("Error: No valid options provided")
                                        continue
                                    if len(set(options)) != len(options):
                                        print("Error: Duplicate options found")
                                        continue
                                    break
                                print("\nAvailable options:")
                                for idx, opt in enumerate(options, 1):
                                    print(f"{idx}. {opt}")
                                while True:
                                    default_str = input(f"Enter default option (1-{len(options)}): ").strip()
                                    try:
                                        default_idx = int(default_str) - 1
                                        if 0 <= default_idx < len(options):
                                            break
                                        print(f"Please enter a number between 1 and {len(options)}")
                                    except ValueError:
                                        print("Please enter a valid number")
                            i += 1
            if choice == "1":
                missing = get_unimplemented_functions()
                if not missing:
                    print("No missing functions to configure.")
                    continue
                for func_name in missing:
                    print(f"\nConfiguring function: {func_name}")
                    try:
                        passes_path = str(Path(__file__).parent / "passes.py")
                        func_info = extract_function_info(passes_path, func_name)
                        if not func_info:
                            print(f"Skipping unsupported or invalid function '{func_name}'.")
                            continue
                        ui_name = input(f"Enter UI name for '{func_name}' (or press Enter to use function name, or type 'skip' to skip): ").strip()
                        if ui_name.lower() == 'skip':
                            print(f"Skipping function '{func_name}'.")
                            continue
                        if ui_name.lower() == 'disable':
                            passes_path_obj = Path(__file__).parent / "passes.py"
                            with open(passes_path_obj, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                            for idx, line in enumerate(lines):
                                if line.strip().startswith(f"def {func_name}(") and "#globalignore" not in line:
                                    lines[idx] = line.rstrip() + " #globalignore\n"
                                    break
                            with open(passes_path_obj, "w", encoding="utf-8") as f:
                                f.writelines(lines)
                            print(f"Function '{func_name}' disabled with #globalignore.")
                            continue
                        param_mappings = {}
                        param_labels = []
                        j = 0
                        params_len = len(func_info['params'])
                        while j < params_len:
                            param = func_info['params'][j]
                            ui_label = input(f"UI label for parameter '{param['name']}' (or press Enter to use same name): ").strip()
                            if ui_label.lower() == 'edit' and j > 0:
                                j -= 1
                                continue
                            if ui_label and ui_label != param['name']:
                                param_mappings[ui_label] = param['name']
                                param_labels.append(ui_label)
                            else:
                                param_labels.append(param['name'])
                            j += 1
                        settings = []
                        i = 0
                        params = func_info['params']
                        while i < len(params):
                            param = params[i]
                            ui_label = param_labels[i] if i < len(param_labels) else param['name']
                            annotation = param['annotation']
                            if (
                                i + 1 < len(params)
                                and annotation in ['int', 'float']
                                and params[i + 1]['annotation'] == annotation
                                and 'min' in param['name'].lower()
                                and 'max' in params[i + 1]['name'].lower()
                            ):
                                print(f"\nDetected consecutive '{annotation}' parameters: '{param['name']}' and '{params[i+1]['name']}'")
                                use_multislider = input("Would you like to use a multislider for these? (y/n): ").strip().lower()
                                if use_multislider == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                if use_multislider == 'y':
                                    min_label = param_labels[i] if i < len(param_labels) else param['name']
                                    max_label = param_labels[i+1] if i+1 < len(param_labels) else params[i+1]['name']
                                    min_val = validate_numeric_input(input(f"Enter minimum value for {min_label}: "), min_label, is_min=True)
                                    if str(min_val).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    max_val = validate_numeric_input(input(f"Enter maximum value for {max_label}: "), max_label)
                                    if str(max_val).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    while max_val is not None and min_val is not None and max_val <= min_val:
                                        print(f"Maximum value must be greater than minimum ({min_val})")
                                        max_val = validate_numeric_input(input(f"Enter maximum value for {max_label}: "), max_label)
                                        if str(max_val).lower() == 'edit' and i > 0:
                                            i -= 1
                                            continue
                                    default_min = validate_numeric_input(input(f"Enter default min value: "), min_label)
                                    if str(default_min).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    default_max = validate_numeric_input(input(f"Enter default max value: "), max_label)
                                    if str(default_max).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    while (default_min is not None and default_max is not None) and (default_min < min_val or default_max > max_val or default_min > default_max):
                                        print(f"Defaults must be within [{min_val}, {max_val}] and min <= max")
                                        default_min = validate_numeric_input(input(f"Enter default min value: "), min_label)
                                        if str(default_min).lower() == 'edit' and i > 0:
                                            i -= 1
                                            continue
                                        default_max = validate_numeric_input(input(f"Enter default max value: "), max_label)
                                        if str(default_max).lower() == 'edit' and i > 0:
                                            i -= 1
                                            continue
                                    settings.append({
                                        "label": f"{min_label} / {max_label}",
                                        "type": "multislider",
                                        "min": min_val,
                                        "max": max_val,
                                        "default": [default_min, default_max],
                                        "integer": annotation == 'int'
                                    })
                                    i += 2
                                    continue
                            print(f"\nConfiguring parameter: {ui_label}")
                            if annotation == 'bool':
                                default_str = input("Enable by default? (y/n): ").strip().lower()
                                if default_str == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                if default_str not in ['y', 'n']:
                                    default_str = 'n'
                                settings.append({"label": ui_label, "type": "switch", "default": default_str == 'y'})
                            elif annotation in ['int', 'float']:
                                min_val = validate_numeric_input(input(f"Enter minimum value for {ui_label}: "), ui_label, is_min=True)
                                if str(min_val).lower() == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                max_val = validate_numeric_input(input(f"Enter maximum value for {ui_label}: "), ui_label)
                                if str(max_val).lower() == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                while max_val is not None and min_val is not None and max_val <= min_val:
                                    print(f"Maximum value must be greater than minimum ({min_val})")
                                    max_val = validate_numeric_input(input(f"Enter maximum value for {ui_label}: "), ui_label)
                                    if str(max_val).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                default = validate_numeric_input(input(f"Enter default value for {ui_label}: "), ui_label)
                                if str(default).lower() == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                while default is not None and (default < min_val or default > max_val):
                                    print(f"Default must be between {min_val} and {max_val}")
                                    default = validate_numeric_input(input(f"Enter default value for {ui_label}: "), ui_label)
                                    if str(default).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                settings.append({"label": ui_label, "type": "slider", "min": min_val, "max": max_val, "default": default, "integer": annotation == 'int'})
                            elif annotation == 'str':
                                options_str = input(f"Enter comma-separated options for {ui_label}: ").strip()
                                if options_str.lower() == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                options = [opt.strip() for opt in options_str.split(',') if opt.strip()]
                                if not options:
                                    options = ['option1']
                                default = options[0]
                                settings.append({"label": ui_label, "type": "dropdown", "options": options, "default": default})
                            i += 1
                        update_json_config(ui_name, settings, func_info['num_inputs'])
                        update_renderhook_maps(func_name, ui_name, param_mappings)
                        print(f"\nFunction '{func_name}' configured successfully.")
                    except Exception as e:
                        print(f"Error configuring '{func_name}': {e}")
                    again = input("\nWould you like to configure another missing function? (y/n): ").strip().lower()
                    if again != 'y':
                        break
            elif choice == "2":
                while True:
                    func_name = input("Enter the name of the function to configure (or 'back' to return): ").strip()
                    if func_name.lower() in ['back','exit','quit','escape','esc']:
                        break
                    passes_path = str(Path(__file__).parent / "passes.py")
                    func_info = extract_function_info(passes_path, func_name)
                    if not func_info:
                        print(f"Skipping unsupported or invalid function '{func_name}'.")
                        continue
                    ui_name = input(f"Enter UI name for '{func_name}' (or press Enter to use function name, or type 'skip' to skip): ").strip()
                    if ui_name.lower() == 'skip':
                        print(f"Skipping function '{func_name}'.")
                        continue
                    if ui_name.lower() == 'disable':
                        passes_path_obj = Path(__file__).parent / "passes.py"
                        with open(passes_path_obj, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        for idx, line in enumerate(lines):
                            if line.strip().startswith(f"def {func_name}(") and "#globalignore" not in line:
                                lines[idx] = line.rstrip() + " #globalignore\n"
                                break
                        with open(passes_path_obj, "w", encoding="utf-8") as f:
                            f.writelines(lines)
                        print(f"Function '{func_name}' disabled with #globalignore.")
                        continue
                    param_mappings = {}
                    param_labels = []
                    j = 0
                    params_len = len(func_info['params'])
                    while j < params_len:
                        param = func_info['params'][j]
                        ui_label = input(f"UI label for parameter '{param['name']}' (or press Enter to use same name): ").strip()
                        if ui_label.lower() == 'edit' and j > 0:
                            j -= 1
                            continue
                        if ui_label and ui_label != param['name']:
                            param_mappings[ui_label] = param['name']
                            param_labels.append(ui_label)
                        else:
                            param_labels.append(param['name'])
                        j += 1
                    settings = []
                    i = 0
                    params = func_info['params']
                    while i < len(params):
                        param = params[i]
                        ui_label = param_labels[i] if i < len(param_labels) else param['name']
                        annotation = param['annotation']
                        if (
                            i + 1 < len(params)
                            and annotation in ['int', 'float']
                            and params[i + 1]['annotation'] == annotation
                            and 'min' in param['name'].lower()
                            and 'max' in params[i + 1]['name'].lower()
                        ):
                            print(f"\nDetected consecutive '{annotation}' parameters: '{param['name']}' and '{params[i+1]['name']}'")
                            use_multislider = input("Would you like to use a multislider for these? (y/n): ").strip().lower()
                            if use_multislider == 'edit' and i > 0:
                                i -= 1
                                continue
                            if use_multislider == 'y':
                                min_label = param_labels[i] if i < len(param_labels) else param['name']
                                max_label = param_labels[i+1] if i+1 < len(param_labels) else params[i+1]['name']
                                min_val = validate_numeric_input(input(f"Enter minimum value for {min_label}: "), min_label, is_min=True)
                                if str(min_val).lower() == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                max_val = validate_numeric_input(input(f"Enter maximum value for {max_label}: "), max_label)
                                if str(max_val).lower() == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                while max_val is not None and min_val is not None and max_val <= min_val:
                                    print(f"Maximum value must be greater than minimum ({min_val})")
                                    max_val = validate_numeric_input(input(f"Enter maximum value for {max_label}: "), max_label)
                                    if str(max_val).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                default_min = validate_numeric_input(input(f"Enter default min value: "), min_label)
                                if str(default_min).lower() == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                default_max = validate_numeric_input(input(f"Enter default max value: "), max_label)
                                if str(default_max).lower() == 'edit' and i > 0:
                                    i -= 1
                                    continue
                                while (default_min is not None and default_max is not None) and (default_min < min_val or default_max > max_val or default_min > default_max):
                                    print(f"Defaults must be within [{min_val}, {max_val}] and min <= max")
                                    default_min = validate_numeric_input(input(f"Enter default min value: "), min_label)
                                    if str(default_min).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                    default_max = validate_numeric_input(input(f"Enter default max value: "), max_label)
                                    if str(default_max).lower() == 'edit' and i > 0:
                                        i -= 1
                                        continue
                                settings.append({
                                    "label": f"{min_label} / {max_label}",
                                    "type": "multislider",
                                    "min": min_val,
                                    "max": max_val,
                                    "default": [default_min, default_max],
                                    "integer": annotation == 'int'
                                })
                                i += 2
                                continue
                        print(f"\nConfiguring parameter: {ui_label}")
                        if annotation == 'bool':
                            default_str = input("Enable by default? (y/n): ").strip().lower()
                            if default_str == 'edit' and i > 0:
                                i -= 1
                                continue
                            if default_str not in ['y', 'n']:
                                default_str = 'n'
                            settings.append({"label": ui_label, "type": "switch", "default": default_str == 'y'})
                        elif annotation in ['int', 'float']:
                            min_val = validate_numeric_input(input(f"Enter minimum value for {ui_label}: "), ui_label, is_min=True)
                            if str(min_val).lower() == 'edit' and i > 0:
                                i -= 1
                                continue
                            max_val = validate_numeric_input(input(f"Enter maximum value for {ui_label}: "), ui_label)
                            if str(max_val).lower() == 'edit' and i > 0:
                                i -= 1
                                continue
                            while max_val is not None and min_val is not None and max_val <= min_val:
                                print(f"Maximum value must be greater than minimum ({min_val})")
                                max_val = validate_numeric_input(input(f"Enter maximum value for {ui_label}: "), ui_label)
                                if str(max_val).lower() == 'edit' and i > 0:
                                    i -= 1
                                    continue
                            default = validate_numeric_input(input(f"Enter default value for {ui_label}: "), ui_label)
                            if str(default).lower() == 'edit' and i > 0:
                                i -= 1
                                continue
                            while default is not None and (default < min_val or default > max_val):
                                print(f"Default must be between {min_val} and {max_val}")
                                default = validate_numeric_input(input(f"Enter default value for {ui_label}: "), ui_label)
                                if str(default).lower() == 'edit' and i > 0:
                                    i -= 1
                                    continue
                            settings.append({"label": ui_label, "type": "slider", "min": min_val, "max": max_val, "default": default, "integer": annotation == 'int'})
                        elif annotation == 'str':
                            options_str = input(f"Enter comma-separated options for {ui_label}: ").strip()
                            if options_str.lower() == 'edit' and i > 0:
                                i -= 1
                                continue
                            options = [opt.strip() for opt in options_str.split(',') if opt.strip()]
                            if not options:
                                options = ['option1']
                            default = options[0]
                            settings.append({"label": ui_label, "type": "dropdown", "options": options, "default": default})
                        i += 1
                    update_json_config(ui_name, settings, func_info['num_inputs'])
                    update_renderhook_maps(func_name, ui_name, param_mappings)
                    print(f"\nFunction '{func_name}' configured successfully.")
            elif choice == "3":
                print("Exiting.")
                import sys
                sys.exit(0)
            elif choice == "4":
                show_quick_tutorial()
            else:
                print("Invalid choice. Please select 1, 2, 3, or 4.")
            print("Exiting.")
            break

def show_quick_tutorial():
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"
    print(f"\n{BOLD}{CYAN}==== Quick Tutorial: Render Pass Settings Tool ===={RESET}")
    print(f"\n{BOLD}This tool helps you configure render pass settings for your image processing functions.{RESET}")

    print(f"\n{UNDERLINE}{YELLOW}Supported Parameter Types (for your function in passes.py){RESET}")
    print(f"{GREEN}- int{RESET}: Integer value (slider)")
    print(f"{GREEN}- float{RESET}: Floating point value (slider)")
    print(f"{GREEN}- bool{RESET}: Boolean (switch/toggle)")
    print(f"{GREEN}- str{RESET}: String (dropdown or radio selection)")
    print(f"{GREEN}- Image.Image{RESET}: Image input (required for at least one parameter)")

    print(f"\n{UNDERLINE}{YELLOW}Widget Types in the UI{RESET}")
    print(f"{MAGENTA}- slider{RESET}: For int/float values (with min, max, default)")
    print(f"{MAGENTA}- multislider{RESET}: For paired min/max values (e.g., range sliders)")
    print(f"{MAGENTA}- switch{RESET}: For bool values (on/off)")
    print(f"{MAGENTA}- dropdown{RESET}: For str values (choose from options)")
    print(f"{MAGENTA}- radio{RESET}: Group of mutually exclusive options")

    print(f"\n{UNDERLINE}{YELLOW}Special Commands (usable at any input prompt){RESET}")
    print(f"{BOLD}- edit{RESET}: Go back and modify the last entered value or step.")
    print(f"{BOLD}- back{RESET}: Return to the previous menu or cancel current input.")
    print(f"{BOLD}- skip{RESET}: Skip the current function or parameter.")
    print(f"{BOLD}- disable{RESET}: Disable a function by adding #globalignore in passes.py.")
    print(f"{BOLD}- exit{RESET}: Exit the tool immediately (at main menu).\n")

    print(f"{UNDERLINE}{YELLOW}Main Functions of This Tool{RESET}")
    print(f"1. Configure missing functions: Finds functions in passes.py not yet in renderPasses.json and guides you through setup.")
    print(f"2. Configure any function: Lets you set up or edit settings for any function by name.")
    print(f"3. Exit: Quit the tool.")
    print(f"4. Quick tutorial: Show this help message.")

    print(f"{UNDERLINE}{YELLOW}Tips{RESET}")
    print(f"- Your function must have type hints for all parameters and return type (Image.Image).\n- For two-image functions, use 'Mix' or 'Subtract' in the UI name for clarity.")

    print(f"{UNDERLINE}{YELLOW}Example Function{RESET}")
    print(f"{CYAN}def my_effect(img: Image.Image, strength: float, mode: str) -> Image.Image:{RESET}")
    print(f"    ... # your code ...")
    print(f"\nThis will let you configure a slider for 'strength' and a dropdown or radio for 'mode'.")
    print(f"\n{BOLD}{CYAN}==================================================={RESET}\n")

# --- Automated settings generator ---
def auto_generate_settings_from_passes():
    """
    Scans passes.py for functions, extracts parameters and types, and updates renderPasses.json
    with default settings for each function (if not already present). Manual edits in renderPasses.json
    are always preserved. Only missing functions are added.
    """
    from pathlib import Path
    import ast

    passes_path = Path(__file__).parent / "passes.py"
    json_path = Path(__file__).parent.parent / "renderPasses.json"

    # Load existing config
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception:
        config = {}

    # Parse passes.py
    with open(passes_path, 'r', encoding='utf-8') as f:
        content = f.read()
    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            # Skip private/internal functions
            if func_name.startswith('_'):
                continue
            # Use function name as UI name
            ui_name = func_name
            if ui_name in config:
                continue  # Don't overwrite manual edits

            # Extract parameters
            params = []
            image_params = []
            has_mask = False
            for arg in node.args.args:
                if arg.arg == 'self':
                    continue
                annotation = arg.annotation
                type_hint = None
                if isinstance(annotation, ast.Name):
                    type_hint = annotation.id
                elif isinstance(annotation, ast.Attribute):
                    if isinstance(annotation.value, ast.Name) and annotation.value.id == 'Image' and annotation.attr == 'Image':
                        type_hint = 'Image.Image'
                if not type_hint:
                    continue
                if type_hint == 'Image.Image':
                    if arg.arg == 'mask':
                        has_mask = True
                    else:
                        image_params.append(arg.arg)
                    continue
                params.append({'name': arg.arg, 'annotation': type_hint})

            num_inputs = len(image_params)
            # Map types to widget types
            settings = []
            for param in params:
                annotation = param['annotation']
                name = param['name']
                if annotation == 'bool':
                    widget_type = 'switch'
                    default = False
                    settings.append({"label": name, "type": widget_type, "default": default})
                elif annotation == 'int':
                    widget_type = 'slider'
                    settings.append({"label": name, "type": widget_type, "min": 0, "max": 100, "default": 0, "integer": True})
                elif annotation == 'float':
                    widget_type = 'slider'
                    settings.append({"label": name, "type": widget_type, "min": 0.0, "max": 1.0, "default": 0.0, "integer": False})
                elif annotation == 'str':
                    widget_type = 'dropdown'
                    settings.append({"label": name, "type": widget_type, "options": ["option1", "option2"], "default": "option1"})

            # Add category and num_inputs
            category = ui_name.lower().replace(" ", "_")
            settings.insert(0, {"kategory": category, "num_inputs": num_inputs})
            config[ui_name] = settings

    # Write back only if new functions were added
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print("Settings auto-generated for missing functions. Manual edits in renderPasses.json are always preserved.")



if __name__ == "__main__":
    print("\033c",end='')  # Clear console

    interactive_menu()
