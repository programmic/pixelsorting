from typing import List, Dict, Any, Optional, Tuple
import json
import sys
import ast
from pathlib import Path
import re

# --- Configuration / rules ---
FUNCTION_RULES = {
    "name_based_rules": [
        {"pattern": r"subtract", "case_sensitive": False, "requirements": {"exact_image_inputs": 2, "error_message": "Functions with 'subtract' in the name must have exactly 2 image inputs"}},
        {"pattern": r"mix", "case_sensitive": False, "requirements": {"exact_image_inputs": 2, "error_message": "Functions with 'mix' in the name must have exactly 2 image inputs"}},
        {"pattern": r"mask", "case_sensitive": False, "requirements": {"requires_mask": True, "error_message": "Functions with 'mask' in the name must accept a mask parameter"}}
    ],
    "general_rules": {"max_image_inputs": 2, "min_image_inputs": 1, "allowed_types": ["int", "float", "bool", "str", "Image.Image"], "required_return_type": "Image.Image"}
}


def validate_function_rules(func_name: str, num_image_inputs: int, has_mask: bool, param_names: List[str]) -> Tuple[bool, str]:
    rules = FUNCTION_RULES
    # Check general rules
    gen = rules.get("general_rules", {})
    max_inputs = gen.get("max_image_inputs")
    min_inputs = gen.get("min_image_inputs")
    if max_inputs is not None and num_image_inputs > max_inputs:
        return False, f"Function '{func_name}' has too many image inputs: {num_image_inputs} > {max_inputs}"
    if min_inputs is not None and num_image_inputs < min_inputs:
        return False, f"Function '{func_name}' does not have enough image inputs: {num_image_inputs} < {min_inputs}"

    # Name-based rules
    for rule in rules.get("name_based_rules", []):
        pattern = rule.get("pattern")
        case_sensitive = rule.get("case_sensitive", True)
        reqs = rule.get("requirements", {})
        flags = 0 if case_sensitive else re.IGNORECASE
        if pattern and re.search(pattern, func_name, flags=flags):
            # exact image inputs
            exact = reqs.get("exact_image_inputs")
            if exact is not None and num_image_inputs != exact:
                return False, reqs.get("error_message", f"Function '{func_name}' doesn't meet rule {pattern}")
            # requires mask
            if reqs.get("requires_mask") and not has_mask:
                return False, reqs.get("error_message", f"Function '{func_name}' must accept a mask parameter")
    return True, ""


def update_renderhook_maps(func_name: str, ui_name: str, param_mappings: Dict[str, str]) -> None:
    """Updates `renderHook.py` mappings by inserting simple dict entries.

    This is a best-effort string-based update; it won't fully parse complex Python files.
    """
    render_hook_path = Path(__file__).parent / "renderHook.py"
    if not render_hook_path.exists():
        print(f"Warning: {render_hook_path} not found; skipping renderHook update")
        return
    text = render_hook_path.read_text(encoding='utf-8')

    if 'func_name_map' in text:
        entry = f'    "{ui_name}": "{func_name}",\n'
        text = text.replace('func_name_map = {', 'func_name_map = {\n' + entry, 1)

    if param_mappings and 'setting_name_map' in text:
        entries = ''.join(f'    "{k}": "{v}",\n' for k, v in param_mappings.items())
        text = text.replace('setting_name_map = {', 'setting_name_map = {\n' + entries, 1)

    render_hook_path.write_text(text, encoding='utf-8')


def update_json_config(ui_name: str, settings: List[Dict[str, Any]], num_inputs: int) -> None:
    """Updates `renderPasses.json` with the provided settings for a UI name."""
    json_path = Path(__file__).parent.parent / 'renderPasses.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if not json_path.exists():
        json_path.write_text('{}', encoding='utf-8')
    try:
        config = json.loads(json_path.read_text(encoding='utf-8') or '{}')
    except Exception:
        config = {}

    if ui_name in config:
        print(f"Replacing existing pass '{ui_name}' in {json_path}")
        del config[ui_name]

    func_alias = ui_name.lower().replace(' ', '_')
    original_func_name = None
    if settings and isinstance(settings[-1], dict) and 'original_func_name' in settings[-1]:
        original_func_name = settings[-1].pop('original_func_name')

    pass_dict = {'original_func_name': original_func_name or ui_name, 'num_inputs': num_inputs, 'function_alias': func_alias, 'settings': []}
    for s in settings:
        # prefer explicit label, fallback to name or alias
        label = s.get('label') or s.get('name') or s.get('alias')
        alias = s.get('alias') or (str(label).lower().replace(' ', '_') if label else '')
        sub = {'label': label, 'alias': alias, 'type': s.get('type'), 'default': s.get('default')}
        for key in ('min', 'max', 'integer', 'options', 'availableSlots'):
            if key in s:
                sub[key] = s[key]
        # normalize requirement key to 'requires' (used internally)
        if 'requires' in s:
            sub['requires'] = s['requires']
        elif 'requirements' in s:
            sub['requires'] = s['requirements']
        pass_dict['settings'].append(sub)

    config[ui_name] = pass_dict
    json_path.write_text(json.dumps(config, indent=2), encoding='utf-8')


def validate_type_hint(annotation: ast.AST) -> Optional[str]:
    """Return a simplified string type for supported annotations, or None if unsupported."""
    if annotation is None:
        return None
    ann = None
    try:
        ann = ast.unparse(annotation)
    except Exception:
        try:
            # fallback for simple Name nodes
            if isinstance(annotation, ast.Name):
                ann = annotation.id
        except Exception:
            ann = None
    if not ann:
        return None
    ann = ann.replace('typing.', '')
    if ann.startswith('Optional['):
        inner = ann[len('Optional['):-1]
        if inner == 'Image.Image':
            return 'Optional[Image.Image]'
        return inner
    if ann in ('Image.Image', 'int', 'float', 'bool', 'str'):
        return ann
    print(f"\nError: Unsupported type hint: {ann}")
    print("Supported types: Image.Image, int, float, bool, str, Optional[Image.Image]")
    return None


def extract_function_info(func_path: str, func_name: str) -> Optional[Dict[str, Any]]:
    """Analyze a function in a Python file using AST and return metadata used to generate UI settings.

    Returns a dict with keys: num_inputs, has_mask, params (list of {name, annotation}),
    and returns True/False for validity.
    """
    try:
        path = Path(func_path)
        src = path.read_text(encoding='utf-8')
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                params = []
                num_image_inputs = 0
                has_mask = False
                for arg in node.args.args:
                    if arg.arg == 'self':
                        continue
                    ann = validate_type_hint(arg.annotation)
                    params.append({'name': arg.arg, 'annotation': ann})
                    if ann == 'Image.Image' or ann == 'Optional[Image.Image]':
                        num_image_inputs += 1
                    if arg.arg.lower() == 'mask':
                        has_mask = True
                # Return type check
                return_ann = validate_type_hint(node.returns)
                ok, msg = validate_function_rules(func_name, num_image_inputs, has_mask, [p['name'] for p in params])
                if not ok:
                    print(f"Function '{func_name}' failed validation: {msg}")
                    return None
                return {'num_inputs': num_image_inputs, 'has_mask': has_mask, 'params': params, 'return_ann': return_ann}
    except Exception as e:
        print("\n\033[91m==============================\033[0m")
        print(f"\033[91mError analyzing function: {e}\033[0m")
        print("\033[91mMake sure your function:\033[0m")
        print("\033[91m1. Has proper type hints\033[0m")
        print("\033[91m2. Uses supported parameter types\033[0m")
        print("\033[91m3. Has correct syntax\033[0m")
        print("\033[91m==============================\033[0m\n")
        return None


# --- Interactive helpers ---

def _prompt_label_for_param(param_name: str) -> str:
    lbl = input(f"UI label for parameter '{param_name}' (press Enter to use '{param_name}'): ").strip()
    return lbl or param_name


def _prompt_requirements(ui_label: str) -> Optional[Dict[str, str]]:
    """Prompt the user to add requirements (dependencies) for a setting.

    Returns a dict mapping dependent label -> required value/expression, or None if none.
    """
    requires: Dict[str, str] = {}
    add_req = input(f"Add requirement to show/change '{ui_label}'? (y/n): ").strip().lower() == 'y'
    while add_req:
        dep = input("  Enter dependent parameter name or alias: ").strip()
        logic = input("  Enter required value/expression (e.g. true/false, >10): ").strip()
        if dep and logic:
            requires[dep] = logic
        add_req = input("  Add another requirement? (y/n): ").strip().lower() == 'y'
    return requires if requires else None


def _build_settings_from_params(params: List[Dict[str, Any]], param_labels: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    settings: List[Dict[str, Any]] = []
    param_labels = param_labels or {}
    i = 0
    while i < len(params):
        p = params[i]
        name = p['name']
        annotation = p['annotation']
        # paired min/max -> multislider
        if i + 1 < len(params) and annotation in ('int', 'float') and params[i + 1]['annotation'] == annotation and 'min' in name.lower() and 'max' in params[i + 1]['name'].lower():
            # Use provided labels if available, otherwise prompt (but avoid prompting for image inputs)
            min_label = param_labels.get(name) or _prompt_label_for_param(name)
            max_label = param_labels.get(params[i + 1]['name']) or _prompt_label_for_param(params[i + 1]['name'])
            use_multi = input(f"Detected range pair '{name}'/'{params[i+1]['name']}'. Use multislider? (y/n): ").strip().lower() == 'y'
            if use_multi:
                while True:
                    min_v = validate_numeric_input(input(f"Min for '{min_label}': "), min_label, is_min=True)
                    if min_v is not None:
                        break
                while True:
                    max_v = validate_numeric_input(input(f"Max for '{max_label}': "), max_label)
                    if max_v is not None and max_v > min_v:
                        break
                    print(f"Maximum must be greater than minimum ({min_v})")
                while True:
                    default_min = validate_numeric_input(input(f"Default min for '{min_label}': "), min_label)
                    default_max = validate_numeric_input(input(f"Default max for '{max_label}': "), max_label)
                    if default_min is not None and default_max is not None and min_v <= default_min <= default_max <= max_v:
                        break
                    print(f"Defaults must be within [{min_v}, {max_v}] and min <= max")
                settings.append({"label": f"{min_label} / {max_label}", "type": "multislider", "min": min_v, "max": max_v, "default": [default_min, default_max], "integer": annotation == 'int'})
                i += 2
                continue
        # Skip prompting for Image inputs (they are not UI settings)
        if annotation in ('Image.Image', 'Optional[Image.Image]'):
            i += 1
            continue

        ui_label = param_labels.get(name) or _prompt_label_for_param(name)
        if annotation == 'bool':
            default = input(f"Default for '{ui_label}' (y/n, default n): ").strip().lower() == 'y'
            setting = {"label": ui_label, "type": "switch", "default": default}
            reqs = _prompt_requirements(ui_label)
            if reqs:
                setting["requires"] = reqs
            settings.append(setting)
        elif annotation in ('int', 'float'):
            while True:
                min_v = validate_numeric_input(input(f"Min for '{ui_label}': "), ui_label, is_min=True)
                if min_v is not None:
                    break
            while True:
                max_v = validate_numeric_input(input(f"Max for '{ui_label}': "), ui_label)
                if max_v is not None and max_v > min_v:
                    break
                print(f"Maximum must be greater than minimum ({min_v})")
            while True:
                default = validate_numeric_input(input(f"Default for '{ui_label}': "), ui_label)
                if default is not None and min_v <= default <= max_v:
                    break
                print(f"Default must be between {min_v} and {max_v}")
            setting = {"label": ui_label, "type": "slider", "min": min_v, "max": max_v, "default": default, "integer": annotation == 'int'}
            reqs = _prompt_requirements(ui_label)
            if reqs:
                setting["requires"] = reqs
            settings.append(setting)
        elif annotation == 'str':
            opts = []
            while not opts:
                opts_input = input(f"Comma-separated options for '{ui_label}': ").strip()
                opts = [o.strip() for o in opts_input.split(',') if o.strip()]
                if not opts:
                    print("Please provide at least one option")
            default = opts[0]
            setting = {"label": ui_label, "type": "dropdown", "options": opts, "default": default}
            reqs = _prompt_requirements(ui_label)
            if reqs:
                setting["requires"] = reqs
            settings.append(setting)
        i += 1
    return settings


def ensure_valid_json(json_path):
    p = Path(json_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.stat().st_size == 0:
        p.write_text('{}', encoding='utf-8')


def get_unimplemented_functions() -> List[str]:
    """Return a list of function names in 'passes.py' that are not present in renderPasses.json"""
    passes_path = Path(__file__).parent / 'passes.py'
    if not passes_path.exists():
        return []
    try:
        src = passes_path.read_text(encoding='utf-8')
        tree = ast.parse(src)
        funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    except Exception:
        return []
    json_path = Path(__file__).parent.parent / 'renderPasses.json'
    try:
        config = json.loads(json_path.read_text(encoding='utf-8') or '{}')
    except Exception:
        config = {}
    existing = set(config.keys())
    return [f for f in funcs if f not in existing]


def show_quick_tutorial():
    print("\nQuick tutorial:\n- Functions should accept image inputs with type 'Image.Image'\n- Supported parameter types: int, float, bool, str\n- Use 'min' and 'max' parameter name pairs to produce multisliders\n- After configuration, entries are written to 'renderPasses.json' and 'renderHook.py'\n")


def auto_generate_settings_from_passes():
    """Attempt to generate settings for all functions in passes.py without prompting.

    This provides a baseline auto-generation using defaults and basic mapping rules.
    """
    passes_path = Path(__file__).parent / 'passes.py'
    if not passes_path.exists():
        print("passes.py not found")
        return
    src = passes_path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            info = extract_function_info(str(passes_path), func_name)
            if not info:
                continue
            settings = []
            for p in info['params']:
                ann = p['annotation']
                if ann == 'bool':
                    settings.append({'label': p['name'], 'type': 'switch', 'default': False, 'name': p['name']})
                elif ann in ('int', 'float'):
                    settings.append({'label': p['name'], 'type': 'slider', 'min': 0, 'max': 100, 'default': 50, 'name': p['name']})
                elif ann == 'str':
                    settings.append({'label': p['name'], 'type': 'dropdown', 'options': ['option1'], 'default': 'option1', 'name': p['name']})
            settings.append({'original_func_name': func_name})
            update_json_config(func_name, settings, info['num_inputs'])
    print("Auto-generation completed.")


def validate_numeric_input(value: str, label: str, is_min: bool = False) -> Optional[float]:
    try:
        if '.' in value:
            v = float(value)
        else:
            v = int(value)
        return v
    except Exception:
        if isinstance(value, str) and value.lower() == 'edit':
            return value
        print(f"Invalid numeric input for {label}")
        return None


def interactive_menu():
    while True:
        print("\n==== Render Pass Settings Menu ====")
        print("1. Configure missing functions")
        print("2. Configure any function by name")
        print("3. Exit")
        print("4. Quick tutorial (inputs, widgets, usage)")
        choice = input("Select option (1-4): ").strip()
        if choice == '1':
            missing = get_unimplemented_functions() or []
            if not missing:
                print("No missing functions to configure.")
                continue
            for func_name in missing:
                print(f"\nConfiguring function: {func_name}")
                func_info = extract_function_info(str(Path(__file__).parent / 'passes.py'), func_name)
                if not func_info:
                    print(f"Skipping invalid function: {func_name}")
                    continue
                ui_name = input(f"Enter UI name for '{func_name}' (press Enter to use '{func_name}', 'skip' to skip, 'disable' to add #globalignore): ").strip()
                if ui_name.lower() == 'skip':
                    continue
                if ui_name.lower() == 'disable':
                    passes_path_obj = Path(__file__).parent / 'passes.py'
                    with open(passes_path_obj, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    for idx, line in enumerate(lines):
                        if line.strip().startswith(f"def {func_name}(") and '#globalignore' not in line:
                            lines[idx] = line.rstrip() + ' #globalignore\n'
                            break
                    with open(passes_path_obj, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    print(f"Function '{func_name}' disabled.")
                    continue
                if not ui_name:
                    ui_name = func_name
                param_mappings: Dict[str, str] = {}
                # collect labels once, but skip Image inputs
                param_labels: Dict[str, str] = {}
                for p in func_info['params']:
                    if p['annotation'] in ('Image.Image', 'Optional[Image.Image]'):
                        continue
                    label = input(f"UI label for parameter '{p['name']}' (press Enter to keep): ").strip()
                    if label and label != p['name']:
                        param_mappings[label] = p['name']
                        param_labels[p['name']] = label
                settings = _build_settings_from_params(func_info['params'], param_labels)
                settings.append({'original_func_name': func_name})
                update_json_config(ui_name, settings, func_info['num_inputs'])
                update_renderhook_maps(func_name, ui_name, param_mappings)
                print(f"Configured '{ui_name}' -> {func_name}")
        elif choice == '2':
            func_name = input("Enter function name (or 'back' to return): ").strip()
            if func_name.lower() in ('back', 'exit', 'quit'):
                continue
            func_info = extract_function_info(str(Path(__file__).parent / 'passes.py'), func_name)
            if not func_info:
                print(f"Function '{func_name}' not found or invalid.")
                continue
            ui_name = input(f"Enter UI name for '{func_name}' (press Enter to use '{func_name}', 'disable' to add #globalignore): ").strip()
            if ui_name.lower() == 'disable':
                passes_path_obj = Path(__file__).parent / 'passes.py'
                with open(passes_path_obj, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for idx, line in enumerate(lines):
                    if line.strip().startswith(f"def {func_name}(") and '#globalignore' not in line:
                        lines[idx] = line.rstrip() + ' #globalignore\n'
                        break
                with open(passes_path_obj, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                print(f"Function '{func_name}' disabled.")
                continue
            if not ui_name:
                ui_name = func_name
            param_mappings: Dict[str, str] = {}
            param_labels: Dict[str, str] = {}
            for p in func_info['params']:
                if p['annotation'] in ('Image.Image', 'Optional[Image.Image]'):
                    continue
                label = input(f"UI label for parameter '{p['name']}' (press Enter to keep): ").strip()
                if label and label != p['name']:
                    param_mappings[label] = p['name']
                    param_labels[p['name']] = label
            settings = _build_settings_from_params(func_info['params'], param_labels)
            settings.append({'original_func_name': func_name})
            update_json_config(ui_name, settings, func_info['num_inputs'])
            update_renderhook_maps(func_name, ui_name, param_mappings)
            print(f"Configured '{ui_name}' -> {func_name}")
        elif choice == '3':
            print('Exiting.')
            sys.exit(0)
        elif choice == '4':
            show_quick_tutorial()
        else:
            print('Invalid choice. Please select 1-4')


if __name__ == '__main__':
    interactive_menu()
