"""
settings_generator.py
Procedural RenderPass Settings Tool

Features:
- Discover functions in project, analyze signatures and type hints
- Interactive CLI for parameter config, with -fix to move back
- Enum member prompting, writes to enums.json
- Batch/single mode selection at startup
- #globalignore support
- Output: replace, append, or merge with renderPasses.json
- Help system: --help flag and HELP menu option (with color/examples)
"""
import ast
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
except ImportError:
    # fallback if colorama not installed
    class Dummy:
        RESET = RED = GREEN = YELLOW = CYAN = MAGENTA = ''
    Fore = Style = Dummy()

PROJECT_ROOT = Path(__file__).parent.parent
ENUMS_PATH = PROJECT_ROOT / 'enums.json'
RENDERPASSES_PATH = PROJECT_ROOT / 'renderPasses.json'
SCRIPTS_PATH = PROJECT_ROOT / 'scripts'

HELP_TEXT = f"""
{Fore.CYAN}RenderPass Settings Tool Help{Style.RESET_ALL}

Modes:
  1. Single function: Configure one function interactively.
  2. Batch: Configure all unconfigured functions in the project.

Features:
  - Discovers functions and analyzes their parameters/type hints.
  - Prompts for all UI/config details for each parameter.
  - Enum support: prompts for members, writes to enums.json.
  - Output: replace, append, or merge with renderPasses.json.
  - #globalignore support.
  - At any prompt, type {Fore.YELLOW}-fix{Style.RESET_ALL} to move back one setting.
  - Help: {Fore.YELLOW}--help{Style.RESET_ALL} flag or {Fore.YELLOW}HELP{Style.RESET_ALL} menu option.

Examples:
  $ python settings_generator.py --help
  $ python settings_generator.py
"""

def print_help():
    print(HELP_TEXT)
    sys.exit(0)

def input_with_fix(prompt: str, history: List[Any]) -> str:
    shortcut_hint = f"{Fore.MAGENTA} (Type -fix to go back, -ignore to skip, -exit to quit, HELP for help){Style.RESET_ALL}"
    while True:
        val = input(prompt + shortcut_hint)
        val_strip = val.strip().lower()
        if val_strip == 'help' or val_strip == '--help':
            print_help()
            continue
        if val_strip == '-fix' and history:
            print(f"{Fore.YELLOW}Going back one step...{Style.RESET_ALL}")
            return '-fix'
        if val_strip == '-ignore':
            return '-ignore'
        if val_strip == '-exit':
            return '-exit'
        return val

def find_functions_in_project() -> List[Dict[str, Any]]:
    """Scan all .py files in scripts/ for function definitions."""
    funcs = []
    for pyfile in SCRIPTS_PATH.rglob('*.py'):
        try:
            src = pyfile.read_text(encoding='utf-8')
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    funcs.append({'name': node.name, 'file': pyfile, 'node': node})
        except Exception:
            continue
    return funcs

def get_function_signature(func: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of parameter dicts: name, annotation, default."""
    node = func['node']
    params = []
    for arg in node.args.args:
        if arg.arg == 'self':
            continue
        ann = None
        try:
            ann = ast.unparse(arg.annotation) if arg.annotation else None
        except Exception:
            ann = None
        params.append({'name': arg.arg, 'annotation': ann})
    return params

def prompt_enum(param_name: str) -> Dict[str, Any]:
    print(f"{Fore.CYAN}Parameter '{param_name}' requires an Enum.{Style.RESET_ALL}")
    members = []
    history = []
    while True:
        val = input_with_fix(f"Enter Enum member name (or blank to finish): ", history)
        if val == '-fix' and members:
            members.pop()
            continue
        if not val:
            break
        members.append(val)
        history.append(val)
    return {'name': param_name, 'members': members}

def write_enums_to_json(enums: List[Dict[str, Any]]):
    enums_dict = {}
    if ENUMS_PATH.exists():
        try:
            enums_dict = json.loads(ENUMS_PATH.read_text(encoding='utf-8'))
        except Exception:
            enums_dict = {}
    for enum in enums:
        enums_dict[enum['name']] = enum['members']
    ENUMS_PATH.write_text(json.dumps(enums_dict, indent=2), encoding='utf-8')
    print(f"{Fore.GREEN}Enums written to {ENUMS_PATH}{Style.RESET_ALL}")

def prompt_for_param(param: Dict[str, Any], history: List[Any]) -> Dict[str, Any]:
    name = param['name']
    ann = param['annotation']
    result = {'name': name}
    print(f"{Fore.CYAN}\n--- Configuring Parameter: {name} ---{Style.RESET_ALL}")
    while True:
        label = input_with_fix(f"Label for '{name}': ", history)
        if label == '-fix':
            return '-fix'
        if label == '-ignore':
            return '-ignore'
        if label == '-exit':
            return '-exit'
        label = label.strip()
        result['label'] = label if label else name
        break
    # Type selection
    allowed_types = ('int', 'float', 'bool', 'str', 'enum', 'range')
    while True:
        # Support union types like 'str | EnumType'
        if ann:
            ann_types = [a.strip().lower() for a in ann.split('|')]
            valid_types = [t for t in ann_types if t in allowed_types]
            if valid_types:
                # If union, prompt user to pick one
                if len(valid_types) > 1:
                    print(f"{Fore.YELLOW}Parameter '{name}' supports multiple types: {', '.join(valid_types)}{Style.RESET_ALL}")
                    type_hint = input_with_fix(f"Type for '{name}' (choose: {', '.join(valid_types)}): ", history)
                else:
                    type_hint = valid_types[0]
            else:
                type_hint = input_with_fix(f"Type for '{name}' (int, float, bool, str, enum, range): ", history)
        else:
            type_hint = input_with_fix(f"Type for '{name}' (int, float, bool, str, enum, range): ", history)
        if type_hint == '-fix':
            return '-fix'
        if type_hint == '-ignore':
            return '-ignore'
        if type_hint == '-exit':
            return '-exit'
        type_hint = type_hint.strip().lower()
        if type_hint not in allowed_types:
            print(f"{Fore.RED}Invalid type. Please enter one of: int, float, bool, str, enum, range.{Style.RESET_ALL}")
            continue
        result['type'] = type_hint
        break
    # Always prompt for options for string/enum
    if type_hint in ('str', 'enum'):
        while True:
            opts = input_with_fix(f"Comma-separated options for '{name}': ", history)
            if opts == '-fix':
                return '-fix'
            if opts == '-ignore':
                return '-ignore'
            if opts == '-exit':
                return '-exit'
            options = [o.strip() for o in opts.split(',') if o.strip()]
            if not options:
                print(f"{Fore.RED}Please enter at least one option.{Style.RESET_ALL}")
                continue
            result['options'] = options
            result['default'] = options[0]
            break
        # Prompt for UI type: dropdown or radio
        while True:
            ui_type = input_with_fix(f"UI type for '{name}' options (dropdown/radio): ", history)
            if ui_type == '-fix':
                return '-fix'
            if ui_type == '-ignore':
                return '-ignore'
            if ui_type == '-exit':
                return '-exit'
            ui_type = ui_type.strip().lower()
            if ui_type in ('dropdown', 'radio'):
                result['ui_type'] = ui_type
                break
            else:
                print(f"{Fore.YELLOW}Please enter 'dropdown' or 'radio'.{Style.RESET_ALL}")
        if type_hint == 'enum':
            write_enums_to_json([{'name': name, 'members': result['options']}])
    elif type_hint.lower() in ('int', 'float'):
        while True:
            min_val = input_with_fix(f"Min for '{name}': ", history)
            if min_val == '-fix':
                return '-fix'
            if min_val == '-ignore':
                return '-ignore'
            if min_val == '-exit':
                return '-exit'
            max_val = input_with_fix(f"Max for '{name}': ", history)
            if max_val == '-fix':
                return '-fix'
            if max_val == '-ignore':
                return '-ignore'
            if max_val == '-exit':
                return '-exit'
            default = input_with_fix(f"Default for '{name}': ", history)
            if default == '-fix':
                return '-fix'
            if default == '-ignore':
                return '-ignore'
            if default == '-exit':
                return '-exit'
            result['min'] = float(min_val)
            result['max'] = float(max_val)
            result['default'] = float(default)
            break
    elif type_hint.lower() == 'bool':
        while True:
            default = input_with_fix(f"Default for '{name}' (y/n/0/1): ", history)
            if default == '-fix':
                return '-fix'
            if default == '-ignore':
                return '-ignore'
            if default == '-exit':
                return '-exit'
            val = default.strip().lower()
            result['default'] = val in ('y', '1')
            break
    elif type_hint.lower() == 'range':
        while True:
            min_val = input_with_fix(f"Min for '{name}': ", history)
            if min_val == '-fix':
                return '-fix'
            if min_val == '-ignore':
                return '-ignore'
            if min_val == '-exit':
                return '-exit'
            max_val = input_with_fix(f"Max for '{name}': ", history)
            if max_val == '-fix':
                return '-fix'
            if max_val == '-ignore':
                return '-ignore'
            if max_val == '-exit':
                return '-exit'
            default_min = input_with_fix(f"Default min for '{name}': ", history)
            if default_min == '-fix':
                return '-fix'
            if default_min == '-ignore':
                return '-ignore'
            if default_min == '-exit':
                return '-exit'
            default_max = input_with_fix(f"Default max for '{name}': ", history)
            if default_max == '-fix':
                return '-fix'
            if default_max == '-ignore':
                return '-ignore'
            if default_max == '-exit':
                return '-exit'
            result['min'] = float(min_val)
            result['max'] = float(max_val)
            result['default'] = [float(default_min), float(default_max)]
            break
    # Requirements
    reqs = {}
    while True:
        add_req = input_with_fix(f"Add requirement for '{name}'? (y/n/0/1): ", history)
        if add_req == '-fix':
            return '-fix'
        if add_req == '-ignore':
            return '-ignore'
        if add_req == '-exit':
            return '-exit'
        val = add_req.strip().lower()
        if val not in ('y', '1'):
            break
        dep = input_with_fix("  Dependent parameter name: ", history)
        if dep == '-fix':
            return '-fix'
        if dep == '-ignore':
            return '-ignore'
        if dep == '-exit':
            return '-exit'
        logic = input_with_fix("  Required value/expression: ", history)
        if logic == '-fix':
            return '-fix'
        if logic == '-ignore':
            return '-ignore'
        if logic == '-exit':
            return '-exit'
        reqs[dep] = logic
    if reqs:
        result['requirements'] = reqs
    return result

def prompt_for_function(func: Dict[str, Any]) -> Dict[str, Any]:
    print(f"{Fore.GREEN}\n=== Configuring Function: {func['name']} ==={Style.RESET_ALL}")
    params = get_function_signature(func)
    settings = {}
    history = []
    display_name = input_with_fix("Display name for function: ", history)
    category = input_with_fix("Category (or blank): ", history)
    i = 0
    total_params = len(params)
    while i < total_params:
        param = params[i]
        print(f"{Fore.YELLOW}Parameter {i+1} of {total_params}{Style.RESET_ALL}")
        # Skip image params unless mask/optional
        if param['annotation'] and 'Image.Image' in param['annotation']:
            i += 1
            continue
        res = prompt_for_param(param, history)
        if res == '-fix' and i > 0:
            i -= 1
            continue
        elif res == '-fix':
            continue
        if res == '-ignore':
            # Add #globalignore to function in file
            file_path = func['file']
            src_lines = file_path.read_text(encoding='utf-8').splitlines()
            for idx, line in enumerate(src_lines):
                if line.strip().startswith(f"def {func['name']}(") and '#globalignore' not in line:
                    src_lines[idx] = line.rstrip() + ' #globalignore'
                    break
            file_path.write_text('\n'.join(src_lines), encoding='utf-8')
            print(f"{Fore.YELLOW}Function '{func['name']}' marked as #globalignore.{Style.RESET_ALL}")
            return None
        if res == '-exit':
            print(f"{Fore.YELLOW}Exiting function configuration early.{Style.RESET_ALL}")
            return 'exit'
        # If label is None, use display_name or func['name']
        if 'label' in res and (res['label'] is None or res['label'].strip() == ''):
            res['label'] = display_name or func['name']
        settings[param['name']] = res
        history.append(res)
        i += 1
    # Show summary before saving
    print(f"{Fore.CYAN}\n--- Summary of Configuration ---{Style.RESET_ALL}")
    print(f"Function: {display_name or func['name']}")
    print(f"Category: {category or 'None'}")
    for pname, pconfig in settings.items():
        print(f"  {Fore.GREEN}{pname}{Style.RESET_ALL}: {pconfig}")
    while True:
        confirm = input_with_fix(f"{Fore.YELLOW}Save this configuration? (y/n): {Style.RESET_ALL}", history)
        if confirm.strip().lower() in ('y', 'yes', '1'):
            break
        elif confirm.strip().lower() in ('n', 'no', '0'):
            print(f"{Fore.RED}Configuration discarded.{Style.RESET_ALL}")
            return None
        else:
            print(f"{Fore.YELLOW}Please enter 'y' or 'n'.{Style.RESET_ALL}")
    return {
        'func_name': func['name'],
        'display_name': display_name or func['name'],
        'category': category or None,
        'settings': settings
    }

def load_renderpasses_json() -> Dict[str, Any]:
    if RENDERPASSES_PATH.exists():
        try:
            return json.loads(RENDERPASSES_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}

def save_renderpasses_json(data: Dict[str, Any], mode: str):
    existing = load_renderpasses_json()
    if mode == 'replace':
        out = data
    elif mode == 'append':
        # Only add new keys, do not overwrite existing
        out = existing.copy()
        for k, v in data.items():
            if k not in out:
                out[k] = v
    elif mode == 'merge':
        out = existing.copy()
        for k, v in data.items():
            out[k] = v
    else:
        out = data
    RENDERPASSES_PATH.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"{Fore.GREEN}renderPasses.json updated ({mode}){Style.RESET_ALL}")

def main():
    if '--help' in sys.argv:
        print_help()
    print(f"{Fore.CYAN}Welcome to RenderPass Settings Tool!{Style.RESET_ALL}")
    print("Select mode:")
    print("1. Single function")
    print("2. Batch (all functions in a file not #globalignore)")
    mode = input("Mode (1/2): ").strip()
    funcs = find_functions_in_project()
    if mode == '1':
        fname = input("Function name to configure: ").strip()
        func = next((f for f in funcs if f['name'] == fname), None)
        if not func:
            print(f"{Fore.RED}Function not found.{Style.RESET_ALL}")
            # Fuzzy match suggestion
            try:
                import difflib
                names = [f['name'] for f in funcs]
                close = difflib.get_close_matches(fname, names, n=3, cutoff=0.6)
                if close:
                    print(f"{Fore.YELLOW}Did you mean: {', '.join(close)}{Style.RESET_ALL}")
            except Exception:
                pass
            return
        config = prompt_for_function(func)
        if config is None or config == 'exit':
            print(f"{Fore.YELLOW}No config saved for function.{Style.RESET_ALL}")
            return
        out_mode = input("Output mode (replace/append/merge): ").strip()
        save_renderpasses_json({config['func_name']: config}, out_mode)
    elif mode == '2':
        file_path = input("Enter the path to the Python file to process: ").strip()
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            # If user entered a function name, assume passes.py
            passes_py = SCRIPTS_PATH / "passes.py"
            known_funcs = {"wrap_sort", "sort", "multiply", "sharpen"}
            if file_path.lower() in known_funcs and passes_py.exists():
                print(f"{Fore.YELLOW}File not found. Assuming you meant passes.py.{Style.RESET_ALL}")
                file_path_obj = passes_py
            else:
                print(f"{Fore.RED}File not found.{Style.RESET_ALL}")
                return
        try:
            src = file_path_obj.read_text(encoding='utf-8')
            tree = ast.parse(src)
        except Exception as e:
            print(f"{Fore.RED}Error reading/parsing file: {e}{Style.RESET_ALL}")
            return
        out_mode = input("Select output mode for saving after each function (replace/append/merge): ").strip()
        def is_ignored(func_name: str, src_text: str) -> bool:
            import re
            pattern = re.compile(rf"^\s*def\s+{re.escape(func_name)}\s*\(")
            for line in src_text.splitlines():
                if pattern.match(line):
                    if '#globalignore' in line:
                        return True
                    break
            lines = src_text.splitlines()
            for i, line in enumerate(lines):
                if line.strip().startswith(f"def {func_name}("):
                    if i > 0 and '#globalignore' in lines[i-1]:
                        return True
                    break
            return False
        # Load already defined functions from renderPasses.json
        existing_renderpasses = load_renderpasses_json()
        existing_func_names = set(existing_renderpasses.keys())
        funcs_in_file = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not is_ignored(node.name, src) and node.name not in existing_func_names:
                    funcs_in_file.append({'name': node.name, 'file': file_path_obj, 'node': node})
        if not funcs_in_file:
            print(f"{Fore.YELLOW}No functions to process in file (all are already defined or ignored).{Style.RESET_ALL}")
            return
        for func in funcs_in_file:
            config = prompt_for_function(func)
            if config is not None and config != 'exit':
                save_renderpasses_json({config['func_name']: config}, out_mode)
    else:
        print(f"{Fore.RED}Invalid mode.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Done.{Style.RESET_ALL}")

if __name__ == '__main__':
    main()
