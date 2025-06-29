from typing import List, Dict, Any
import json
import sys

def create_setting(config_type: str) -> Dict[str, Any]:
    """Creates a single setting configuration based on type."""
    name = input("Enter setting name (label): ").strip()
    
    config = {"label": name, "type": config_type}
    
    if config_type in ["radio", "dropdown"]:
        options = input("Enter comma-separated options (e.g., Gaussian,Box,Median): ").strip()
        config["options"] = [opt.strip() for opt in options.split(",")]
        config["default"] = input(f"Enter default option from {config['options']}: ").strip()
    
    elif config_type in ["slider", "multislider"]:
        config["min"] = float(input("Enter minimum value: "))
        config["max"] = float(input("Enter maximum value: "))
        
        if config_type == "multislider":
            default_values = input("Enter comma-separated default values (e.g., 1,0.5,0): ").strip()
            config["default"] = [float(v) for v in default_values.split(",")]
        else:
            config["default"] = float(input("Enter default value: "))
    
    elif config_type == "switch":
        config["default"] = input("Enable by default? (y/n): ").strip().lower() == "y"
    
    return config

def create_renderpass_settings() -> List[Dict[str, Any]]:
    """Creates a complete settings configuration for a render pass type."""
    settings = []
    
    print("\nAvailable setting types: radio, dropdown, slider, multislider, switch")
    
    while True:
        print("\nCurrent settings:")
        for i, setting in enumerate(settings, 1):
            print(f"{i}. {setting['label']} ({setting['type']})")
            
        print("\nAdd a new setting")
        print("1. Add radio button group")
        print("2. Add dropdown select box")
        print("3. Add slider")
        print("4. Add multislider")
        print("5. Add toggle switch")
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

def main():
    print("="*50)
    print("Render Pass Settings Generator")
    print("="*50)
    
    renderpass_type = input("\nEnter the name of your new render pass type: ").strip()
    settings = create_renderpass_settings()
    
    print("\n" + "="*50)
    print("Your settings configuration code:\n")
    
    code_block = generate_code_block(renderpass_type, settings)
    print(code_block)
    
    print("Copy the code block above into your get_settings_config() method.")
    print("="*50)

if __name__ == "__main__":
    main()
