from __future__ import annotations
import os
import time
from PIL import Image
import passes
import inspect
from utils import get_output_dir
from typing import Dict, List, Set, Optional, Tuple, Callable, TYPE_CHECKING
from collections import defaultdict, deque

from guiElements.modernSlotTableWidget import ModernSlotTableWidget
from guiElements.renderPassWidget import RenderPassWidget

if TYPE_CHECKING:
    from masterGUI import GUI



def loadImageFromSlot(slotName: str, slotTable: 'ModernSlotTableWidget') -> Image.Image:
    """Load image from a slot."""
    img = slotTable.get_image(slotName)
    if img is None:
        slot_images = {k: bool(v) for k, v in slotTable.slot_images.items()}
        raise ValueError(f"No image found in slot '{slotName}'. Available slots: {slot_images}")
    return img


def saveImageToSlot(image: Image.Image, slotName: str, slotTable: 'ModernSlotTableWidget') -> None:
    """Save image to a slot. For slot15, saves as a new file."""
    if slotName == "slot15":
        # Get the output directory path
        output_dir = get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to file with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = os.path.join(output_dir, f"{timestamp}.png")
        image.save(filepath)
        
    # Update the UI image slot
    slotTable.set_image(slotName, image)
    
    # For all other slots, update normally
    slotTable.set_image(slotName, image)


def getSlotDependencies(renderPassWidget: 'RenderPassWidget') -> Set[str]:
    """Get all slot dependencies for a render pass."""
    dependencies = set()
    
    # Add input slots
    for slot in renderPassWidget.selectedInputs:
        if slot is not None:
            dependencies.add(slot)
    
    # Add mask slot if used
    settings = renderPassWidget.get_settings()
    maskSlot = settings.get('slot')
    if maskSlot and maskSlot != 'None':
        dependencies.add(maskSlot)
    
    return dependencies


def getSlotOutputs(renderPassWidget: 'RenderPassWidget') -> Set[str]:
    """Get output slots for a render pass."""
    outputs = set()
    if renderPassWidget.selectedOutput is not None:
        outputs.add(renderPassWidget.selectedOutput)
    return outputs


def build_dependency_graph(render_pass_widgets: List['RenderPassWidget']) -> Tuple[Dict[int, Set[int]], Dict[str, List[int]]]:
    """
    Build dependency graph for render passes.
    Returns: (pass_dependencies, slot_producers)
    """
    pass_dependencies = defaultdict(set)
    slot_producers = defaultdict(list)
    
    # Map slots to their producing passes
    for pass_idx, widget in enumerate(render_pass_widgets):
        outputs = getSlotOutputs(widget)
        for slot in outputs:
            slot_producers[slot].append(pass_idx)
    
    # Build pass dependencies based on slot usage
    for pass_idx, widget in enumerate(render_pass_widgets):
        dependencies = getSlotOutputs(widget)
        
        for slot in dependencies:
            if slot in slot_producers:
                # This pass depends on all passes that produce this slot
                for producer_idx in slot_producers[slot]:
                    if producer_idx < pass_idx:  # Only depend on earlier passes
                        pass_dependencies[pass_idx].add(producer_idx)
    
    return dict(pass_dependencies), dict(slot_producers)


def topological_sort(passes: List['RenderPassWidget'], dependencies: Dict[int, Set[int]]) -> List[int]:
    """
    Perform topological sort on render passes based on dependencies.
    Returns ordered list of pass indices.
    """
    n = len(passes)
    in_degree = [0] * n
    adj_list = defaultdict(list)
    
    # Build adjacency list and in-degree
    for pass_idx, deps in dependencies.items():
        for dep in deps:
            adj_list[dep].append(pass_idx)
            in_degree[pass_idx] += 1
    
    # Find all passes with no dependencies
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    ordered_indices = []
    
    while queue:
        current = queue.popleft()
        ordered_indices.append(current)
        
        # Remove this pass from dependencies
        for neighbor in adj_list.get(current, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for cycles
    if len(ordered_indices) != n:
        raise ValueError("Circular dependency detected in render passes")
    
    return ordered_indices


def run_render_pass(render_pass_widget: 'RenderPassWidget', slot_table: 'ModernSlotTableWidget', progress_callback: Optional[Callable[[str], None]] = None) -> Optional[Image.Image]:
    """
    Run a single render pass widget with dynamic function loading and progress tracking.
    """
    renderpass_type = render_pass_widget.renderpass_type
    inputs = render_pass_widget.selectedInputs
    output_slot = render_pass_widget.selectedOutput
    settings = render_pass_widget.get_settings()
    
    # Validate inputs and outputs
    if not output_slot:
        raise ValueError(f"Output slot not set for {renderpass_type}")

    # Skip disabled passes
    if not settings.get('enabled', True):
        if progress_callback:
            progress_callback(f"Skipping disabled pass: {renderpass_type}")
        return None
        
    if progress_callback:
        progress_callback(f"Starting {renderpass_type}...")

    # Check if pass is enabled
    if not settings.get('enabled', True):
        if progress_callback:
            progress_callback(f"Skipping disabled pass: {renderpass_type}")
        return None

    # Map UI names to function names (aligned with passes.py)
    func_name_map = {
    "PixelSort": "wrap_sort",

    "Multiply": "multiply",

    "Add (Clamp Maximum)": "maxAdd",

    "Difference": "difference",

    "Kuwahara": "kuwahara_wrapper",

    "Mix Percent": "mix_percent",

    "Alpha Over": "alpha_over",

        "Mix By Percent": "mix_by_percent",
        "Blur": "blur",
        "Invert": "invert",
        "PixelSort": "sort",
        "Mix Screen": "alpha_over",
        "kuwaharaGPU": "kuwahara_gpu",
        "Cristalline Growth": "cristalline_expansion",
        "Subtract Images": "subtract_images",
        "Contrast Mask": "contrast_mask",
        "Scale to fit": "scale_image",
    }
    
    # Map UI setting names to function parameter names
    setting_name_map = {
    "Mode": "mode",

    "regionsCount": "regions",

    "KernelSize": "kernel",

    "%": "p",

    "Region count*": "regions",

        # Blur settings
        "Blur Type": "blur_type",
        "Blur Kernel": "blur_kernel",
        # Invert settings
        "Invert type": "invert_type",
        "Impact Factor": "impact_factor",
        # Mix settings
        "Mix Factor": "mix_factor",
        # Kuwahara settings
        "kernel_size": "kernel_size",
        # Crystalline Growth settings
        "Cluster Seeds (%)": "c",
        # Scale settings
        "Downscale [%]": "downscale",
        # PixelSort UI (handled specially below)
        "Use vSplitting?": "vSplitting",
        "Flip Horizontal": "flipHorz",
        "Flip Vertical": "flipVert",
        # Contrast Mask handled specially (Luminance Range -> lim_lower/lim_upper)
    }
    
    func_name = func_name_map.get(renderpass_type)
    if not func_name or not hasattr(passes, func_name):
        raise NotImplementedError(f"Renderpass '{renderpass_type}' not implemented.")

    # Create a copy of settings with mapped names
    mapped_settings = {}
    for key, value in settings.items():
        # Map the setting name if it exists in our mapping, otherwise keep original
        mapped_key = setting_name_map.get(key, key)
        mapped_settings[mapped_key] = value
    settings = mapped_settings

    # Special-case transformations based on pass type
    if renderpass_type == "Contrast Mask":
        rng = settings.get("Luminance Range") or settings.get("luminance_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                settings["lim_lower"], settings["lim_upper"] = int(rng[0]), int(rng[1])
            except Exception:
                pass

    if renderpass_type == "PixelSort":
        # Combine flip flags into single flip_dir for sort()
        flip_h = settings.pop("flipHorz", False)
        flip_v = settings.pop("flipVert", False)
        settings["flip_dir"] = bool(flip_h) or bool(flip_v)
        # Provide sensible defaults if not present
        settings.setdefault("mode", "lum")
        settings.setdefault("rotate", False)

    func = getattr(passes, func_name)
    sig = inspect.signature(func)

    # Bilder aus Input-Slots laden
    input_images = []
    for i, slot in enumerate(inputs):
        if slot is None:
            raise ValueError(f"Input slot {i+1} not set for render pass {renderpass_type}")
        img = loadImageFromSlot(slot, slot_table)
        if img is None:
            raise ValueError(f"No image found in slot '{slot}'")
        input_images.append(img)

    # Argumente automatisch zusammenstellen
    args = []
    img_index = 0

    for param in sig.parameters.values():
        param_name = param.name
        
        # Handle image parameters
        if param.annotation == Image.Image or param_name.startswith("img"):
            if img_index >= len(input_images):
                raise ValueError(f"Not enough input images for {renderpass_type}")
            args.append(input_images[img_index])
            img_index += 1
        # Handle mask parameter
        elif param_name == "mask" and settings.get('slot') and settings.get('slot') != 'None':
            mask_slot = settings['slot']
            mask_img = loadImageFromSlot(mask_slot, slot_table)
            if mask_img is None:
                raise ValueError(f"Mask image not found in slot '{mask_slot}'")
            args.append(mask_img)
        # Handle regular settings
        elif param_name in settings:
            val = settings[param_name]
            # Automatisch typisieren
            try:
                if param.annotation == int:
                    val = int(val)
                elif param.annotation == float:
                    val = float(val)
                elif param.annotation == bool:
                    val = str(val).lower() in ("1", "true", "yes")
                # sonst: als string übernehmen
            except Exception as e:
                raise ValueError(f"Could not convert setting '{param_name}': {e}")
            args.append(val)
        elif param.default is not inspect.Parameter.empty:
            args.append(param.default)
        else:
            raise ValueError(f"Missing required parameter '{param_name}' for {renderpass_type}")

    # Execute function
    try:
        if progress_callback:
            progress_callback(f"Processing {renderpass_type}...")
        
        output_img = func(*args)
        
        # Validate output
        if output_img is None:
            raise ValueError(f"{renderpass_type} returned None instead of an image")
            
        if not isinstance(output_img, Image.Image):
            raise ValueError(f"{renderpass_type} returned {type(output_img)} instead of PIL Image")
            
        if progress_callback:
            progress_callback(f"Completed {renderpass_type}")
            
    except Exception as e:
        raise RuntimeError(f"Error in render pass '{renderpass_type}': {e}")

    # Save output
    try:
        saveImageToSlot(output_img, output_slot, slot_table)
    except Exception as e:
        raise RuntimeError(f"Error saving output from {renderpass_type} to {output_slot}: {e}")
    
    if progress_callback:
        progress_callback(f"Saved {renderpass_type} to {output_slot}")

    return output_img


def run_all_render_passes(gui_instance: 'GUI', progress_callback: Optional[Callable[[str], None]] = None) -> None:
    """
    Run all render passes in the GUI's list widget in correct order based on dependencies.
    
    Args:
        gui_instance: The main GUI instance
        progress_callback: Optional callback for progress updates (message: str)
    """
    lw = gui_instance.listWidget.list_widget
    slot_table = gui_instance.slotTable
    
    if lw.count() == 0:
        if progress_callback:
            progress_callback("No render passes to process")
        return
    
    # Collect all render pass widgets
    render_pass_widgets = []
    for i in range(lw.count()):
        widget = lw.itemWidget(lw.item(i))
        render_pass_widgets.append(widget)
    
    if not render_pass_widgets:
        if progress_callback:
            progress_callback("No valid render passes found")
        return
    
    try:
        # Build dependency graph
        if progress_callback:
            progress_callback("Analyzing dependencies...")
        
        dependencies, slot_producers = build_dependency_graph(render_pass_widgets)
        
        # Perform topological sort
        ordered_indices = topological_sort(render_pass_widgets, dependencies)
        
        if progress_callback:
            progress_callback(f"Processing {len(ordered_indices)} render passes...")
        
        # Execute passes in correct order
        for idx, pass_idx in enumerate(ordered_indices):
            widget = render_pass_widgets[pass_idx]
            progress_msg = f"Pass {idx+1}/{len(ordered_indices)}"
            
            try:
                run_render_pass(widget, slot_table, 
                              lambda msg: progress_callback(f"{progress_msg}: {msg}") if progress_callback else None)
                
            except Exception as e:
                # Provide detailed error context
                error_msg = f"Error in pass {idx+1} ({widget.renderpass_type}): {str(e)}"
                if progress_callback:
                    progress_callback(error_msg)
                raise RuntimeError(error_msg)
        
        if progress_callback:
            progress_callback("All render passes completed successfully")
            
    except ValueError as e:
        # Handle dependency errors
        error_msg = f"Dependency error: {str(e)}"
        if progress_callback:
            progress_callback(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        # Handle other errors
        error_msg = f"Render pipeline error: {str(e)}"
        if progress_callback:
            progress_callback(error_msg)
        raise RuntimeError(error_msg)


def validate_render_pipeline(gui_instance: 'GUI') -> List[str]:
    """
    Validate the render pipeline for potential issues.
    
    Args:
        gui_instance: The main GUI instance
    Returns:
        List of validation warnings/errors
    """
    lw = gui_instance.listWidget.list_widget
    slot_table = gui_instance.slotTable
    issues = []
    
    if lw.count() == 0:
        issues.append("No render passes configured")
        return issues
    
    # Collect all render pass widgets
    render_pass_widgets = []
    for i in range(lw.count()):
        widget = lw.itemWidget(lw.item(i))
        render_pass_widgets.append(widget)
    
    # Get dependency information
    _, slot_producers = build_dependency_graph(render_pass_widgets)
    
    # Track which slots will have images after each pass
    will_have_image = {slot: False for slot in gui_instance.available_slots}
    will_have_image["slot0"] = True  # Input slot always has an image
    
    # Add slots that already have images
    for slot in gui_instance.available_slots:
        if slot_table.get_image(slot) is not None:
            will_have_image[slot] = True
    
    # Check passes in order
    for i, widget in enumerate(render_pass_widgets):
        renderpass_type = widget.renderpass_type
        
        # Check input slots
        for j, slot in enumerate(widget.selectedInputs):
            if slot is None:
                issues.append(f"Pass {i+1} ({renderpass_type}): Missing input {j+1}")
            elif slot == "slot0":
                continue  # slot0 always has input image
            elif not will_have_image[slot]:
                # Check if this slot already has an image
                if slot_table.get_image(slot) is not None:
                    will_have_image[slot] = True
                    continue
                    
                # Check if any earlier pass will produce this slot
                produces_slot = False
                for producer_idx in slot_producers.get(slot, []):
                    if producer_idx < i:  # Only consider earlier passes
                        produces_slot = True
                        break
                
                if not produces_slot:
                    issues.append(f"Pass {i+1} ({renderpass_type}): No image in input slot '{slot}' and no earlier pass will produce it")
        
        # Check output slot
        if widget.selectedOutput is None:
            issues.append(f"Pass {i+1} ({renderpass_type}): Missing output slot")
        else:
            # Mark that this slot will have an image after this pass
            will_have_image[widget.selectedOutput] = True
    
    # Check for circular dependencies
    try:
        dependencies, _ = build_dependency_graph(render_pass_widgets)
        topological_sort(render_pass_widgets, dependencies)
    except ValueError as e:
        issues.append(str(e))
    
    return issues
