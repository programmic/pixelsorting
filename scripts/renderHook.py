from __future__ import annotations

import os
import time
import inspect
from typing import Dict, List, Set, Optional, Tuple, Callable, TYPE_CHECKING
from collections import defaultdict, deque

from PIL import Image

import passes

from guiElements.modernSlotTableWidget import ModernSlotTableWidget
from guiElements.renderPassWidget import RenderPassWidget

if TYPE_CHECKING:
    from masterGUI import GUI

import os
import time
import inspect
from typing import Dict, List, Set, Optional, Tuple, Callable, TYPE_CHECKING
from collections import defaultdict, deque

from PIL import Image

import passes

from guiElements.modernSlotTableWidget import ModernSlotTableWidget
from guiElements.renderPassWidget import RenderPassWidget

if TYPE_CHECKING:
    from masterGUI import GUI


def loadImageFromSlot(slotName: str, slotTable: 'ModernSlotTableWidget') -> Image.Image:
    img = slotTable.get_image(slotName)
    if img is None:
        slot_images = {k: bool(v) for k, v in getattr(slotTable, 'slot_images', {}).items()}
        raise ValueError(f"No image found in slot '{slotName}'. Available slots: {slot_images}")
    return img


def saveImageToSlot(image: Image.Image, slotName: str, slotTable: 'ModernSlotTableWidget') -> None:
    if slotName == "slot15":
        try:
            from utils import get_output_dir

            output_dir = get_output_dir()
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(output_dir, f"{timestamp}.png")
            image.save(filepath)
        except Exception:
            pass
    slotTable.set_image(slotName, image)


def getSlotDependencies(renderPassWidget: 'RenderPassWidget') -> Set[str]:
    dependencies: Set[str] = set()
    for slot in getattr(renderPassWidget, 'selectedInputs', []):

        import os
        import time
        import inspect
        from typing import Dict, List, Set, Optional, Tuple, Callable
        from collections import defaultdict, deque

        from PIL import Image

        import passes

        from guiElements.modernSlotTableWidget import ModernSlotTableWidget
        from guiElements.renderPassWidget import RenderPassWidget


        def loadImageFromSlot(slotName: str, slotTable: 'ModernSlotTableWidget') -> Image.Image:
            img = slotTable.get_image(slotName)
            if img is None:
                slot_images = {k: bool(v) for k, v in getattr(slotTable, 'slot_images', {}).items()}
                raise ValueError(f"No image found in slot '{slotName}'. Available slots: {slot_images}")
            return img


        def saveImageToSlot(image: Image.Image, slotName: str, slotTable: 'ModernSlotTableWidget') -> None:
            if slotName == "slot15":
                try:
                    from utils import get_output_dir

                    output_dir = get_output_dir()
                    os.makedirs(output_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filepath = os.path.join(output_dir, f"{timestamp}.png")
                    image.save(filepath)
                except Exception:
                    pass
            slotTable.set_image(slotName, image)



import os
import time
import inspect
from typing import Dict, List, Set, Optional, Tuple, Callable, TYPE_CHECKING
from collections import defaultdict, deque

from PIL import Image

import passes

from guiElements.modernSlotTableWidget import ModernSlotTableWidget
from guiElements.renderPassWidget import RenderPassWidget

if TYPE_CHECKING:
    from masterGUI import GUI


def loadImageFromSlot(slotName: str, slotTable: 'ModernSlotTableWidget') -> Image.Image:
    img = slotTable.get_image(slotName)
    if img is None:
        slot_images = {k: bool(v) for k, v in getattr(slotTable, 'slot_images', {}).items()}
        raise ValueError(f"No image found in slot '{slotName}'. Available slots: {slot_images}")
    return img


def saveImageToSlot(image: Image.Image, slotName: str, slotTable: 'ModernSlotTableWidget') -> None:
    if slotName == "slot15":
        try:
            from utils import get_output_dir

            output_dir = get_output_dir()
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(output_dir, f"{timestamp}.png")
            image.save(filepath)
        except Exception:
            pass
    slotTable.set_image(slotName, image)


def getSlotDependencies(renderPassWidget: 'RenderPassWidget') -> Set[str]:
    dependencies: Set[str] = set()
    for slot in getattr(renderPassWidget, 'selectedInputs', []):
        if slot is not None:
            dependencies.add(slot)
    settings = renderPassWidget.get_settings()
    mask = None
    if isinstance(settings.get('mask'), dict):
        mask = settings['mask'].get('slot')
    else:
        mask = settings.get('mask')
    if isinstance(mask, str) and mask and mask != 'None':
        dependencies.add(mask)
    return dependencies


def getSlotOutputs(renderPassWidget: 'RenderPassWidget') -> Set[str]:
    outputs: Set[str] = set()
    if getattr(renderPassWidget, 'selectedOutput', None) is not None:
        outputs.add(renderPassWidget.selectedOutput)
    return outputs


def build_dependency_graph(render_pass_widgets: List['RenderPassWidget']) -> Tuple[Dict[int, Set[int]], Dict[str, List[int]]]:
    pass_dependencies: Dict[int, Set[int]] = defaultdict(set)
    slot_producers: Dict[str, List[int]] = defaultdict(list)

    for pass_idx, widget in enumerate(render_pass_widgets):
        outputs = getSlotOutputs(widget)
        for slot in outputs:
            slot_producers[slot].append(pass_idx)

    for pass_idx, widget in enumerate(render_pass_widgets):
        inputs = getSlotDependencies(widget)
        for slot in inputs:
            if slot in slot_producers:
                for producer_idx in slot_producers[slot]:
                    if producer_idx < pass_idx:
                        pass_dependencies[pass_idx].add(producer_idx)

    return dict(pass_dependencies), dict(slot_producers)


def topological_sort(passes_list: List['RenderPassWidget'], dependencies: Dict[int, Set[int]]) -> List[int]:
    n = len(passes_list)
    in_degree = [0] * n
    adj_list: Dict[int, List[int]] = defaultdict(list)

    for pass_idx, deps in dependencies.items():
        for dep in deps:
            adj_list[dep].append(pass_idx)
            in_degree[pass_idx] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    ordered_indices: List[int] = []

    while queue:
        current = queue.popleft()
        ordered_indices.append(current)
        for neighbor in adj_list.get(current, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered_indices) != n:
        raise ValueError("Circular dependency detected in render passes")

    return ordered_indices


def run_render_pass(render_pass_widget: 'RenderPassWidget', slot_table: 'ModernSlotTableWidget', progress_callback: Optional[Callable[[str], None]] = None) -> Optional[Image.Image]:
    renderpass_type = render_pass_widget.renderpass_type
    inputs = getattr(render_pass_widget, 'selectedInputs', [])
    output_slot = getattr(render_pass_widget, 'selectedOutput', None)
    settings = render_pass_widget.get_settings()

    if not output_slot:
        raise ValueError(f"Output slot not set for {renderpass_type}")

    if not settings.get('enabled', True):
        if progress_callback:
            progress_callback(f"Skipping disabled pass: {renderpass_type}")
        return None

    if progress_callback:
        progress_callback(f"Starting {renderpass_type}...")

    func_name: Optional[str] = None
    meta = None
    try:
        if hasattr(RenderPassWidget, '_settings_cache') and RenderPassWidget._settings_cache:
            meta = RenderPassWidget._settings_cache.get(render_pass_widget.renderpass_type)
        if isinstance(meta, dict):
            func_name = meta.get('original_func_name') or meta.get('function_alias')
    except Exception:
        func_name = None

    if not func_name:
        func_name = renderpass_type

    if not hasattr(passes, func_name):
        raise NotImplementedError(f"Renderpass '{renderpass_type}' (resolved to '{func_name}') not implemented.")

    setting_name_map: Dict[str, str] = {}
    if isinstance(meta, dict) and isinstance(meta.get('settings'), list):
        for s in meta.get('settings'):
            target = s.get('name') or s.get('alias') or s.get('label')
            if not target:
                continue
            if s.get('label'):
                setting_name_map[s.get('label')] = target
            if s.get('alias'):
                setting_name_map[s.get('alias')] = target

    mapped_settings: Dict[str, object] = {}
    for key, value in settings.items():
        mapped_key = setting_name_map.get(key, key)
        mapped_settings[mapped_key] = value
    settings = mapped_settings

    if renderpass_type == "Contrast Mask":
        rng = settings.get("Luminance Range") or settings.get("luminance_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                settings["lim_lower"], settings["lim_upper"] = int(rng[0]), int(rng[1])
            except Exception:
                pass

    if renderpass_type == "PixelSort":
        flip_h = settings.pop("flipHorz", False)
        flip_v = settings.pop("flipVert", False)
        settings["flip_dir"] = bool(flip_h) or bool(flip_v)
        settings.setdefault("mode", "lum")
        settings.setdefault("rotate", False)

    if func_name in ["blur_box_gpu", "blur_box"]:
        blur_kernel = settings.get("blur_kernel")
        if blur_kernel is None:
            blur_kernel = 3
        try:
            blur_kernel = int(blur_kernel)
        except Exception:
            blur_kernel = 3
        settings["blur_kernel"] = blur_kernel

    func = getattr(passes, func_name)
    sig = inspect.signature(func)

    input_images: List[Image.Image] = []
    for i, slot in enumerate(inputs):
        if slot is None:
            raise ValueError(f"Input slot {i+1} not set for render pass {renderpass_type}")
        img = loadImageFromSlot(slot, slot_table)
        input_images.append(img)

    args: List[object] = []
    img_index = 0
    for param in sig.parameters.values():
        param_name = param.name
        ann = param.annotation
        if ann == Image.Image or param_name.startswith("img"):
            if img_index >= len(input_images):
                raise ValueError(f"Not enough input images for {renderpass_type}")
            args.append(input_images[img_index])
            img_index += 1
        elif param_name == "mask":
            mask_slot = None
            if isinstance(settings.get('mask'), dict):
                mask_slot = settings['mask'].get('slot')
            else:
                mask_slot = settings.get('mask')
            if mask_slot and mask_slot != 'None':
                mask_img = loadImageFromSlot(mask_slot, slot_table)
                args.append(mask_img)
            else:
                if param.default is not inspect.Parameter.empty:
                    args.append(param.default)
                else:
                    args.append(None)
        elif param_name in settings:
            val = settings[param_name]
            if val is None or (isinstance(val, str) and val.strip() == ""):
                if ann == int:
                    val = 0
                elif ann == float:
                    val = 0.0
                elif ann == bool:
                    val = False
                else:
                    val = ""
            try:
                if ann == int:
                    val = int(val)
                elif ann == float:
                    val = float(val)
                elif ann == bool:
                    val = str(val).lower() in ("1", "true", "yes")
            except Exception as e:
                raise ValueError(f"Could not convert setting '{param_name}': {e}")
            args.append(val)
        elif param.default is not inspect.Parameter.empty:
            args.append(param.default)
        else:
            raise ValueError(f"Missing required parameter '{param_name}' for {renderpass_type}")

    try:
        if progress_callback:
            progress_callback(f"Processing {renderpass_type}...")
        output_img = func(*args)
        if output_img is None:
            raise ValueError(f"{renderpass_type} returned None instead of an image")
        if not isinstance(output_img, Image.Image):
            raise ValueError(f"{renderpass_type} returned {type(output_img)} instead of PIL Image")
        if progress_callback:
            progress_callback(f"Completed {renderpass_type}")
    except Exception as e:
        raise RuntimeError(f"Error in render pass '{renderpass_type}': {e}")

    try:
        saveImageToSlot(output_img, output_slot, slot_table)
    except Exception as e:
        raise RuntimeError(f"Error saving output from {renderpass_type} to {output_slot}: {e}")

    if progress_callback:
        progress_callback(f"Saved {renderpass_type} to {output_slot}")

    return output_img


def run_all_render_passes(gui_instance: 'GUI', progress_callback: Optional[Callable[[str], None]] = None) -> None:
    lw = gui_instance.listWidget.list_widget
    slot_table = gui_instance.slotTable

    if lw.count() == 0:
        if progress_callback:
            progress_callback("No render passes to process")
        return

    render_pass_widgets: List[RenderPassWidget] = []
    for i in range(lw.count()):
        widget = lw.itemWidget(lw.item(i))
        render_pass_widgets.append(widget)

    if not render_pass_widgets:
        if progress_callback:
            progress_callback("No valid render passes found")
        return

    try:
        if progress_callback:
            progress_callback("Analyzing dependencies...")
        dependencies, slot_producers = build_dependency_graph(render_pass_widgets)
        ordered_indices = topological_sort(render_pass_widgets, dependencies)
        if progress_callback:
            progress_callback(f"Processing {len(ordered_indices)} render passes...")
        for idx, pass_idx in enumerate(ordered_indices):
            widget = render_pass_widgets[pass_idx]
            progress_msg = f"Pass {idx+1}/{len(ordered_indices)}"
            try:
                run_render_pass(widget, slot_table,
                                lambda msg: progress_callback(f"{progress_msg}: {msg}") if progress_callback else None)
            except Exception as e:
                error_msg = f"Error in pass {idx+1} ({widget.renderpass_type}): {str(e)}"
                if progress_callback:
                    progress_callback(error_msg)
                raise RuntimeError(error_msg)
        if progress_callback:
            progress_callback("All render passes completed successfully")
    except ValueError as e:
        error_msg = f"Dependency error: {str(e)}"
        if progress_callback:
            progress_callback(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Render pipeline error: {str(e)}"
        if progress_callback:
            progress_callback(error_msg)
        raise RuntimeError(error_msg)


def validate_render_pipeline(gui_instance: 'GUI') -> List[str]:
    lw = gui_instance.listWidget.list_widget
    slot_table = gui_instance.slotTable
    issues: List[str] = []

    if lw.count() == 0:
        issues.append("No render passes configured")
        return issues

    render_pass_widgets: List[RenderPassWidget] = []
    for i in range(lw.count()):
        widget = lw.itemWidget(lw.item(i))
        render_pass_widgets.append(widget)

    _, slot_producers = build_dependency_graph(render_pass_widgets)

    will_have_image = {slot: False for slot in gui_instance.available_slots}
    will_have_image["slot0"] = True

    for slot in gui_instance.available_slots:
        if slot_table.get_image(slot) is not None:
            will_have_image[slot] = True

    for i, widget in enumerate(render_pass_widgets):
        renderpass_type = widget.renderpass_type
        for j, slot in enumerate(getattr(widget, 'selectedInputs', [])):
            if slot is None:
                issues.append(f"Pass {i+1} ({renderpass_type}): Missing input {j+1}")
            elif slot == "slot0":
                continue
            elif not will_have_image.get(slot, False):
                if slot_table.get_image(slot) is not None:
                    will_have_image[slot] = True
                    continue
                produces_slot = False
                for producer_idx in slot_producers.get(slot, []):
                    if producer_idx < i:
                        produces_slot = True
                        break
                if not produces_slot:
                    issues.append(f"Pass {i+1} ({renderpass_type}): No image in input slot '{slot}' and no earlier pass will produce it")
        if getattr(widget, 'selectedOutput', None) is None:
            issues.append(f"Pass {i+1} ({renderpass_type}): Missing output slot")
        else:
            will_have_image[widget.selectedOutput] = True

    try:
        dependencies, _ = build_dependency_graph(render_pass_widgets)
        topological_sort(render_pass_widgets, dependencies)
    except ValueError as e:
        issues.append(str(e))

    return issues
