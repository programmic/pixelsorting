from __future__ import annotations

import os
import time
import inspect
from typing import Dict, List, Set, Optional, Tuple, Callable, TYPE_CHECKING, Any
from collections import defaultdict, deque

from PIL import Image
import io
import hashlib

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


try:
    from enums import Slot
except Exception:
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("enums", os.path.join(os.path.dirname(__file__), "enums.py"))
    enums = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enums)
    Slot = enums.Slot


def loadImageFromSlot(slotName: Any, slotTable: 'ModernSlotTableWidget') -> Image.Image:
    slot_key = slotName
    try:
        slot_key = Slot.from_value(slotName).value
    except Exception:
        pass
    img = slotTable.get_image(slot_key)
    if img is None:
        slot_images = {k: bool(v) for k, v in getattr(slotTable, 'slot_images', {}).items()}
        raise ValueError(f"No image found in slot '{slot_key}'. Available slots: {slot_images}")
    return img


def saveImageToSlot(image: Image.Image, slotName: Any, slotTable: 'ModernSlotTableWidget') -> None:
    slot_enum = None
    try:
        slot_enum = Slot.from_value(slotName)
    except Exception:
        slot_enum = None
    slot_key = slot_enum.value if slot_enum is not None else slotName
    if slot_enum == Slot.SLOT15 or (isinstance(slot_key, str) and slot_key == "slot15"):
        try:
            from utils import get_output_dir

            output_dir = get_output_dir()
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(output_dir, f"{timestamp}.png")
            image.save(filepath)
        except Exception:
            pass
    slotTable.set_image(slot_key, image)
    # Update image cache if available
    try:
        img_hash = _compute_image_hash(image)
        if hasattr(slotTable, '_image_cache') and isinstance(slotTable._image_cache, dict):
            slotTable._image_cache[slot_key] = {'image_hash': img_hash, 'pass_hash': None, 'image': image}
    except Exception:
        pass


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
    # update image cache if present
    try:
        img_hash = _compute_image_hash(image)
        if hasattr(slotTable, '_image_cache') and isinstance(slotTable._image_cache, dict):
            slotTable._image_cache[slotName] = {'image_hash': img_hash, 'pass_hash': None, 'image': image}
    except Exception:
        pass


def getSlotDependencies(renderPassWidget: 'RenderPassWidget') -> Set[str]:
    dependencies: Set[str] = set()
    for slot in getattr(renderPassWidget, 'selectedInputs', []):
        if slot is not None:
            try:
                dependencies.add(Slot.from_value(slot).value)
            except Exception:
                dependencies.add(slot)
    settings = renderPassWidget.get_settings()
    mask = None
    if isinstance(settings.get('mask'), dict):
        mask = settings['mask'].get('slot')
    else:
        mask = settings.get('mask')
    if isinstance(mask, str) and mask and mask != 'None':
        try:
            dependencies.add(Slot.from_value(mask).value)
        except Exception:
            dependencies.add(mask)
    return dependencies


def getSlotOutputs(renderPassWidget: 'RenderPassWidget') -> Set[str]:
    outputs: Set[str] = set()
    if getattr(renderPassWidget, 'selectedOutput', None) is not None:
        out = renderPassWidget.selectedOutput
        try:
            outputs.add(Slot.from_value(out).value)
        except Exception:
            outputs.add(out)
    return outputs


def build_setting_name_map(meta_obj: Optional[dict]) -> Dict[str, str]:
    """Build mapping from UI keys (label/alias) to function parameter names.

    This prefers an explicit 'alias_to_param' mapping when present (authoritative),
    and falls back to deriving the mapping from individual setting entries.
    """
    mapping: Dict[str, str] = {}
    if not isinstance(meta_obj, dict):
        return mapping

    alias_map = meta_obj.get('alias_to_param') if isinstance(meta_obj.get('alias_to_param'), dict) else {}
    # Use authoritative alias_to_param first
    for a, p in alias_map.items():
        if a and p:
            mapping[a] = p

    # Fallback: derive mapping from per-setting entries
    settings_list = meta_obj.get('settings')
    if isinstance(settings_list, list):
        for s in settings_list:
            if not isinstance(s, dict):
                continue
            # prefer explicit 'original_name' or 'name', else try to use alias_map resolution, else alias, else label
            param_name = s.get('original_name') or s.get('name') or alias_map.get(s.get('alias')) or s.get('alias') or s.get('label')
            if not param_name:
                continue
            if s.get('label'):
                mapping[s.get('label')] = param_name
            if s.get('alias'):
                mapping[s.get('alias')] = param_name
    return mapping


def _compute_image_hash(img: Image.Image) -> str:
    """Compute a stable hash for a PIL Image using PNG bytes."""
    try:
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        h = hashlib.sha256(buf.read()).hexdigest()
        return h
    except Exception:
        # Fallback: use repr
        return hashlib.sha256(repr(img).encode('utf-8')).hexdigest()


def _compute_pass_signature(func_name: str, settings: dict, input_hashes: List[str], mask_hash: Optional[str]) -> str:
    """Create a deterministic signature string for a render pass invocation."""
    try:
        import json
        settings_str = json.dumps(settings or {}, sort_keys=True, default=str)
    except Exception:
        settings_str = str(settings)
    parts = [func_name, settings_str]
    parts.extend(input_hashes or [])
    if mask_hash:
        parts.append(mask_hash)
    sig = '|'.join(parts)
    return hashlib.sha256(sig.encode('utf-8')).hexdigest()


def _tokens(s: str) -> set:
    if not s:
        return set()
    import re
    parts = re.split(r"[^0-9a-zA-Z]+", s.lower())
    return set(p for p in parts if p)


def resolve_func_name(renderpass_type: str) -> str:
    """Resolve the underlying function name for a render pass display string.

    This prefers an exact attribute on the `passes` module, then consults the
    cached `renderPasses.json` metadata (via `RenderPassWidget._settings_cache`) to
    map display names or keys to the configured `func_name`. If nothing matches
    we fall back to the original display string.
    """
    # If the passes module already provides a function with this name, use it.
    if hasattr(passes, renderpass_type):
        return renderpass_type

    cache = getattr(RenderPassWidget, '_settings_cache', None)
    if not isinstance(cache, dict):
        return renderpass_type

    tgt = _tokens(renderpass_type)
    # prefer exact token equality on display_name/func_name
    for key, val in cache.items():
        if not isinstance(val, dict):
            continue
        dn = val.get('display_name') or ''
        fn = val.get('func_name') or ''
        if _tokens(dn) == tgt or _tokens(fn) == tgt:
            return fn or key

    # allow subset matching (e.g. 'mix percent' -> 'Mix (By percent)')
    for key, val in cache.items():
        if not isinstance(val, dict):
            continue
        dn = val.get('display_name') or ''
        fn = val.get('func_name') or ''
        key_tokens = _tokens(key)
        if tgt and (tgt.issubset(_tokens(dn)) or tgt.issubset(_tokens(fn)) or tgt.issubset(key_tokens)):
            return fn or key

    return renderpass_type


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

    # Normalize common UI/global keys to the parameter names expected by
    # render functions. This handles cases where older GUI code or saved
    # settings use different keys (e.g. `useVerticalSplitting`, `sortMode`).
    if isinstance(settings, dict):
        lower_map = {k.lower(): k for k in list(settings.keys())}
        def _has_lower(key):
            return key.lower() in lower_map
        # vSplitting aliases
        if 'vSplitting' not in settings and _has_lower('useVerticalSplitting'):
            source_key = lower_map['useverticalsplitting']
            settings['vSplitting'] = bool(settings.get(source_key))
        # sort mode aliases
        if 'mode' not in settings and _has_lower('sortmode'):
            source_key = lower_map['sortmode']
            settings['mode'] = settings.get(source_key)
        # rotate aliases
        if 'rotate' not in settings and _has_lower('rotateimage'):
            source_key = lower_map['rotateimage']
            # map boolean to "0"/"90" style if necessary; here we keep boolean
            settings['rotate'] = settings.get(source_key)
        # flip aliases
        if 'flipHorz' not in settings:
            for candidate in ('fliphoriz', 'flip_horiz', 'flip_horizontal', 'mirror'):
                if _has_lower(candidate):
                    settings['flipHorz'] = bool(settings.get(lower_map[candidate]))
                    break
        if 'flipVert' not in settings:
            for candidate in ('flipvert', 'flip_vert', 'flip_vertical'):
                if _has_lower(candidate):
                    settings['flipVert'] = bool(settings.get(lower_map[candidate]))
                    break
        # allow lowercase 'fliphoriz' typo mapping
        if 'fliphoriz' in settings and 'flipHorz' not in settings:
            settings['flipHorz'] = settings.get('fliphoriz')
        # If vSplitting still not present, try to read the default from
        # the renderPasses.json metadata (RenderPassWidget._settings_cache)
        if 'vSplitting' not in settings:
            try:
                cache = getattr(RenderPassWidget, '_settings_cache', None)
                if isinstance(cache, dict):
                    meta = cache.get(render_pass_widget.renderpass_type)
                    if isinstance(meta, dict):
                        s_list = meta.get('settings')
                        # settings in config may be dict or list
                        if isinstance(s_list, dict):
                            s_items = list(s_list.values())
                        else:
                            s_items = s_list or []
                        for s in s_items:
                            if not isinstance(s, dict):
                                continue
                            name = (s.get('name') or '')
                            alias = (s.get('alias') or '')
                            label = (s.get('label') or '')
                            if any(x.lower() == 'vsplitting' for x in (name, alias, label)):
                                settings['vSplitting'] = bool(s.get('default', True))
                                break
            except Exception:
                # if anything fails, fall back to vertical splitting as user requested
                if 'vSplitting' not in settings:
                    settings['vSplitting'] = True

    if not output_slot:
        raise ValueError(f"Output slot not set for {renderpass_type}")

    if not settings.get('enabled', True):
        if progress_callback:
            progress_callback(f"Skipping disabled pass: {renderpass_type}")
        return None

    if progress_callback:
        progress_callback(f"Starting {renderpass_type}...")

    # Resolve the actual function name using the configured metadata
    func_name = resolve_func_name(renderpass_type)
    # All settings and mapping should be handled by the new config system
    if not hasattr(passes, func_name):
        raise NotImplementedError(f"Renderpass '{renderpass_type}' (resolved to '{func_name}') not implemented.")

    if renderpass_type == "Contrast Mask":
        rng = settings.get("Luminance Range") or settings.get("luminance_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                settings["lim_lower"], settings["lim_upper"] = int(rng[0]), int(rng[1])
            except Exception:
                pass

    # Special handling for pixel sort style passes: normalize flip/rotate settings
    if func_name in ("wrap_sort", "sort", "PixelSort", "pixel_sort"):
        # Preserve original keys so functions that expect `flipHorz`/`flipVert`
        # still receive them. Also provide a compatibility `flip_dir` key.
        flip_h = settings.get("flipHorz", False)
        flip_v = settings.get("flipVert", False)
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

    # --- CACHING: compute input image hashes and pass signature; skip if unchanged ---
    input_hashes: List[str] = []
    input_slot_keys: List[str] = []
    for slot in inputs:
        # normalize slot key
        try:
            sk = Slot.from_value(slot).value
        except Exception:
            sk = slot
        input_slot_keys.append(sk)
        # prefer cached image hash if available
        ih = None
        try:
            if hasattr(slot_table, '_image_cache') and isinstance(slot_table._image_cache, dict):
                entry = slot_table._image_cache.get(sk)
                if isinstance(entry, dict) and entry.get('image_hash'):
                    ih = entry['image_hash']
        except Exception:
            ih = None
        if ih is None:
            try:
                img = loadImageFromSlot(sk, slot_table)
                ih = _compute_image_hash(img)
            except Exception:
                ih = hashlib.sha256(str(sk).encode('utf-8')).hexdigest()
        input_hashes.append(ih)

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
        # compute mask hash if applicable
        mask_slot = None
        if isinstance(settings.get('mask'), dict):
            mask_slot = settings['mask'].get('slot')
        else:
            mask_slot = settings.get('mask')
        mask_hash = None
        if isinstance(mask_slot, str) and mask_slot and mask_slot != 'None':
            try:
                mk = Slot.from_value(mask_slot).value
            except Exception:
                mk = mask_slot
            try:
                if hasattr(slot_table, '_image_cache') and isinstance(slot_table._image_cache, dict) and slot_table._image_cache.get(mk):
                    mask_hash = slot_table._image_cache.get(mk, {}).get('image_hash')
                else:
                    mask_img = loadImageFromSlot(mk, slot_table)
                    mask_hash = _compute_image_hash(mask_img)
            except Exception:
                mask_hash = None
        # compute pass signature/hash
        try:
            pass_hash = _compute_pass_signature(func_name, settings, input_hashes, mask_hash)
        except Exception:
            pass_hash = None
        # If output slot already has cached result with same pass_hash, reuse it
        try:
            out_slot_key = None
            try:
                out_slot_key = Slot.from_value(output_slot).value
            except Exception:
                out_slot_key = output_slot
            cache_entry = None
            if hasattr(slot_table, '_image_cache') and isinstance(slot_table._image_cache, dict):
                cache_entry = slot_table._image_cache.get(out_slot_key)
            if cache_entry and pass_hash and cache_entry.get('pass_hash') == pass_hash:
                # reuse cached image
                cached_img = cache_entry.get('image')
                if cached_img is not None:
                    if progress_callback:
                        progress_callback(f"Skipping {renderpass_type} (cached)")
                    # Ensure slot table has the image set (it likely already does)
                    try:
                        slot_table.set_image(out_slot_key, cached_img)
                    except Exception:
                        pass
                    return cached_img
        except Exception:
            pass
        # Print a concise summary of the arguments we're about to pass to the render function
        def _short_repr(v):
            try:
                from PIL import Image as PILImage
            except Exception:
                PILImage = None
            if PILImage is not None and isinstance(v, PILImage.Image):
                try:
                    return f"<PIL.Image size={v.size} mode={v.mode}>"
                except Exception:
                    return "<PIL.Image>"
            if isinstance(v, (list, tuple)):
                return f"{type(v).__name__}(len={len(v)})"
            if isinstance(v, dict):
                return f"dict(len={len(v)})"
            s = repr(v)
            if isinstance(s, str) and len(s) > 200:
                return s[:197] + '...'
            return s

        # Map parameter names to argument values for clearer debugging
        param_names = [p.name for p in sig.parameters.values()]
        arg_map = {}
        for i, name in enumerate(param_names):
            try:
                val = args[i]
            except Exception:
                val = '<missing>'
            arg_map[name] = _short_repr(val)

        print(f"[DEBUG] Calling render pass '{renderpass_type}' -> function '{func_name}' with args:")
        for k, v in arg_map.items():
            print(f"  {k}: {v}")
        if progress_callback:
            progress_callback(f"Calling {renderpass_type} (function {func_name}) with {len(arg_map)} args")

        # Prepare to pass an inner progress callback if the function accepts it
        kwargs = {}
        try:
            accept_progress = False
            progress_param_name = None
            if 'progress' in sig.parameters:
                progress_param_name = 'progress'
                accept_progress = True
            elif 'progress_callback' in sig.parameters:
                progress_param_name = 'progress_callback'
                accept_progress = True
            else:
                # If the function accepts **kwargs, we'll pass a 'progress' kw
                for p in sig.parameters.values():
                    if p.kind == inspect.Parameter.VAR_KEYWORD:
                        progress_param_name = 'progress'
                        accept_progress = True
                        break
            if accept_progress and progress_param_name:
                def _pass_inner_progress(payload):
                    # payload can be string, dict, or percent number
                    try:
                        if isinstance(payload, dict):
                            # forward structured inner progress
                            if progress_callback:
                                progress_callback({'type': 'inner', 'pass': renderpass_type, **payload})
                        elif isinstance(payload, (int, float)):
                            if progress_callback:
                                progress_callback({'type': 'inner', 'pass': renderpass_type, 'percent': int(payload), 'message': ''})
                        else:
                            if progress_callback:
                                progress_callback({'type': 'inner', 'pass': renderpass_type, 'message': str(payload)})
                    except Exception:
                        pass

                # If the parameter exists as a positional parameter and we already
                # prepared a positional args list that includes an entry for it,
                # place the progress callable into that position to avoid passing
                # it twice (positional + kw). Otherwise, pass as kw.
                param_names = [p.name for p in sig.parameters.values()]
                try:
                    if progress_param_name in param_names:
                        idx = param_names.index(progress_param_name)
                        if idx < len(args):
                            args[idx] = _pass_inner_progress
                        else:
                            kwargs[progress_param_name] = _pass_inner_progress
                    else:
                        kwargs[progress_param_name] = _pass_inner_progress
                except Exception:
                    kwargs[progress_param_name] = _pass_inner_progress
        except Exception:
            kwargs = {}

        result = func(*args, **kwargs)

        # If the function is a generator/yielding progress, iterate it
        try:
            import types
            final_img = None
            if isinstance(result, types.GeneratorType):
                for item in result:
                    # items may be progress payloads or the final image
                    if isinstance(item, dict) or isinstance(item, str) or isinstance(item, (int, float)):
                        try:
                            if progress_callback:
                                progress_callback({'type': 'inner', 'pass': renderpass_type, 'message': item})
                        except Exception:
                            pass
                    elif isinstance(item, Image.Image):
                        final_img = item
                # Some generators may return via StopIteration.value (Python 3.3+)
                try:
                    # exhaust generator to get return value if not already
                    leftover = None
                except Exception:
                    pass
                if final_img is not None:
                    output_img = final_img
                else:
                    # If generator did not yield an Image, assume last yielded thing was image-like
                    output_img = result
            else:
                output_img = result
        except Exception:
            output_img = result
        if output_img is None:
            raise ValueError(f"{renderpass_type} returned None instead of an image")
        if not isinstance(output_img, Image.Image):
            raise ValueError(f"{renderpass_type} returned {type(output_img)} instead of PIL Image")
        if progress_callback:
            progress_callback(f"Completed {renderpass_type}")
    except Exception as e:
        raise RuntimeError(f"Error in render pass '{renderpass_type}': {e}")

    try:
        # Save image and store pass_hash in the slot cache if available
        try:
            saveImageToSlot(output_img, output_slot, slot_table)
            # update pass_hash in cache
            try:
                out_slot_key = None
                try:
                    out_slot_key = Slot.from_value(output_slot).value
                except Exception:
                    out_slot_key = output_slot
                if hasattr(slot_table, '_image_cache') and isinstance(slot_table._image_cache, dict):
                    entry = slot_table._image_cache.get(out_slot_key) or {}
                    entry['pass_hash'] = pass_hash
                    entry['image'] = output_img
                    slot_table._image_cache[out_slot_key] = entry
            except Exception:
                pass
        except Exception:
            # fallback save without caching
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
        total = len(ordered_indices)
        # Notify start of processing with total count
        try:
            if progress_callback:
                progress_callback({'type': 'start', 'total': total, 'message': f'Processing {total} render passes...'})
        except Exception:
            pass

        durations: List[float] = []
        for idx, pass_idx in enumerate(ordered_indices):
            widget = render_pass_widgets[pass_idx]
            pass_index = idx + 1
            # wrapper to translate inner textual messages into structured pass messages
            def _inner_progress(msg_str, _pi=pass_index, _tot=total, _w=widget):
                try:
                    if progress_callback:
                        progress_callback({'type': 'pass_message', 'index': _pi, 'total': _tot, 'pass': getattr(_w, 'renderpass_type', ''), 'message': msg_str})
                except Exception:
                    # swallow progress errors
                    pass

            start_t = time.time()
            try:
                run_render_pass(widget, slot_table, _inner_progress)
            except Exception as e:
                error_msg = f"Error in pass {pass_index} ({widget.renderpass_type}): {str(e)}"
                try:
                    if progress_callback:
                        progress_callback({'type': 'error', 'index': pass_index, 'total': total, 'message': error_msg})
                except Exception:
                    pass
                raise RuntimeError(error_msg)
            elapsed = max(0.0, time.time() - start_t)
            durations.append(elapsed)

            # Compute simple ETA using average of completed durations
            try:
                avg = sum(durations) / len(durations)
                remaining = total - pass_index
                eta = avg * remaining
                percent = int((pass_index / total) * 100)
                if progress_callback:
                    progress_callback({'type': 'progress', 'index': pass_index, 'total': total, 'percent': percent, 'eta': eta, 'message': f'Completed {pass_index}/{total} passes'})
            except Exception:
                try:
                    if progress_callback:
                        progress_callback({'type': 'progress', 'index': pass_index, 'total': total, 'percent': int((pass_index / total) * 100), 'eta': None, 'message': f'Completed {pass_index}/{total} passes'})
                except Exception:
                    pass

        try:
            if progress_callback:
                progress_callback({'type': 'done', 'total': total, 'message': 'All render passes completed successfully'})
        except Exception:
            pass
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

    build_dependency_graph(render_pass_widgets)

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
                # slot0 is always available, so no issue to report
                pass
    return issues


def visualize_pipeline(gui_instance: 'GUI', out_dot: Optional[str] = None, out_png: Optional[str] = None) -> str:
    """Generate a DOT representation of the current render pipeline and
    optionally render it to PNG using Graphviz `dot` if `out_png` is provided.

    Returns the path to the written DOT file.
    """
    lw = gui_instance.listWidget.list_widget
    render_pass_widgets: List[RenderPassWidget] = []
    for i in range(lw.count()):
        widget = lw.itemWidget(lw.item(i))
        render_pass_widgets.append(widget)

    nodes: List[str] = []
    edges: List[str] = []

    # create slot nodes for any referenced slots
    referenced_slots = set()
    for idx, w in enumerate(render_pass_widgets):
        for s in getattr(w, 'selectedInputs', []):
            if s is not None:
                try:
                    referenced_slots.add(Slot.from_value(s).value)
                except Exception:
                    referenced_slots.add(s)
        out = getattr(w, 'selectedOutput', None)
        if out is not None:
            try:
                referenced_slots.add(Slot.from_value(out).value)
            except Exception:
                referenced_slots.add(out)

    for slot in sorted(referenced_slots):
        nodes.append(f'"{slot}" [shape=oval, style=filled, fillcolor="#f0f0f0"];')

    for idx, w in enumerate(render_pass_widgets):
        pname = f'P{idx+1}'
        label = getattr(w, 'renderpass_type', f'Pass {idx+1}')
        nodes.append(f'"{pname}" [shape=box, style=filled, fillcolor="#d0e1f9", label="{label}"];')
        # edges from input slots -> pass
        for s in getattr(w, 'selectedInputs', []):
            if s is None:
                continue
            try:
                sk = Slot.from_value(s).value
            except Exception:
                sk = s
            edges.append(f'"{sk}" -> "{pname}";')
        # edge from pass -> output slot
        out = getattr(w, 'selectedOutput', None)
        if out is not None:
            try:
                ok = Slot.from_value(out).value
            except Exception:
                ok = out
            edges.append(f'"{pname}" -> "{ok}";')

    dot_lines = ['digraph pipeline {', '  rankdir=LR;', '  node [fontname="Helvetica"];']
    dot_lines.extend('  ' + n for n in nodes)
    dot_lines.extend('  ' + e for e in edges)
    dot_lines.append('}')

    if not out_dot:
        out_dot = os.path.join('saved', 'pipeline.dot')
    os.makedirs(os.path.dirname(out_dot), exist_ok=True)
    with open(out_dot, 'w', encoding='utf-8') as f:
        f.write('\n'.join(dot_lines))

    # Try to render PNG if requested and `dot` is available
    if out_png:
        try:
            import subprocess
            subprocess.run(['dot', '-Tpng', out_dot, '-o', out_png], check=True)
        except Exception:
            # Ignore rendering errors; DOT file is still written
            pass

    return out_dot
