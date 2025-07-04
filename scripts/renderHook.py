import os
from PIL import Image
import passes
import inspect


def load_image_from_slot(slot_name, slot_table):
    return slot_table.get_image(slot_name)


def save_image_to_slot(image: Image.Image, slot_name, slot_table):
    slot_table.set_image(slot_name, image)


def run_render_pass(render_pass_widget, slot_table):
    """
    Run a single render pass widget with dynamic function loading.
    """
    renderpass_type = render_pass_widget.renderpass_type
    inputs = render_pass_widget.selected_inputs
    output_slot = render_pass_widget.selected_output
    settings = render_pass_widget.get_settings()

    # Funktion suchen im passes-Modul
    func_name = renderpass_type.replace(" ", "_")
    if not hasattr(passes, func_name):
        raise NotImplementedError(f"Renderpass '{renderpass_type}' not implemented.")

    func = getattr(passes, func_name)
    sig = inspect.signature(func)

    # Bilder aus Input-Slots laden
    input_images = []
    for slot in inputs:
        if slot is None:
            raise ValueError(f"Input slot not set for render pass {renderpass_type}")
        img = load_image_from_slot(slot, slot_table)
        if img is None:
            raise ValueError(f"No image found in slot '{slot}'")
        input_images.append(img)

    # Argumente automatisch zusammenstellen
    args = []
    img_index = 0

    for param in sig.parameters.values():
        if param.annotation == Image.Image or param.name.startswith("img"):
            # Bildparameter
            if img_index >= len(input_images):
                raise ValueError(f"Not enough input images for {renderpass_type}")
            args.append(input_images[img_index])
            img_index += 1
        elif param.name in settings:
            val = settings[param.name]
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
                raise ValueError(f"Could not convert setting '{param.name}': {e}")
            args.append(val)
        elif param.default is not inspect.Parameter.empty:
            args.append(param.default)
        else:
            raise ValueError(f"Missing required parameter '{param.name}' for {renderpass_type}")

    # Funktion ausführen
    try:
        output_img = func(*args)
    except Exception as e:
        raise RuntimeError(f"Error in render pass '{renderpass_type}': {e}")

    if output_slot is None:
        raise ValueError(f"Output slot not set for render pass {renderpass_type}")
    save_image_to_slot(output_img, output_slot, slot_table)

    return output_img


def run_all_render_passes(gui_instance):
    """
    Run all render passes in the GUI's list widget in order.
    """
    lw = gui_instance.list_widget.list_widget
    slot_table = gui_instance.slot_table

    for i in range(lw.count()):
        widget = lw.itemWidget(lw.item(i))
        run_render_pass(widget, slot_table)
