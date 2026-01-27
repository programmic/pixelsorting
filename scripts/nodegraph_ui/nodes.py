# scripts/nodegraph_ui/nodes.py

from .classes import Node, SocketType, InputSocket, OutputSocket
from PIL import Image
import time
import os

from .. import passes

class SourceImageNode(Node):
    """A simple source node that loads images from `assets/images`.

    This implementation keeps a list of loaded PIL images (`_images`) and
    a parallel list of shortened display names (`_image_files`) used by the UI
    combo box. The full file paths are stored in `_image_paths` for debugging
    or future use.
    """
    def __init__(self):
        super().__init__()
        self.display_name = "Source Image"
        self.category = "Base Nodes"
        self.description = "Provides a source image from the assets/images directory."
        self.tooltips_in = {}
        self.tooltips_out = {"image": "Output image from the source node."}

        self.index = 0

        # no inputs for a source node
        self.inputs = {}
        self.outputs["image"] = OutputSocket(self, "image", SocketType.PIL_IMG)

        # Internal lists: full paths, loaded PIL images, and shortened display names
        self._image_paths = []
        self._images = []
        self._image_files = []

        self._load_images()

    def _short_name(self, name: str, maxlen: int = 14) -> str:
        if not name:
            return ""
        if len(name) <= maxlen:
            return name
        # keep start and end with ellipsis in the middle
        part = max(4, (maxlen - 1) // 2)
        return f"{name[:part]}…{name[-part:]}"

    def _load_images(self):
        self._image_paths.clear()
        self._images.clear()
        self._image_files.clear()
        try:
            image_dir = os.path.join(os.getcwd(), "assets", "images")
            if not os.path.isdir(image_dir):
                return
            for fname in sorted(os.listdir(image_dir)):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    continue
                full = os.path.join(image_dir, fname)
                try:
                    img = Image.open(full).convert("RGB")
                except Exception:
                    continue
                self._image_paths.append(full)
                self._images.append(img)
                # display name: file name without extension, shortened to fit
                try:
                    base = os.path.splitext(fname)[0]
                except Exception:
                    base = fname
                self._image_files.append(self._short_name(base, maxlen=14))
        except Exception:
            # be resilient on load errors
            pass

    def compute(self):
        if not self._images:
            self.outputs["image"]._cache = None
            return
        idx = int(getattr(self, "index", 0) or 0)
        idx = max(0, min(idx, len(self._images) - 1))
        try:
            self.outputs["image"]._cache = self._images[idx]
        except Exception:
            self.outputs["image"]._cache = None

class BlurNode(Node):
    def __init__(self):
        super().__init__()
        self.display_name = "Blur Image"
        self.category = "Image Effects"
        self.description = "Applies a blur effect to the input image using the specified blur type and kernel size."
        self.tooltips_in = {
            "image": "Input image to be blurred.",
            "mask": "Optional mask to control where the blur is applied.",
            "blur_type": "Type of blur to apply ('Gaussian', 'Box').",
            "kernel": "Size of the blur kernel (odd integer)."
        }
        self.tooltips_out = {
            "image": "Output blurred image."
        }

        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )
        self.inputs["mask"] = InputSocket(
            self, "mask", SocketType.PIL_IMG_MONOCH, is_optional=True
        )
        self.inputs["blur_type"] = InputSocket(
            self, "blur_type", SocketType.STRING
        )
        self.inputs["kernel"] = InputSocket(
            self, "kernel", SocketType.INT
        )

        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )
    
    def compute(self):
        img = self.inputs["image"].get()
        mask = self.inputs["mask"].get()
        blur_type = self.inputs["blur_type"].get()
        kernel = self.inputs["kernel"].get()

        # Provide sensible defaults when optional control inputs are not connected
        if blur_type is None:
            blur_type = "Gaussian"
        if kernel is None:
            try:
                kernel = 3
            except Exception:
                kernel = 3

        # If required image input is missing, ensure output cache is cleared
        if img is None:
            self.outputs["image"]._cache = None
            return

        blurred = None
        try:
            blurred = passes.blur(img, blur_type, int(kernel))
        except Exception as e:
            # Primary blur failed (possibly GPU/OpenCL). Try CPU fallbacks.
            try:
                if blur_type == "Gaussian":
                    try:
                        blurred = passes.blur_gaussian_fast(img, int(kernel))
                    except Exception:
                        blurred = passes.blur_gaussian(img, int(kernel))
                elif blur_type == "Box":
                    try:
                        blurred = passes.blur_box(img, int(kernel))
                    except Exception:
                        blurred = img
                else:
                    try:
                        blurred = passes.blur_box(img, int(kernel))
                    except Exception:
                        blurred = img
            except Exception:
                blurred = img

        if mask:
            try:
                blurred = Image.composite(blurred, img, mask)
            except Exception:
                pass

        self.outputs["image"]._cache = blurred
#img: Image.Image,
#       kernel: int,
#       regions: int = 8,
#       isAnisotropic: bool = False,
#       stylePapari: bool = False,
#       progress: Optional[callable] = None
#       ) -> Image.Image:

class KuwaharaNode(Node):
    def __init__(self):
        super().__init__()
        self.display_name = "Kuwahara Filter"
        self.category = "Image Effects"
        self.description = "Applies a Kuwahara filter to the input image for artistic smoothing."
        self.tooltips_in = {
            "image": "Input image to be processed with the Kuwahara filter.",
            "radius": "Radius of the Kuwahara filter effect.",
            "isAnisotropic": "If true, applies anisotropic Kuwahara filtering.",
            "stylePapari": "If true, uses the Papari-style Kuwahara filter.",
            "regions": "Number of regions to consider in the Kuwahara filter when using anisotropic / papari mode."
        }
        self.tooltips_out = {
            "image": "Output image after applying the Kuwahara filter."
        }
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )
        self.inputs["radius"] = InputSocket(
            self, "radius", SocketType.INT
        )
        self.inputs["isAnisotropic"] = InputSocket(
            self, "isAnisotropic", SocketType.BOOLEAN
        )
        self.inputs["stylePapari"] = InputSocket(
            self, "stylePapari", SocketType.BOOLEAN
        )
        self.inputs["regions"] = InputSocket(
            self, "regions", SocketType.INT
        )
        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )

    
    def compute(self):
        img = self.inputs["image"].get()
        radius = self.inputs["radius"].get()
        is_anisotropic = self.inputs["isAnisotropic"].get()
        style_papari = self.inputs["stylePapari"].get()
        if style_papari is True:
            is_anisotropic = True
        regions = self.inputs["regions"].get()

        if img is None:
            self.outputs["image"]._cache = None
            return

        if radius is None:
            radius = 5

        try:
            print(f"KuwaharaNode: computing Kuwahara filter with radius={radius}")
            filtered = passes.kuwahara_wrapper(
                img, int(radius),
                regions=int(regions),
                isAnisotropic=bool(is_anisotropic),
                stylePapari=bool(style_papari)
    )
        except Exception as e:
            import traceback
            print(f"KuwaharaNode: Kuwahara filter failed ({e}), using original image.")
            traceback.print_exc()
            filtered = img

        self.outputs["image"]._cache = filtered

class InvertNode(Node):
    def __init__(self):
        super().__init__()
        self.display_name = "Invert Colors"
        self.category = "Image Effects"
        self.description = "Inverts the colors of the input image based on specified type and impact factor."
        self.tooltips_in = {
            "image": "Input image to be color inverted.",
            "invertType": "Type of inversion (e.g., 'RGB', 'R', 'G', 'B', 'Lum').",
            "impactFactor": "Impact factor for inversion strength (0 to 100)."
        }
        self.tooltips_out = {
            "image": "Output image with colors inverted."
        }
        
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )

        self.inputs["invertType"] = InputSocket(
            self, "invertType", SocketType.STRING
        )

        self.inputs["impactFactor"] = InputSocket(
            self, "impactFactor", SocketType.FLOAT
        )

        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )
    
    def compute(self):
        img = self.inputs["image"].get()
        invert_type = self.inputs["invertType"].get()
        impact_factor = self.inputs["impactFactor"].get()
        if img is None:
            self.outputs["image"]._cache = None
            print("InvertNode: No input image.")
            return

        if invert_type is None:
            invert_type = "RGB"
        if impact_factor is None:
            impact_factor = 1.0

        # normalize impact_factor: accept 0..1 or 0..100 ranges
        try:
            if isinstance(impact_factor, str):
                impact_factor = float(impact_factor)
        except Exception:
            try:
                impact_factor = float(impact_factor)
            except Exception:
                impact_factor = 1.0

        try:
            if 0.0 <= float(impact_factor) <= 1.0:
                # interpret as 0..1 -> convert to 0..100
                impact_factor = float(impact_factor) * 100.0
            else:
                impact_factor = float(impact_factor)
        except Exception:
            impact_factor = 100.0

        try:
            print(f"InvertNode: computing invert type={invert_type} impact={impact_factor}")
            inverted = passes.invert(img, invert_type, impact_factor=float(impact_factor))
        except Exception as e:
            import traceback
            print(f"InvertNode: Inversion failed ({e}), using original image.")
            traceback.print_exc()
            inverted = img

        self.outputs["image"]._cache = inverted

class RescaleNode(Node):
    def __init__(self):
        super().__init__()
        self.display_name = "Rescale Image"
        self.category = "Base Nodes"
        self.description = "Rescales the input image to the specified resolution."
        self.tooltips_in = {
            "image": "Input image to be rescaled.",
            "resolution": "Target resolution in WIDTHxHEIGHT format (e.g., '800x600')."
        }
        self.tooltips_out = {
            "image": "Output rescaled image."
        }
        
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )

        self.inputs["resolution"] = InputSocket(
            self, "resolution", SocketType.STRING
        )

        self.inputs["allow_upscale"] = InputSocket(
            self, "allow_upscale", SocketType.BOOLEAN
        )

        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )
    
    def compute(self):
        img = self.inputs["image"].get()
        resolution = self.inputs["resolution"].get()
        if img is None:
            self.outputs["image"]._cache = None
            return

        if resolution is None:
            resolution = "1920x1080"

        try:
            print(f"RescaleNode: computing rescale to resolution={resolution}")
            reduced = passes.reduce_size(img, resolution=resolution, upscale=bool(self.inputs["allow_upscale"].get()))
            try:
                print(f"RescaleNode: result size={getattr(reduced, 'size', None)}")
            except Exception:
                pass
        except Exception as e:
            import traceback
            print(f"RescaleNode: Rescale failed ({e}), using original image.")
            traceback.print_exc()
            reduced = img

        self.outputs["image"]._cache = reduced

class ContrastMaskNode(Node):
    # args: (img: Image.Image, limMin: int, limMax: int)
    def __init__(self):
        super().__init__()
        self.category = "Mask Nodes"
        self.display_name = "Contrast Mask"
        self.description = "Generates a contrast mask from the input image based on specified minimum and maximum contrast limits."
        self.tooltips_in = {
            "image": "Input image to generate contrast mask from.",
            "limMin": "Minimum contrast limit.",
            "limMax": "Maximum contrast limit."
        }
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )

        self.inputs["limMin"] = InputSocket(
            self, "limMin", SocketType.INT
        )

        self.inputs["limMax"] = InputSocket(
            self, "limMax", SocketType.INT
        )

        self.outputs["mask"] = OutputSocket(
            self, "mask", SocketType.PIL_IMG_MONOCH
        )
    def compute(self):
        img = self.inputs["image"].get()
        limMin = self.inputs["limMin"].get()
        limMax = self.inputs["limMax"].get()
        if img is None:
            self.outputs["mask"]._cache = None
            return

        if limMin is None:
            limMin = 30
        if limMax is None:
            limMax = 60

        try:
            print(f"ContrastMaskNode: computing contrast mask limMin={limMin} limMax={limMax}")
            mask = passes.generate_contrast_mask(img, int(limMin), int(limMax))
        except Exception as e:
            import traceback
            print(f"ContrastMaskNode: Contrast mask failed ({e}), using blank mask.")
            traceback.print_exc()
            mask = Image.new("L", img.size, 0)

        self.outputs["mask"]._cache = mask

class DifferenceNode(Node):
    def __init__(self):
        super().__init__()
        self.display_name = "Difference"
        self.description = "Computes the difference between two input images."
        self.tooltips_in = {
            "imageA": "First input image for difference computation.",
            "imageB": "Second input image for difference computation."
        }
        self.tooltips_out = {
            "image": "Output image representing the difference between the two inputs."
        }
        
        self.inputs["imageA"] = InputSocket(
            self, "imageA", SocketType.PIL_IMG
        )

        self.inputs["imageB"] = InputSocket(
            self, "imageB", SocketType.PIL_IMG
        )

        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )
    
    def compute(self):
        print("DifferenceNode: starting compute")
        imgA = self.inputs["imageA"].get()
        imgB = self.inputs["imageB"].get()
        if imgA is None or imgB is None:
            self.outputs["image"]._cache = None
            return

        try:
            print(f"DifferenceNode: computing difference between two images")
            diff = passes.difference(imgA, imgB)
        except Exception as e:
            import traceback
            print(f"DifferenceNode: Difference computation failed ({e}), using blank image.")
            traceback.print_exc()
            diff = Image.new("RGB", imgA.size, (0, 0, 0))

        self.outputs["image"]._cache = diff

class DitherNode(Node):
    def __init__(self):
        super().__init__()
        self.display_name = "Dither Image"
        self.category = "Image Effects"
        self.description = "Applies dithering to the input image."
        self.tooltips_in = {
            "image": "Input image to be dithered.",
            "method": "Dithering method to use [ FLOYD_STEINBERG / ATKINSON ].",
            "num_colors": "Number of colors to reduce the image to.",
            "palette_selection": "Method for selecting the color palette [median_cut / most_represented / most_different].",
            "palette": "Optional custom color palette to use for dithering."
        }
        self.tooltips_out = {
            "image": "Output dithered image."
        }
        
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )

        self.inputs["method"] = InputSocket(
            self, "method", SocketType.STRING
        )

        self.inputs["num_colors"] = InputSocket(
            self, "num_colors", SocketType.INT
        )

        self.inputs["palette_selection"] = InputSocket(
            self, "palette_selection", SocketType.STRING
        )

        self.inputs["palette"] = InputSocket(
            self, "palette", SocketType.LIST_COLORS
        )

        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )
    
    def compute(self):
        img = self.inputs["image"].get()
        if img is None:
            self.outputs["image"]._cache = None
            return

        try:
            print(f"DitherNode: computing dithered image")
            method = self.inputs["method"].get()
            num_colors = self.inputs["num_colors"].get()
            palette_selection = self.inputs["palette_selection"].get()
            if method is None:
                method = "FLOYD_STEINBERG"
            if num_colors is None:
                num_colors = 16
            if palette_selection is None:
                palette_selection = "median_cut"
            dithered = passes.dither(img, num_colors=num_colors, method=method, palette_selection=palette_selection)
        except Exception as e:
            import traceback
            print(f"DitherNode: \033[31m[ERROR]\033[0m: Dithering failed ({e}), using original image.")
            traceback.print_exc()
            dithered = img

        print(f"DitherNode: dithered image size={getattr(dithered, 'size', None)}")
        self.outputs["image"]._cache = dithered

class ValueIntNode(Node):
    def __init__(self):
        super().__init__()
        self.category = "Input Nodes"
        self.display_name = "Integer Value"
        
        self.outputs["value"] = OutputSocket(
            self, "value", SocketType.INT, is_modifiable=True
        )
        self.value: int = 0
    
    def compute(self):
        self.outputs["value"]._cache = self.value

class ValueFloatNode(Node):
    def __init__(self):
        super().__init__()
        self.category = "Input Nodes"
        self.display_name = "Float Value"
        self.outputs["value"] = OutputSocket(
            self, "value", SocketType.FLOAT, is_modifiable=True
        )
        self.value: float = 0.0
    
    def compute(self):
        self.outputs["value"]._cache = self.value

class ValueStringNode(Node):
    def __init__(self):
        super().__init__()
        self.category = "Input Nodes"
        self.display_name = "String Value"
        
        self.outputs["value"] = OutputSocket(
            self, "value", SocketType.STRING, is_modifiable=True
        )
        self.value: str = ""
    
    def get(self):
        return self.value
    def set(self, val: str):
        self.value = val

    def compute(self):
        self.outputs["value"]._cache = self.value
    

class ValueBoolNode(Node):
    def __init__(self):
        super().__init__()
        self.category = "Input Nodes"
        self.display_name = "Boolean Value"
        self.outputs["value"] = OutputSocket(
            self, "value", SocketType.BOOLEAN, is_modifiable=True
        )
        self.value: bool = False
    
    def compute(self):
        self.outputs["value"]._cache = self.value

class ValueColorNode(Node):
    def __init__(self):
        super().__init__()
        self.category = "Input Nodes"
        self.display_name = "Color Value"
        self.outputs["value"] = OutputSocket(
            self, "value", SocketType.COLOR, is_modifiable=True
        )
        self.value: tuple[int, int, int] = (255, 255, 255)
    
    def compute(self):
        self.outputs["value"]._cache = self.value


class ViewerNode(Node):
    def __init__(self):
        super().__init__()
        self.category = "Base Nodes"
        self.display_name = "Viewer"
        self.description = "Displays the input image in the viewer panel."
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )

    def compute(self):
        # viewer doesn't produce outputs; it just reads the input
        img = self.inputs["image"].get()
        # store into a cache attribute for potential UI use
        self._last_image = img

class RenderToFileNode(Node):
    def __init__(self):
        super().__init__()
        self.category = "Base Nodes"
        self.display_name = "Render to File"
        self.description = "Saves the input image to a specified file path."
        self.tooltips_in = {
            "image": "Input image to be saved to file.",
            "filepath": "File path where the image will be saved. If unset, a default path will be used.",
            "disableFetchOnly": "If true, disables fetch-only mode to actually run previous render steps instead of just fetching cached results. Default is false."
        }
        
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )

        self.inputs["filepath"] = InputSocket(
            self, "filepath", SocketType.STRING
        )

        self.inputs["disableFetchOnly"] = InputSocket(
            self, "disableFetchOnly", SocketType.BOOLEAN
        )
    
    def compute(self):
        if self.inputs["disableFetchOnly"].get() is True:
            self.graph.fetch_only = False
        img = self.inputs["image"].get()
        filepath = self.inputs["filepath"].get()
        if img is None:
            return
        if filepath is None:
            filepath = f"assets/printouts/output_{time.time()}.png"

        try:
            print(f"RenderToFileNode: saving image to {filepath}")
            img.save(filepath)
        except Exception as e:
            import traceback
            print(f"RenderToFileNode: Saving image failed ({e}).")
            traceback.print_exc()