# scripts/nodegraph_ui/nodes.py

from .classes import InputNode, ProcessorNode, OutputNode, SocketType, InputSocket, OutputSocket
from PIL import Image
import time
import os

from .. import passes

# Math nodes imported to keep nodes.py organized
from .nodes_math import *

class SourceImageNode(InputNode):
    def __init__(self):
        """
        Initialize a "Source Image" node instance.
        Sets up metadata and I/O for a node that provides a source image from an assets/images
        directory. Attributes created:
        - display_name (str): Human-readable name ("Source Image").
        - category (str): Node category ("Base Nodes").
        - description (str): Short description of the node's purpose.
        - tooltips_in (dict): Mapping of input socket names to tooltip strings (empty for this node).
        - tooltips_out (dict): Mapping of output socket names to tooltip strings (contains "image").
        - index (int): Numeric index for the node (initialized to 0).
        - inputs (dict): Input sockets mapping (empty for this source node).
        - outputs (dict): Output sockets mapping; provides an "image" OutputSocket that emits a PIL image.
        - _images (list[PIL.Image.Image]): Loaded in-memory images (initially empty).
        - _image_files (list[str]): File paths for available images (initially empty).
        Side effects:
        - Calls self._load_images() to populate _images and _image_files from the assets/images
            directory. Implementation of _load_images() determines error handling and file I/O behavior.
        """
        super().__init__()
        self.display_name = "Source Image"
        self.category = "Base Nodes"
        self.description = "Provides a source image from the assets/images directory."
        self.tooltips_in = {}
        self.tooltips_out = {
            "image": "Output image from the source node."
        }

        self.index = 0

        self.inputs = {} # source node has no inputs
        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )

        self._images: list[Image.Image] = []
        self._image_files: list[str] = []
        self._load_images()
    
    def _load_images(self):
        self._images.clear()
        self._image_files.clear()
        for file in sorted(os.listdir("assets/images")):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                img = Image.open(os.path.join("assets/images", file)).convert("RGB")
                self._images.append(img)
                self._image_files.append(file)
    
    def compute(self):
        if not self._images:
            self.outputs["image"]._cache = None
            return
        
        idx = max(0, min(self.index, len(self._images) - 1))
        self.outputs["image"]._cache = self._images[idx]

class BlurNode(ProcessorNode):
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

class KuwaharaNode(ProcessorNode):
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

class InvertNode(ProcessorNode):
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

class RescaleNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Rescale Image"
        self.category = "Base Nodes"
        self.description = "Rescales the input image to the specified resolution."
        self.tooltips_in = {
            "image": "Input image to be rescaled.",
            "resolution": "Target resolution in WIDTHxHEIGHT format (e.g., '800x600'), or target width."
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
        print(f"RescaleNode: starting compute with args: image={self.inputs['image'].get()}, resolution={self.inputs['resolution'].get()}, allow_upscale={self.inputs['allow_upscale'].get()}‼")
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
        # mark downstream nodes as dirty
        for dep in self.dependents:
            dep.mark_dirty()

class ContrastMaskNode(ProcessorNode):
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

class LuminanceMaskNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.category = "Mask Nodes"
        self.display_name = "Luminance Mask"
        self.description = "Generates a luminance mask from the input image based on specified minimum and maximum luminance limits."
        self.tooltips_in = {
            "image": "Input image to generate luminance mask from."
        }
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )

        self.inputs["mask"] = InputSocket(
            self, "mask", SocketType.PIL_IMG_MONOCH, is_optional=True
        )

        self.outputs["mask"] = OutputSocket(
            self, "mask", SocketType.PIL_IMG_MONOCH
        )
    def compute(self):
        img = self.inputs["image"].get()
        mask_input = self.inputs["mask"].get()
        if img is None:
            self.outputs["mask"]._cache = None
            return

        try:
            print(f"LuminanceMaskNode: computing luminance mask ")
            if mask_input is not None:
                mask = passes.generate_luminance_mask(img, mask=mask_input)
            else:
                mask = passes.generate_luminance_mask(img)
        except Exception as e:
            import traceback
            print(f"LuminanceMaskNode: Luminance mask failed ({e}), using blank mask.")
            traceback.print_exc()
            mask = Image.new("L", img.size, 0)

        self.outputs["mask"]._cache = mask

class NoiseNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Add Noise"
        self.category = "Generative Nodes"
        self.description = "Adds noise to the input image based on specified amount and noise type."
        self.tooltips_in = {
            "image": "Input image to add noise to.",
            "amount": "Amount of noise to add (0.0 to 1.0).",
            "noise_type": "Type of noise to add [ GAUSSIAN | SALT_PEPPER | SHOT | UNIFORM | 50_PER_CENT | PERLIN | BLUE | PINK | WHITE ].",
            "size": "Scale factor for noise types that support size (e.g., Perlin). Ignored by other noise types."
        }
        self.tooltips_out = {
            "image": "Output image with added noise."
        }
        
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )

        self.inputs["amount"] = InputSocket(
            self, "amount", SocketType.FLOAT
        )

        self.inputs["noise_type"] = InputSocket(
            self, "noise_type", SocketType.STRING
        )

        self.inputs["size"] = InputSocket(
            self, "size", SocketType.FLOAT, is_optional=True
        )

        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )
    
    def compute(self):
        img = self.inputs["image"].get()
        amount = self.inputs["amount"].get()
        noise_type = self.inputs["noise_type"].get()
        if img is None:
            self.outputs["image"]._cache = None
            return

        if amount is None:
            amount = 0.05
        if noise_type is None:
            noise_type = "GAUSSIAN"

        try:
            size = self.inputs["size"].get()
            print(f"NoiseNode: computing noise type={noise_type} amount={amount} size={size}")
            noised = passes.noise(img, amount=float(amount), noise_type=noise_type, size=size)
        except Exception as e:
            import traceback
            print(f"NoiseNode: Noise addition failed ({e}), using original image.")
            traceback.print_exc()
            noised = img

        self.outputs["image"]._cache = noised

class DifferenceNode(ProcessorNode):
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

class DitherNode(ProcessorNode):
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

        self.outputs["image"]._cache = dithered

class OutlineNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Outline"
        self.category = "Image Effects"
        self.description = "Detects edges in the input image and produces an outline effect based on specified parameters."
        self.tooltips_in = {
            "image": "Input image to be processed for edge detection.",
            "method": "Edge detection method to use [ SOBEL / CANNY / DOG ].",
            "threshold": "Threshold value for edge detection sensitivity. Range: 0.0 to 1.0 (float). Values outside this range will be clipped internally.",
            "limLower": "Low threshold for edge detection (optional). Range: 0.0 to 255.0 (float) or None. If None, computed as threshold * 255.",
            "limUpper": "High threshold for edge detection (optional). Range: 0.0 to 255.0 (float) or None. If None, computed as min(255, low_threshold * 2)."
        }

        self.tooltips_out = {
            "image": "Output image with detected edges outlined."
        }
        
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )

        self.inputs["method"] = InputSocket(
            self, "method", SocketType.STRING
        )

        self.inputs["threshold"] = InputSocket(
            self, "threshold", SocketType.FLOAT
        )

        self.inputs["limLower"] = InputSocket(
            self, "limLower", SocketType.FLOAT
        )

        self.inputs["limUpper"] = InputSocket(
            self, "limUpper", SocketType.FLOAT
        )

        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )
    
    def compute(self):
        img = self.inputs["image"].get()
        limLower = self.inputs["limLower"].get()
        limUpper = self.inputs["limUpper"].get()
        if img is None:
            self.outputs["image"]._cache = None
            return
        if limLower is None:
            limLower = 0.1
        if limUpper is None:
            limUpper = 0.3

        try:
            print(f"OutlineNode: computing outline image")
            method = self.inputs["method"].get()
            threshold = self.inputs["threshold"].get()
            if method is None:
                method = "Sobel"
            if threshold is None:
                threshold = 0.1
            outlined = passes.find_edges(
                img,
                low_threshold=limLower,
                high_threshold=limUpper,
                type=method,
                threshold=threshold
                )
        except Exception as e:
            import traceback
            print(f"OutlineNode: Outline computation failed ({e}), using original image.")
            traceback.print_exc()
            outlined = img

        self.outputs["image"]._cache = outlined

class ValueIntNode(InputNode):
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

class ValueFloatNode(InputNode):
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

class ValueStringNode(InputNode):
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
    
class ValueBoolNode(InputNode):
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

class ValueColorNode(InputNode):
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

class GetImageDataNode(ProcessorNode):
    @property
    def is_soft_computation(self):
        return True
    def __init__(self):
        super().__init__()
        self.display_name = "Get Image Data"
        self.category = "Miscellaneous Nodes"
        self.description = "Retrieves the width and height, mode info and format of the input image."
        self.tooltips_in = {
            "image": "Input image to get data from."
        }
        self.tooltips_out = {
            "width": "Width of the input image in pixels.",
            "height": "Height of the input image in pixels.",
            "mode": "Color mode of the input image (e.g., RGB, CMYK).",
            "format": "File format of the input image (e.g., PNG, JPEG).",
            "info": "Additional info dictionary of the input image."
        }

        try:
            self.inputs = {}
            self.outputs = {}

            self.inputs["image"] = InputSocket(
                self,
                "image", SocketType.PIL_IMG
            )

            self.outputs["width"] = OutputSocket(
                self,
                "width", SocketType.INT
            )

            self.outputs["height"] = OutputSocket(
                self,
                "height", SocketType.INT
            )

            self.outputs["mode"] = OutputSocket(
                self,
                "mode", SocketType.STRING
            )

            self.outputs["format"] = OutputSocket(
                self,
                "format", SocketType.STRING
            )

            self.outputs["info"] = OutputSocket(
                self,
                "info", SocketType.STRING
            )
        except Exception as e:
            print(f"GetImageDataNode: Initialization failed ({e}).")

    # Use default mark_dirty from ProcessorNode/Node to avoid recompute oscillation

    def compute(self):
        img = self.inputs["image"].get(allow_hard=False)
        if img is None:
            self.outputs["width"]._cache = None
            self.outputs["height"]._cache = None
            self.outputs["mode"]._cache = None
            self.outputs["format"]._cache = None
            self.outputs["info"]._cache = None
            print("GetImageDataNode: No input image.")
            return
        try:
            data = passes.getImageData(img)
            self.outputs["width"]._cache = data["width"]
            self.outputs["height"]._cache = data["height"]
            self.outputs["mode"]._cache = data["mode"]
            self.outputs["format"]._cache = data["format"]
            self.outputs["info"]._cache = data["info"]
        except Exception as e:
            print(f"GetImageDataNode: Failed to get image data ({e}).")
            self.outputs["width"]._cache = None
            self.outputs["height"]._cache = None
            self.outputs["mode"]._cache = None
            self.outputs["format"]._cache = None
            self.outputs["info"]._cache = None
    
class ViewerNode(OutputNode):
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

class RenderToFileNode(OutputNode):
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

class DisplayDataNode(OutputNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Display Data"
        self.category = "Miscellaneous Nodes"
        self.description = "Displays input data"
        self.tooltips_in = {
            "data": "Input data to display."
        }
        try:
            self.inputs = {}
            self.inputs["data"] = InputSocket(
                self, "data", SocketType.UNDEFINED
            )
        except Exception as e:
            print(f"DisplayDataNode: Initialization failed ({e}).")
    def compute(self):
        try:
            data = self.inputs["data"].get()
            print(f"DisplayDataNode: Input data = {data}")
        except Exception as e:
            print(f"DisplayDataNode: Failed to get input data ({e}).")
        pass

class ToStringNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "To String"
        self.category = "Miscellaneous Nodes"
        self.description = "Converts the input data to a string representation."
        self.tooltips_in = {
            "data": "Input data to convert to string."
        }
        self.tooltips_out = {
            "string": "String representation of the input data."
        }
        try:
            self.inputs = {}
            self.outputs = {}

            self.inputs["data"] = InputSocket(
                self,
                "data", SocketType.UNDEFINED
            )

            self.outputs["string"] = OutputSocket(
                self,
                "string", SocketType.STRING
            )
        except Exception as e:
            print(f"ToStringNode: Initialization failed ({e}).")

    def compute(self):
        data = self.inputs["data"].get()
        try:
            string_repr = str(data)
        except Exception as e:
            print(f"ToStringNode: Conversion to string failed ({e}).")
            string_repr = ""
        self.outputs["string"]._cache = string_repr