# scripts/nodegraph_ui/nodes.py

from scripts.nodegraph_ui.classes import InputNode, ProcessorNode, OutputNode, SocketType, InputSocket, OutputSocket
from PIL import Image
import time
import os

from scripts import passes

# Math nodes imported to keep nodes.py organized
from scripts.nodegraph_ui.nodes_math import *

# Image Processing Nodes

class BlurNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Blur Image"
        self.category = "Image Effects"
        self.description = "Applies a blur effect to the input image using the specified blur type and kernel size."
        self.progress = 0
        self.tooltips_in = {
            "image": "Input image to be blurred.",
            "mask": "Optional mask to control where the blur is applied.",
            "blur_type": "Type of blur to apply ('Gaussian', 'Box').",
            "kernel": "Size of the blur kernel (odd integer)."
        }
        self.tooltips_out = {
            "image": "Output blurred image."
        }
        self.silent = False  # set to True to suppress print statements for debugging

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
        # caching to avoid unnecessary recompute when inputs unchanged
        self._last_img_id = None
        self._last_resolution = None
        self._last_allow_upscale = None
        # caching to avoid unnecessary recompute when inputs unchanged
        self._last_img_id = None
        self._last_resolution = None
        self._last_allow_upscale = None
    
    def compute(self):
        img = self.inputs["image"].get()
        mask = self.inputs["mask"].get()
        blur_type = self.inputs["blur_type"].get()
        kernel = self.inputs["kernel"].get()
        self.progress = 0

        if not self.silent: print(f"[BlurNode]: compute called with blur_type={blur_type} kernel={kernel}")

        # Provide sensible defaults when optional control inputs are not connected
        if blur_type is None:
            blur_type = "Gaussian"
        if kernel is None:
                kernel = 3

        # If required image input is missing, ensure output cache is cleared
        if img is None:
            self.outputs["image"]._cache = None
            return

        blurred = None
        try:
            if not self.silent: print(f"[BlurNode]: computing blur with type={blur_type} kernel={kernel}")
            self.progress = 0
            blurred = passes.blur(img, blur_type, int(kernel), progress=lambda pct, msg: setattr(self, 'progress', pct))
        except Exception as e:
            # Primary blur failed (possibly GPU/OpenCL). Try CPU fallbacks.
            try:
                if blur_type == "Gaussian":
                    try:
                        if not self.silent: print(f"[BlurNode]: Primary Gaussian blur failed ({e}), attempting fast CPU fallback.")
                        blurred = passes.blur_gaussian_fast(img, int(kernel), progress=lambda pct, msg: setattr(self, 'progress', pct))
                    except Exception:
                        if not self.silent: print(f"[BlurNode]: Fast Gaussian blur failed, attempting standard CPU fallback.")
                        blurred = passes.blur_gaussian(img, int(kernel), progress=lambda pct, msg: setattr(self, 'progress', pct))
                elif blur_type == "Box":
                    try:
                        if not self.silent: print(f"[BlurNode]: Attempting box blur.")
                        blurred = passes.blur_box(img, int(kernel), progress=lambda pct, msg: setattr(self, 'progress', pct))
                    except Exception:
                        if not self.silent: print(f"[BlurNode]: Box blur failed.")
                        blurred = img
                else:
                    try:
                        if not self.silent: print(f"[BlurNode]: Unknown blur type '{blur_type}', defaulting to box blur.")
                        blurred = passes.blur_box(img, int(kernel), progress=lambda pct, msg: setattr(self, 'progress', pct))
                    except Exception:
                        if not self.silent: print(f"[BlurNode]: Box blur failed.")
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
        self.progress = 0
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
                stylePapari=bool(style_papari),
                progress=lambda pct, msg: setattr(self, 'progress', pct)
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
        self.progress = 0
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
    
        self._last_img_id = None
        self._last_resolution = None
        self._last_allow_upscale = None

    def compute(self):
        img = self.inputs["image"].get()
        resolution = self.inputs["resolution"].get()
        if img is None:
            self.outputs["image"]._cache = None
            return

        if resolution is None:
            resolution = "1920x1080"

        # if nothing changed since last compute, skip expensive resize
        current_img_id = id(img)
        current_allow_upscale = bool(self.inputs["allow_upscale"].get())
        if (self._last_img_id == current_img_id and
            self._last_resolution == resolution and
            self._last_allow_upscale == current_allow_upscale):
            return

        self.progress = 0
        try:
            print(f"RescaleNode: computing rescale to resolution={resolution}")
            reduced = passes.reduce_size(img, resolution=resolution, upscale=current_allow_upscale, progress=lambda pct, msg: setattr(self, 'progress', pct))
            try:
                print(f"RescaleNode: result size={getattr(reduced, 'size', None)}")
            except Exception:
                pass
        except Exception as e:
            import traceback
            print(f"RescaleNode: Rescale failed ({e}), using original image.")
            traceback.print_exc()
            reduced = img

        # update cache and mark dependents only if output changed
        old_cache = self.outputs["image"]._cache
        self.outputs["image"]._cache = reduced
        self._last_img_id = current_img_id
        self._last_resolution = resolution
        self._last_allow_upscale = current_allow_upscale
        try:
            changed = old_cache is None or old_cache is not reduced and getattr(old_cache, 'size', None) != getattr(reduced, 'size', None)
        except Exception:
            changed = True
        if changed:
            for dep in self.dependents:
                dep.mark_dirty()

class ContrastMaskNode(ProcessorNode):
    # args: (img: Image.Image, limMin: int, limMax: int)
    def __init__(self):
        super().__init__()
        self.category = "Mask Nodes"
        self.display_name = "Contrast Mask"
        self.progress = 0
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
            # caching: avoid regenerating mask when inputs unchanged
            if not hasattr(self, '_last_mask_inputs'):
                self._last_mask_inputs = (None, None, None)
            current_ids = (id(img), int(limMin) if limMin is not None else None, int(limMax) if limMax is not None else None)
            if self._last_mask_inputs == current_ids and self.outputs["mask"]._cache is not None:
                mask = self.outputs["mask"]._cache
            else:
                mask = passes.generate_contrast_mask(img, int(limMin), int(limMax), progress= lambda pct, msg: setattr(self, 'progress', pct))
                self._last_mask_inputs = current_ids
        except Exception as e:
            import traceback
            print(f"ContrastMaskNode: Contrast mask failed ({e}), using blank mask.")
            traceback.print_exc()
            mask = Image.new("L", img.size, 0)

        # If the generated mask contains no white pixels, fall back to a
        # permissive all-white mask so downstream operations (e.g. sorting)
        # still have something to operate on instead of being a no-op.
        try:
            extrema = mask.getextrema()
            # extrema is (min, max) for 'L' images
            if extrema is None or (isinstance(extrema, tuple) and extrema[1] == 0):
                mask = Image.new("L", img.size, 255)
        except Exception:
            # If anything goes wrong checking extrema, prefer an all-white mask
            try:
                mask = Image.new("L", img.size, 255)
            except Exception:
                pass

        # update cache and mark dependents only if mask changed
        old_mask = self.outputs["mask"]._cache
        self.outputs["mask"]._cache = mask
        try:
            changed = old_mask is None or (getattr(old_mask, 'tobytes', None) is not None and old_mask.tobytes() != mask.tobytes())
        except Exception:
            changed = True
        if changed:
            for dep in self.dependents:
                dep.mark_dirty()

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

class SplitChannelsNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Split Channels"
        self.category = "Image Effects"
        self.description = "Splits the input image into its individual color channels (R, G, B), (Luminance, Brightness), (Alpha)."
        self.tooltips_in = {
            "image": "Input image to be split into channels."
        }
        self.tooltips_out = {
            "R": "Red channel of the input image.",
            "G": "Green channel of the input image.",
            "B": "Blue channel of the input image.",
            "lum": "Luminance channel of the input image.",
            "br": "Brightness channel of the input image.",
            "aph": "Alpha channel of the input image."
        }
        
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )

        self.outputs["R"] = OutputSocket(
            self, "R", SocketType.PIL_IMG
        )
        self.outputs["G"] = OutputSocket(
            self, "G", SocketType.PIL_IMG
        )
        self.outputs["B"] = OutputSocket(
            self, "B", SocketType.PIL_IMG
        )
        self.outputs["lum"] = OutputSocket(
            self, "lum", SocketType.PIL_IMG_MONOCH
        )
        self.outputs["br"] = OutputSocket(
            self, "br", SocketType.PIL_IMG_MONOCH
        )
        self.outputs["aph"] = OutputSocket(
            self, "aph", SocketType.PIL_IMG_MONOCH
        )
        
        
    def compute(self):
        img = self.inputs["image"].get()
        if img is None:
            self.outputs["R"]._cache = None
            self.outputs["G"]._cache = None
            self.outputs["B"]._cache = None
            self.outputs["lum"]._cache = None
            self.outputs["br"]._cache = None
            self.outputs["aph"]._cache = None
            return

        try:
            print(f"SplitChannelsNode: computing channel split")
            # check if anything is connected to the advanced outputs before calling advanced split
            if (self.outputs["lum"].is_connected() or self.outputs["br"].is_connected() or self.outputs["aph"].is_connected()):
                print(f"SplitChannelsNode: advanced channel split requested")
                imgs =passes.split_channels(img, use_additional_channels=True)
                l = imgs.get("lum", Image.new("L", img.size, 0))
                br = imgs.get("br", Image.new("L", img.size, 0))
                aph = imgs.get("aph", Image.new("L", img.size, 0))
            else:
                print(f"SplitChannelsNode: basic channel split requested")
                imgs = passes.split_channels(img, use_additional_channels=False)
                l = Image.new("L", img.size, 0)
                br = Image.new("L", img.size, 0)
                aph = Image.new("L", img.size, 0)
            
            r = imgs.get("R", Image.new("L", img.size, 0))
            g = imgs.get("G", Image.new("L", img.size, 0))
            b = imgs.get("B", Image.new("L", img.size, 0))
        except Exception as e:
            import traceback
            print(f"SplitChannelsNode: Channel split failed ({e}), using blank channels.")
            traceback.print_exc()
            l = br = aph = Image.new("L", img.size, 0)
            r = g = b = Image.new("L", img.size, 0)

        self.outputs["R"]._cache = r
        self.outputs["G"]._cache = g
        self.outputs["B"]._cache = b
        self.outputs["lum"]._cache = l
        self.outputs["br"]._cache = br
        self.outputs["aph"]._cache = aph

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
        self.category = "Image Effects"
        self.description = "Computes the difference between two input images."
        self.progress = 0
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
            diff = passes.difference(imgA, imgB, progress=lambda pct, msg: setattr(self, 'progress', pct))
        except Exception as e:
            print(f"DifferenceNode: Difference computation failed ({e}), using blank image.")
            diff = Image.new("RGB", imgA.size, (0, 0, 0))

        self.outputs["image"]._cache = diff

class MixNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Mix Images"
        self.category = "Multi Image Processing"
        self.description = "Blends two input images together based on a specified mix factor."
        self.progress = 0
        self.tooltips_in = {
            "imageA": "First input image for mixing.",
            "imageB": "Second input image for mixing.",
            "mixMode": "Mode of mixing. [BLEND | multiply | add | subtract]",
            "mixMask": "Mask to control where the mix is applied.",
        }
        self.tooltips_out = {
            "image": "Output image resulting from blending the two inputs."
        }
        
        self.inputs["imageA"] = InputSocket(
            self, "imageA", SocketType.PIL_IMG
        )

        self.inputs["imageB"] = InputSocket(
            self, "imageB", SocketType.PIL_IMG
        )
        
        self.inputs["mixMode"] = InputSocket(
            self, "mixMode", SocketType.STRING
        )

        self.inputs["mixMask"] = InputSocket(
            self, "mixMask", SocketType.PIL_IMG_MONOCH
        )

        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )
        # simple input-id cache to avoid recomputing when inputs unchanged
        self._last_mix_inputs = (None, None, None, None)
    
    def compute(self):
        imgA = self.inputs["imageA"].get()
        imgB = self.inputs["imageB"].get()
        mix_mode = self.inputs["mixMode"].get()
        mix_mask = self.inputs["mixMask"].get()
        if imgA is None or imgB is None:
            self.outputs["image"]._cache = None
            return
        if mix_mask is None:
            mix_mask = Image.new("L", imgA.size, 128)  # default to 50% mix if no mask provided
        # avoid recompute if inputs haven't changed (use id() as a lightweight fingerprint)
        current_ids = (id(imgA), id(imgB), id(mix_mode), id(mix_mask))
        if current_ids == self._last_mix_inputs and self.outputs["image"]._cache is not None:
            return

        try:
            print(f"[MixNode]: computing mix of two images with mix mask")
            mixed = passes.maskMerge(
                imgA,
                imgB,
                mode=mix_mode,
                mask=mix_mask,
                progress=lambda pct, msg: setattr(self, 'progress', pct)
                )
        except Exception as e:
            import traceback
            print(f"[MixNode]: Mix computation failed ({e}), using first input image.")
            traceback.print_exc()
            mixed = imgA

        self.outputs["image"]._cache = mixed
        self._last_mix_inputs = current_ids

class NormalNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Normal Map"
        self.category = "Image Effects"
        self.description = "Generates a normal map from the input image based on specified strength and method."
        self.progress = 0
        self.tooltips_in = {
            "image": "Input image to generate normal map from.",
            "strength": "Strength of the normal map effect.",
            "method": "Method for generating the normal map (e.g., 'sobel', 'prewitt')."
        }
        self.tooltips_out = {
            "image": "Output normal map image."
        }
        
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )
        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )
    
    def compute(self):
        img = self.inputs["image"].get()
        if img is None:
            self.outputs["image"]._cache = None
            return

        self.progress = 0
        try:
            print(f"[NormalNode]: computing normal map")
            normal_map = passes.calculate_image_normals(img, progress=lambda pct, msg: setattr(self, 'progress', pct))
        except Exception as e:
            print(f"[NormalNode]: Normal map computation failed ({e}), using original image.")
            normal_map = img

        self.outputs["image"]._cache = normal_map

class DitherNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Dither Image"
        self.category = "Image Effects"
        self.description = "Applies dithering to the input image."
        self.progress = 0
        self.tooltips_in = {
            "image": "Input image to be dithered.",
            "method": "Dithering method to use [ FLOYD_STEINBERG / atkinson / bayer_monochrome ].",
            "num_colors": "Number of colors to reduce the image to.",
            "matrix_size": "Size of the Bayer matrix for dithering (must be a power of 2, e.g., 2, 4, 8). Only applicable if method is 'bayer_monochrome'.",
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

        self.inputs["matrix_size"] = InputSocket(
            self, "matrix_size", SocketType.INT
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

        self.progress = 0
        try:
            print(f"[DitherNode]: computing dithered image")
            method = self.inputs["method"].get()
            num_colors = self.inputs["num_colors"].get()
            palette_selection = self.inputs["palette_selection"].get()
            if method is None:
                method = "FLOYD_STEINBERG"
            if num_colors is None:
                num_colors = 16
            if palette_selection is None:
                palette_selection = "median_cut"
            matrix_size = self.inputs["matrix_size"].get()
            if matrix_size is None:
                matrix_size = 4

            dithered = passes.dither(
                img,
                num_colors=num_colors,
                method=method,
                palette_selection=palette_selection,
                matrix_size=matrix_size,
                progress=lambda pct, msg: setattr(self, 'progress', pct)
            )
        except Exception as e:
            import traceback
            print(f"[DitherNode]: \033[31m[ERROR]\033[0m: Dithering failed ({e}), using original image.")
            traceback.print_exc()
            dithered = img

        self.outputs["image"]._cache = dithered

class QuantizeNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Quantize Image"
        self.category = "Image Effects"
        self.description = "Reduces the number of colors in the input image to create a quantized effect."
        self.progress = 0
        self.tooltips_in = {
            "image": "Input image to be quantized.",
            "num_colors": "Number of colors to reduce the image to."
        }
        # "method": "Quantization method to use [ KMEANS / MEDIAN_CUT / OCTREE ]."
        self.tooltips_out = {
            "image": "Output quantized image."
        }
        
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )

        self.inputs["num_colors"] = InputSocket(
            self, "num_colors", SocketType.INT
        )

        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )
    
    def compute(self):
        img = self.inputs["image"].get()
        if img is None:
            self.outputs["image"]._cache = None
            return

        self.progress = 0
        try:
            print(f"[QuantizeNode]: computing quantized image")
            num_colors = self.inputs["num_colors"].get()
            if num_colors is None:
                num_colors = 8

            quantized = passes.__quantize(img, num_colors=num_colors, progress=lambda pct, msg: setattr(self, 'progress', pct))
        except Exception as e:
            import traceback
            print(f"[QuantizeNode]: Quantization failed ({e}), using original image.")
            traceback.print_exc()
            quantized = img

        self.outputs["image"]._cache = quantized

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
            limLower = 30.0   # Clear low threshold out of 255
        elif limLower <= 1.0:
            limLower = limLower * 255.0

        if limUpper is None:
            limUpper = 90.0   # Clear high threshold out of 255
        elif limUpper <= 1.0:
            limUpper = limUpper * 255.0

        try:
            print(f"[OutlineNode]: computing outline image")
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
            print(f"[OutlineNode]: Outline computation failed ({e}), using original image.")
            outlined = img

        self.outputs["image"]._cache = outlined

class SortPixelsNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Sort Pixels"
        self.category = "Image Effects"
        self.progress = 0
        self.description = "Sorts the pixels of the input image based on a specified sorting method to create various artistic effects."
        self.tooltips_in = {
            "image": "Input image whose pixels will be sorted.",
            "mode": "Sorting method to use ['LUM', 'hue', 'r', 'g', 'b'].",
            "vSplitting": "If true, splits the image into vertical sections and sorts each section independently.",
            "flipHorz": "If true, flips the image horizontally before sorting.",
            "flipVert": "If true, flips the image vertically before sorting.",
            "rotate": "Rotation to apply to the image before sorting (e.g., '0', '90', '180', '-90')."
        }
        self.tooltips_out = {
            "image": "Output image with sorted pixels."
        }
        self.silent = False 

        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )
        self.inputs["mask"] = InputSocket(
            self, "mask", SocketType.PIL_IMG_MONOCH, is_optional=True
        )
        self.inputs["mode"] = InputSocket(
            self, "mode", SocketType.STRING
        )
        self.inputs["vSplitting"] = InputSocket(
            self, "vSplitting", SocketType.BOOLEAN
        )
        self.inputs["flipHorz"] = InputSocket(
            self, "flipHorz", SocketType.BOOLEAN
        )
        self.inputs["flipVert"] = InputSocket(
            self, "flipVert", SocketType.BOOLEAN
        )
        self.inputs["rotate"] = InputSocket(
            self, "rotate", SocketType.STRING
        )

        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )


    def compute(self):
        try:
            inp_img = self.inputs["image"].get()
        except Exception as e:
            print(f"[SortPixelsNode]: Error retrieving input image ({e}), aborting compute.")
            return
        print(f"[SortPixelsNode]: starting compute with image={inp_img}")
        try: mask = self.inputs["mask"].get()
        except Exception as e: mask = None ; [print(f"[SortPixelsNode]: Error retrieving mask input ({e}), proceeding without mask.") if not self.silent else None]

        try: mode = self.inputs["mode"].get()
        except Exception as e: mode = "lum" ; [print(f"[SortPixelsNode]: Error retrieving mode input ({e}), defaulting to 'lum'.") if not self.silent else None]

        try: vSplitting = self.inputs["vSplitting"].get()
        except Exception as e: vSplitting = True ; [print(f"[SortPixelsNode]: Error retrieving vSplitting input ({e}), defaulting to True.") if not self.silent else None]

        try: flipHorz = self.inputs["flipHorz"].get()
        except Exception as e: flipHorz = False ; [print(f"[SortPixelsNode]: Error retrieving flipHorz input ({e}), defaulting to False.") if not self.silent else None]

        try: flipVert = self.inputs["flipVert"].get()
        except Exception as e: flipVert = False ; [print(f"[SortPixelsNode]: Error retrieving flipVert input ({e}), defaulting to False.") if not self.silent else None]

        try: rotate = self.inputs["rotate"].get()
        except Exception as e: rotate = "0" ; [print(f"[SortPixelsNode]: Error retrieving rotate input ({e}), defaulting to '0'.") if not self.silent else None]
        
        if not self.silent: print(f"[SortPixelsNode]: computing sorted image with mode={mode} vSplitting={vSplitting} flipHorz={flipHorz} flipVert={flipVert} rotate={rotate}")
        try: img = passes.wrap_sort(
            inp_img,
            mask=mask,
            mode=mode,
            vSplitting=vSplitting,
            flipHorz=flipHorz,
            flipVert=flipVert,
            rotate=rotate,
            progress=lambda pct, msg: setattr(self, 'progress', pct),
            silent=self.silent
        )
        except Exception as e: 
            print(f"[SortPixelsNode]: Sorting failed ({e}), using original image.")
            img = inp_img
        self.outputs["image"]._cache = img

class ParticleDrawNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Particle Draw"
        self.category = "Generative Nodes"
        self.description = "Simulates particle movement over the input image to create a dynamic drawing effect."
        self.progress = 0
        self.tooltips_in = {
            "image": "Input PIL image to guide particle movement.",
            "line count": "Maximum number of steps each particle path can take before resetting. Default is 100.",
            "draw cycles": "Total number of internal drawing step iterations per loop cycle. Default is 800.",
            "particle view radius": "Radius size within which particles look for dark target spots. Default is 5.",
            "drop_alpha": "Alpha visibility value (0-255) for dropped decorative accent spots. Default is 120.",
            "draw_alpha": "Alpha opacity value (0-255) for regular drawing stroke lines. Default is 50.",
            "stroke width": "Base stroke width thickness for structural sketch lines. Default is 1.",
            "max_speed": "Maximum allowed speed velocity cap for the tracing particles. Default is 3.0.",
            "noise_scale": "Scale grid distribution factor of the flow field noise patterns. Default is 100.0.",
            "noise_influence": "Influence weight multiplier of the noise field on direction. Default is 0.05.",
            "drop_rate": "Statistical probability chance rate at which splash accents drop. Default is 0.0044.",
            "canvas_color": "Background canvas color for the output image. Default is white (255, 255, 255, 255, 255)."
        }
        self.tooltips_out = {
            "image": "Output PIL image with particle drawing effect."
        }
        
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )
        
        self.inputs["line count"] = InputSocket(
            self, "line count", SocketType.INT
        )
        
        self.inputs["draw cycles"] = InputSocket(
            self, "draw cycles", SocketType.INT
        )
        
        self.inputs["particle view radius"] = InputSocket(
            self, "particle view radius", SocketType.INT
        )

        self.inputs["drop_alpha"] = InputSocket(
            self, "drop_alpha", SocketType.INT
        )

        self.inputs["draw_alpha"] = InputSocket(
            self, "draw_alpha", SocketType.INT
        )

        self.inputs["stroke width"] = InputSocket(
            self, "stroke width", SocketType.INT
        )

        self.inputs["max_speed"] = InputSocket(
            self, "max_speed", SocketType.FLOAT
        )

        self.inputs["noise_scale"] = InputSocket(
            self, "noise_scale", SocketType.FLOAT
        )

        self.inputs["noise_influence"] = InputSocket(
            self, "noise_influence", SocketType.FLOAT
        )

        self.inputs["drop_rate"] = InputSocket(
            self, "drop_rate", SocketType.FLOAT
        )
        
        self.inputs["canvas_color"] = InputSocket(
            self, "canvas_color", SocketType.COLOR
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
            print(f"[ParticleDrawNode]: computing particle draw effect")
            
            # Extract inputs with fallbacks to defaults from your parameters
            max_count = self.inputs["line count"].get() if self.inputs["line count"].get() is not None else 100
            sub_step = self.inputs["draw cycles"].get() if self.inputs["draw cycles"].get() is not None else 800
            perception = self.inputs["particle view radius"].get() if self.inputs["particle view radius"].get() is not None else 5
            drop_alpha = self.inputs["drop_alpha"].get() if self.inputs["drop_alpha"].get() is not None else 120
            draw_alpha = self.inputs["draw_alpha"].get() if self.inputs["draw_alpha"].get() is not None else 40
            draw_weight = self.inputs["stroke width"].get() if self.inputs["stroke width"].get() is not None else 1
            max_speed = self.inputs["max_speed"].get() if self.inputs["max_speed"].get() is not None else 3.0
            noise_scale = self.inputs["noise_scale"].get() if self.inputs["noise_scale"].get() is not None else 100.0
            noise_influence = self.inputs["noise_influence"].get() if self.inputs["noise_influence"].get() is not None else 0.05
            drop_rate = self.inputs["drop_rate"].get() if self.inputs["drop_rate"].get() is not None else 0.0044
            canvas_color = self.inputs["canvas_color"].get() if self.inputs["canvas_color"].get() is not None else (255, 255, 255, 255)

            # Assuming passes.sketch_generator works directly with PIL images or paths.
            # If your generator saves directly to file path instead of returning an image object, 
            # ensure your backend function handles the PIL input-to-output pipeline appropriately.
            particle_drawn_img = passes.sketch_generator(
                input_image=img,
                max_count=max_count,
                sub_step=sub_step,
                perception=perception,
                drop_alpha=drop_alpha,
                draw_alpha=draw_alpha,
                draw_weight=draw_weight,
                max_speed=max_speed,
                noise_scale=noise_scale,
                noise_influence=noise_influence,
                drop_rate=drop_rate,
                progress=lambda pct, msg: setattr(self, 'progress', pct)
            )
        except Exception as e:
            import traceback
            print(f"[ParticleDrawNode]: Particle drawing failed ({e}), using original image.")
            traceback.print_exc()
            particle_drawn_img = img

        self.outputs["image"]._cache = particle_drawn_img

class CircularAutomataNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Circular Automata"
        self.category = "Generative Nodes"
        self.description = "Perform circular automata generation based on the input image and parameters."
        self.progress = 0

        self.tooltips_in = {
            "image": "Input image to guide the circular automata generation.",
            "threshold": "Threshold value for the automata (0.0 to 1.0).",
            "n": "Number of neighbors to consider for the automata rules. Overrides the threshold count if specified.",
            "iterations": "Number of iterations to perform for the automata.",
            "color mode": "Color mode to use for the automata.",
        }
        self.tooltips_out = {
            "image": "Output image after circular automata processing."
        }
        
        self.inputs["image"] = InputSocket(
            self, "image", SocketType.PIL_IMG
        )
        
        self.inputs["threshold"] = InputSocket(
            self, "threshold", SocketType.FLOAT
        )
        
        self.inputs["n"] = InputSocket(
            self, "n", SocketType.INT
        )
        
        self.inputs["iterations"] = InputSocket(
            self, "iterations", SocketType.INT
        )
        
        self.inputs["color mode"] = InputSocket(
            self, "color mode", SocketType.STRING
        )
        
        self.outputs["image"] = OutputSocket(
            self, "image", SocketType.PIL_IMG
        )
        
        self._last_inputs = (None, None, None, None)  # To track changes in inputs for caching
        self._last_output = None  # To store the last computed output image
        
    def compute(self):
        img = self.inputs["image"].get()
        threshold = self.inputs["threshold"].get()
        iterations = self.inputs["iterations"].get()
        n = self.inputs["n"].get()
        color_mode = self.inputs["color mode"].get()

        if img is None:
            self.outputs["image"]._cache = None
            return

        # Setze Standardwerte, falls Eingaben im GUI leer sind
        if threshold is None:
            threshold = 0.125
        if iterations is None:
            iterations = 10
        if color_mode is None:
            color_mode = "hue"

        # Typenabsicherung für n (Sicherstellen, dass es ein Int oder None ist)
        if n is not None:
            try:
                n = int(float(n))
            except (ValueError, TypeError):
                n = None

        # KORREKTUR: 'n' MUSS in das current_inputs-Tupel für den Cache-Vergleich!
        current_inputs = (id(img), threshold, iterations, color_mode, n)
        
        # Falls sich kein Parameter geändert hat, nutze den Cache
        if current_inputs == self._last_inputs and self._last_output is not None:
            self.outputs["image"]._cache = self._last_output
            return

        try:
            print(f"[CircularAutomataNode]: computing circular automata with threshold={threshold}, n={n}, iterations={iterations}, color_mode={color_mode}")
            output_img = passes.cyclic_cellular_automata(
                img,
                threshold=threshold,
                iterations=iterations,
                value=color_mode,
                n=n,
                progress=lambda pct, msg: setattr(self, 'progress', pct)
            )
        except Exception as e:
            import traceback
            print(f"[CircularAutomataNode]: Circular automata computation failed ({e}), using original image.")
            traceback.print_exc()
            output_img = img

        self.outputs["image"]._cache = output_img
        self._last_inputs = current_inputs
        self._last_output = output_img

# Input Nodes

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

class ValuePaletteNode(InputNode):
    def __init__(self):
        super().__init__()
        self.category = "Input Nodes"
        self.display_name = "Color Palette Value"
        self.outputs["palette"] = OutputSocket(
            self, "palette", SocketType.LIST_COLORS, is_modifiable=True
        )
        self.value: list[tuple[int, int, int]] = [(255, 255, 255)]
    
    def compute(self):
        self.outputs["palette"]._cache = self.value

# Base Nodes

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
        
        self._tmp_img_paths: list[str] = []  # Store paths of temporary images added via add_img_tmp
        
        self._load_images()
    
    def _load_images(self):
        self._images.clear()
        self._image_files.clear()
        for file in sorted(os.listdir("assets/images")):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                img = Image.open(os.path.join("assets/images", file)).convert("RGB")
                self._images.append(img)
                self._image_files.append(file)
        for path in self._tmp_img_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    self._images.append(img)
                    self._image_files.append(os.path.basename(path))
                except Exception as e:
                    print(f"SourceImageNode: Failed to load temporary image from {path} ({e}).")
            else:
                print(f"SourceImageNode: Temporary image path does not exist: {path}")
    
    def add_img_tmp(self, img_path: str):
        """
        Add a temporary external image to the source node.
        """
        if img_path in self._tmp_img_paths:
            return

        if not os.path.exists(img_path):
            print(f"SourceImageNode: Temporary image path does not exist: {img_path}")
            return

        try:
            img = Image.open(img_path).convert("RGB")

            self._tmp_img_paths.append(img_path)
            self._images.append(img)
            self._image_files.append(os.path.basename(img_path))

        except Exception as e:
            print(f"SourceImageNode: Failed to load image from {img_path} ({e}).")


    def reload_images(self, image_paths: list[str]):
        """
        Replace the current temporary/external images with the provided images.
        """
        self._tmp_img_paths.clear()
        self._images.clear()
        self._image_files.clear()

        for path in image_paths:
            if not os.path.exists(path):
                print(f"SourceImageNode: Image path does not exist: {path}")
                continue

            try:
                img = Image.open(path).convert("RGB")

                self._tmp_img_paths.append(path)
                self._images.append(img)
                self._image_files.append(os.path.basename(path))

            except Exception as e:
                print(f"SourceImageNode: Failed to load image from {path} ({e}).")
        def compute(self):
            if not self._images:
                self.outputs["image"]._cache = None
                return
            
            idx = max(0, min(self.index, len(self._images) - 1))
            self.outputs["image"]._cache = self._images[idx]
    
    def compute(self):
        if not self._images:
            self.outputs["image"]._cache = None
            return
        
        idx = max(0, min(self.index, len(self._images) - 1))
        self.outputs["image"]._cache = self._images[idx]

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

            self.outputs["primary_color"] = OutputSocket(
                self,
                "primary_color", SocketType.COLOR
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
            self.outputs["primary_color"]._cache = None
            print("GetImageDataNode: No input image.")
            return
        try:
            data = passes.getImageData(img)
            self.outputs["width"]._cache = data["width"]
            self.outputs["height"]._cache = data["height"]
            self.outputs["mode"]._cache = data["mode"]
            self.outputs["format"]._cache = data["format"]
            self.outputs["info"]._cache = data["info"]
            self.outputs["primary_color"]._cache = data["primary_color"]
        except Exception as e:
            print(f"GetImageDataNode: Failed to get image data ({e}).")
            self.outputs["width"]._cache = None
            self.outputs["height"]._cache = None
            self.outputs["mode"]._cache = None
            self.outputs["format"]._cache = None
            self.outputs["info"]._cache = None
            self.outputs["primary_color"]._cache = None

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