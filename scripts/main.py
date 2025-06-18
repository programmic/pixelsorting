from PIL import Image
import os
from datetime import datetime
from tqdm import tqdm
import passes as passes
import converters as converters


def load_image_from_folder(folder: str = "assets/images/") -> Image.Image:
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder, filename)
            print(f"Lade Bild: {image_path}")
            return Image.open(image_path).convert("RGB")
    raise FileNotFoundError("Kein gültiges Bild im Ordner gefunden.")


def process_image(
    image: Image.Image,
    contrastLimLower: int = 20,
    contrastLimUpper: int = 200,
    sortMode: str = "lum",
    inverse: bool = True,
    useVerticalSplitting: bool = True,
    rotateImage: bool = True,
    save_output: bool = True,
    show_output: bool = True,
    show_masks: bool = False,
    exportPath = f"C:/Users/{os.getlogin()}/Downloads"
    ) -> dict:
    
    contrastMask = passes.contrastMask(image, contrastLimLower, contrastLimUpper)

    chunks = passes.getCoherentImageChunks(contrastMask, rotateImage)

    if useVerticalSplitting:
        chunks = passes.toVerticalChunks(chunks)
        chunks = passes.splitConnectedChunks(chunks)

    visualizeImageBase = image.rotate(90, expand=True) if rotateImage else image
    results = {}

    if show_masks:
        results["chunksVis"] = passes.visualizeChunks(
            visualizeImageBase,
            chunks,
            rotateImage
        )
        if show_output:
            results["chunksVis"].show()
    else: results["chunksVis"] = None

    results["sorted"] = passes.sort(
        image,
        chunks,
        sortMode,
        inverse,
        rotateImage
    )

    if show_output:
        results["sorted"].show()

    if save_output:
        now = datetime.now().strftime("%H-%M-%S")
        os.makedirs(exportPath, exist_ok=True)
        results["sorted"].save(f"{exportPath}/{now}.png")

    return results


if __name__ == "__main__":
    image = load_image_from_folder()





    print("image loaded, running filter")
    kwh = passes.kuwaharaGPU(image, 8)
    kwh.show()
    p = f"C:/Users/{os.getlogin()}/Downloads"

    now = datetime.now().strftime("%H-%M-%S")
    os.makedirs(p, exist_ok=True)
    kwh.save(f"{p}/{now}.png")

    



    #imgA = passes.blurGaussian1d(image, 1, num_processes=32)
    #imgB = passes.blurGaussian1d(image, 4, num_processes=32)
    #
    #delta = passes.subtractImages(imgA, imgB)
    #delta = passes.adjustBrightness(delta, 2)
    #delta.show()
#
    #mask = passes.contrastMask(delta, 8, 500)
    #mask.show()
