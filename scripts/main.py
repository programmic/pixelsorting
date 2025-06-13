# main.py

from PIL import Image
import threading
import os
from datetime import datetime
from tqdm import tqdm
import passes as passes
import converters as converters

def loadImage() -> Image.Image:
    image_folder = "assets/images/"
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            print(f"Lade Bild: {image_path}")
            return Image.open(image_path).convert("RGB")
    raise FileNotFoundError("Kein gültiges Bild im Ordner gefunden.")

###########################
contrastLimLower = 20 
contrastLimUpper = 200 
sortMode = "lum"   
inverse = True
useVerticalSplitting = True
rotateImage = True
###########################

image = loadImage()

contrastMask = passes.contrastMask(image, contrastLimLower, contrastLimUpper)
contrastMask.show()

chunks = passes.getCoherentImageChunks(contrastMask, rotateImage)

if useVerticalSplitting:
    chunks = passes.toVerticalChunks(chunks)
    chunks = passes.splitConnectedChunks(chunks)

visualizeImageBase = image.rotate(90, expand=True) if rotateImage else image
results = {}
threads = [
    threading.Thread(
        target=lambda: results.update({"chunksVis": passes.visualizeChunks(visualizeImageBase, chunks, rotateImage)})
    ),
    threading.Thread(
        target=lambda: results.update({"sorted": passes.sort(image, chunks, sortMode, inverse, rotateImage)})
    )
]

for t in threads:
    t.start()
for t in threads:
    t.join()

chunksVis = results["chunksVis"]
sorted: Image.Image = results["sorted"]

chunksVis.show()
sorted.show()

now = datetime.now().strftime("%H-%M-%S")  # z.B. "14-23-07"
sorted.save(f"assets/printouts/{now}.png")