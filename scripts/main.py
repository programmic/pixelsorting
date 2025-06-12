from PIL import Image
import threading
import os
from datetime import datetime
from tqdm import tqdm
import prepasses

def loadImage() -> Image.Image:
    image_folder = "assets/images/"
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            print(f"Lade Bild: {image_path}")
            return Image.open(image_path).convert("RGB")
    raise FileNotFoundError("Kein gültiges Bild im Ordner gefunden.")

###########################
contrastLimLower = -5  
contrastLimUpper = 180 
sortMode = "lum"   
inverse = False
useVerticalSplitting = False
###########################

image = loadImage()

contrastMask = prepasses.contrastMask(image, contrastLimLower, contrastLimUpper)
contrastMask.show()

chunks = prepasses.getCoherentImageChunks(contrastMask)
if useVerticalSplitting:
    chunks = prepasses.toVerticalChunks(chunks)
    chunks = prepasses.splitConnectedChunks(chunks)


results = {}
threads = [
    threading.Thread(
        target=lambda: results.update({"chunksVis": prepasses.visualizeChunks(image, chunks)})
    ),
    threading.Thread(
        target=lambda: results.update({"sorted": prepasses.sort(image, chunks, sortMode, inverse)})
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