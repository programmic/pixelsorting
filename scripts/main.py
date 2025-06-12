from PIL import Image
import threading
import os
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
contrastLimUpper = 100 
sortMode = "lum"   
inverse = False
###########################

image = loadImage()

contrastMask = prepasses.contrastMask(image, contrastLimLower, contrastLimUpper)
contrastMask.show()

chunks = prepasses.getCoherentImageChunks(contrastMask)

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
sorted = results["sorted"]

chunksVis.show()
sorted.show()