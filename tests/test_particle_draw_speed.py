try:
    from scripts.passes import sketch_generator
except ImportError:
    print("Failed to import sketch_generator.")
    print("Use command \033[33mpython -m tests.test_particle_draw_speed\033[0m to run this test script.")
    exit(0)
    
# tests/test_particle_draw_speed.py
"""Manually tests for scripts/passes.py - specifically for the particle draw speed functionality."""
import os
from PIL import Image
import time
from functools import wraps

# compare sketch_generator() with and without deprecated mode to check wether the applied changes to the code are significant or actually harmful to the output image quality.

# load five different test images
print("Loading test images...")
test_images = []
for i in range(1, 6):
    img_path = f"tests/data/test_image_{i}.jpg"
    if os.path.exists(img_path):
        test_images.append(Image.open(img_path))
    else:
        print(f"Image {i} not found.")

def get_current_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

print("All test images loaded.")

try:
    from scripts.passes import sketch_generator
except ImportError:
    print("Failed to import sketch_generator.")
    raise ImportError("Could not import sketch_generator from scripts.passes. Ensure the module is available.")

results = []
times = {}
print("Running sketch_generator on test images...")
for idx, img in enumerate(test_images):
    img_label = f"Image {idx+1}"
    print(f"\nProcessing {img_label}...")
    
    # --- DEPRECATED MODE ---
    print(f"Generating sketch with deprecated mode...\nStart: {get_current_time_str()}")
    t0 = time.perf_counter() # Präziser als time.time() für Messungen
    sketch_deprecated = sketch_generator(img, use_deprecated=True)
    t1 = time.perf_counter()
    
    duration_dep = t1 - t0
    times[f"{img_label} - Deprecated"] = duration_dep
    print(f"Finished. End: {get_current_time_str()}")
    print(f"Execution time for deprecated mode: {duration_dep:.4f} seconds\n")
    
    # --- NEW MODE ---
    print(f"Generating sketch with new mode...\nStart: {get_current_time_str()}")
    t2 = time.perf_counter()
    sketch_new = sketch_generator(img, use_deprecated=False)
    t3 = time.perf_counter()
    
    duration_new = t3 - t2
    times[f"{img_label} - New"] = duration_new
    print(f"Finished. End: {get_current_time_str()}")
    print(f"Execution time for new mode: {duration_new:.4f} seconds\n")
    
    results.append((sketch_deprecated, sketch_new))


# tile the results for visual comparison
tiled_results = []
for sketch_deprecated, sketch_new in results:
    print("Tiling results for visual comparison...")
    # Create a new image to tile the two sketches side by side
    width = sketch_deprecated.width + sketch_new.width
    height = max(sketch_deprecated.height, sketch_new.height)
    tiled_image = Image.new('RGB', (width, height))
    
    # Paste the deprecated sketch on the left
    tiled_image.paste(sketch_deprecated, (0, 0))
    
    # Paste the new sketch on the right
    tiled_image.paste(sketch_new, (sketch_deprecated.width, 0))
    
    tiled_results.append(tiled_image)


# Save the tiled results for visual inspection
print("Saving tiled results for visual inspection...")
for idx, tiled_image in enumerate(tiled_results):
    tiled_image.save(f"tests/data/tiled_result_{idx+1}.jpg")

# Print the timing results for comparison
print("Timing results for sketch_generator:")
for key, value in times.items():
    print(f"{key}: {value:.4f} seconds")