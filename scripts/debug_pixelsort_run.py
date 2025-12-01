from PIL import Image
from passes import wrap_sort

# create a horizontal gradient image (left red -> right blue)
W, H = 512, 384
img = Image.new('RGB', (W, H))
for x in range(W):
    for y in range(H):
        r = int(255 * (1 - x / (W - 1)))
        b = int(255 * (x / (W - 1)))
        g = int(128 * (y / (H - 1)))
        img.putpixel((x, y), (r, g, b))

# run wrap_sort with vSplitting True, no flips, no rotation, mode lum
out = wrap_sort(img, mode='lum', vSplitting=True, flipHorz=False, flipVert=False, rotate='0')

out.save('saved/debug_sort.png')
print('Wrote saved/debug_sort.png')
out2 = wrap_sort(img, mode='lum', vSplitting=False, flipHorz=False, flipVert=False, rotate='0')
out2.save('saved/debug_sort_horiz.png')
print('Wrote saved/debug_sort_horiz.png')
