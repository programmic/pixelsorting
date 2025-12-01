from PIL import Image
import numpy as np
im = Image.open('saved/debug_sort.png')
print('mode', im.mode, 'size', im.size)
arr = np.array(im)
# print average color of first 16 columns
W = im.width
for x in range(0, W, max(1, W//16)):
    col = arr[:, x, :3]
    mean = col.mean(axis=0).astype(int)
    uniq = len(np.unique(col.reshape(-1, 3), axis=0))
    print(f'col {x}: mean={tuple(mean)} uniques={uniq}')
# show a small sample of first row pixels
row = arr[0, :20, :3]
print('first row sample:', [tuple(px) for px in row])
print('\n--- HORIZONTAL variant (vSplitting=False) ---')
im2 = Image.open('saved/debug_sort_horiz.png')
print('mode', im2.mode, 'size', im2.size)
arr2 = np.array(im2)
W2 = im2.width
for x in range(0, W2, max(1, W2//16)):
    col = arr2[:, x, :3]
    mean = col.mean(axis=0).astype(int)
    uniq = len(np.unique(col.reshape(-1, 3), axis=0))
    print(f'col {x}: mean={tuple(mean)} uniques={uniq}')
row2 = arr2[0, :20, :3]
print('first row sample:', [tuple(px) for px in row2])
