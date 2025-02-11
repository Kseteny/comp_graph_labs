import numpy as np
import math
from PIL import Image

def create_image(H, W, filename):
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    image_array[:H, :W, 1] = 255
    for x in range(H):
        for y in range(W):
            if (y == x or y == 200-x):
                image_array[x, y] = 0
    image = Image.fromarray(image_array, mode="RGB")
    image.save(filename)

create_image(600, 800, 'image.png')