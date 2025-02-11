import numpy as np
import math
from PIL import Image

def v_fromFile(path:str):
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] == 'v' and line[1] == ' ':
                line = line[2:-1]
                List.append(list(map(float, line.split(' '))))
    return List
def create_image(H, W, filename, List):
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    image_array[:H, :W] = 255
    for ls in List:
        image_array[int(ls[0] * 5000 + 500) , int(ls[1] * 5000 + 250)] = 0
        image_array[int(ls[0] * 5000 + 500) , int(ls[1] * 5000 + 250)] = 0
    image = Image.fromarray(image_array, mode="RGB")
    image.save(filename)

List = v_fromFile("model_1.obj")
create_image(1000, 1000, 'image.png', List)