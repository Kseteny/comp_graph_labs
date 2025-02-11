import numpy as np
import math
from PIL import Image

def draw(mat, x0, y0,  x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2*abs(y1 -y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range (x0, x1):
        if (xchange):
            mat[x, y] = color
        else:
            mat[y, x] = color
        derror += dy
        if derror > 2*(x1 - x0)*0.5:
            derror -= 2.0*(x1 - x0)*1.0
            y += y_update  
def v_fromFile(path:str):
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] == 'v' and line[1] == ' ':
                line = line[2:-1]
                List.append(list(map(float, line.split(' '))))
    return List
def f_fromFile(path:str):
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] == 'f' and line[1] == ' ':
                line = line[2:-1]
                List.append(list(map(int, [line.split(' ')[0].split('/')[0], line.split(' ')[1].split('/')[0], line.split(' ')[2].split('/')[0]])))
    return List
def create_image(H, W, filename, v, f):
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    image_array[:H, :W] = 255
    for fi in range(len(f)):
        x0 = int(v[f[fi][0] - 1][0]*5000 + 500)
        y0 = int(v[f[fi][0] - 1][1]*5000 + 250)
        x1 = int(v[f[fi][1] - 1][0]*5000 + 500)
        y1 = int(v[f[fi][1] - 1][1]*5000 + 250)

        x2 = int(v[f[fi][2] - 1][0]*5000 + 500)
        y2 = int(v[f[fi][2] - 1][1]*5000 + 250)


        draw(image_array, x0, y0, x1, y1, 0)
        draw(image_array, x0, y0, x2, y2, 0)
        draw(image_array, x1, y1, x2, y2, 0)
    image = Image.fromarray(image_array, mode="RGB")
    image.save(filename)

f = f_fromFile("model_1.obj")
v = v_fromFile("model_1.obj")
create_image(1000, 1000, 'image.png', v, f)