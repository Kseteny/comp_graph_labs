import numpy as np
import math
import random
from PIL import Image
def f_fromFile(path:str):
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] == 'f' and line[1] == ' ':
                line = line[2:-1]
                List.append(list(map(int, [line.split(' ')[0].split('/')[0], line.split(' ')[1].split('/')[0], line.split(' ')[2].split('/')[0]])))
    return List
def v_fromFile(path:str):
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] == 'v' and line[1] == ' ':
                line = line[2:-1]
                List.append(list(map(float, line.split(' '))))
    return List
def bar_coord(x, y, x0, y0, x1, y1, x2, y2)->tuple:
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)) 
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /  ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1 
    return (lambda0, lambda1, lambda2)
def draw_triangle(mat, x0, y0, x1, y1, x2, y2, color):
    xmin = math.floor(min(x0, x1, x2))
    ymin = math.floor(min(y0, y1, y2))
    xmax = math.ceil(max(x0, x1, x2))
    ymax = math.ceil(max(y0, y1, y2))
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if(xmax > 1000): xmax = 1000
    if(ymax > 1000): ymax = 1000
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            coords = bar_coord(x, y, x0, y0, x1, y1, x2, y2)
            if(coords[0] >=0 and coords[1] >= 0 and coords[2] >= 0):
                mat[y, x] = color

def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n = np.cross(np.array([x1 - x2, y1 - y2, z1 - z2]), np.array([x1 - x0, y1 - y0, z1 - z0]))
    return n
def cosa(n:np.array):
    l = np.array([0,0,1])
    cosa = np.dot(n, l)/ (math.sqrt(n[0]**2 + n[1]**2 + n[2]**2))
    return cosa
def create_image(H, W, filename, v, f):
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    image_array[:H, :W] = 255
    scaled_time = 4000
    for fi in range(len(f)):
        x0 = v[f[fi][0] - 1][0]*scaled_time + 500
        y0 = v[f[fi][0] - 1][1]*scaled_time + 250
        z0 = v[f[fi][0] - 1][2]*scaled_time
        x1 = v[f[fi][1] - 1][0]*scaled_time + 500
        y1 = v[f[fi][1] - 1][1]*scaled_time + 250
        z1 = v[f[fi][1] - 1][2]*scaled_time
        x2 = v[f[fi][2] - 1][0]*scaled_time + 500
        y2 = v[f[fi][2] - 1][1]*scaled_time + 250
        z2 = v[f[fi][2] - 1][2]*scaled_time
        cosA = cosa(normal(x0, y0, z0, x1, y1, z1, x2, y2, z2))
        if(cosA < 0):
            draw_triangle(image_array, x0, y0, x1, y1, x2, y2, -cosA*255)
    image = Image.fromarray(image_array, mode="RGB")
    image.save(filename)


f = f_fromFile("model_1.obj")
v = v_fromFile("model_1.obj")
create_image(1000, 1000, 'image.png', v, f)