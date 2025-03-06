import numpy as np
import math
import random
from PIL import Image

WIDTH = 1000
HEIGHT = 1000
Z_BUFFER = np.zeros((WIDTH, HEIGHT), dtype=np.float32)
Z_BUFFER[...] = np.inf

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
def draw_triangle(mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, color):
    scaled_time = 584
    sdvig = 500
    x0, y0 = scaled_time*x0/z0 + sdvig, scaled_time*y0/z0 + sdvig
    x1, y1 = scaled_time*x1/z1 + sdvig, scaled_time*y1/z1 + sdvig
    x2, y2 = scaled_time*x2/z2 + sdvig, scaled_time*y2/z2 + sdvig
    xmin = math.floor(min(x0, x1, x2))
    ymin = math.floor(min(y0, y1, y2))
    xmax = math.ceil(max(x0, x1, x2))
    ymax = math.ceil(max(y0, y1, y2))
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if(xmax > 1000): xmax = WIDTH
    if(ymax > 1000): ymax = HEIGHT
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            coords = bar_coord(x, y, x0, y0, x1, y1, x2, y2)
            if(coords[0] >=0 and coords[1] >= 0 and coords[2] >= 0):
                z_= coords[0]*z0 + coords[1]*z1 + coords[2]*z2
                if(z_ < Z_BUFFER[y, x]):
                    Z_BUFFER[y, x] = z_
                    mat[y, x] = color

def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n = np.cross(np.array([x1 - x2, y1 - y2, z1 - z2]), np.array([x1 - x0, y1 - y0, z1 - z0]))
    return n
def cosa(n:np.array):
    l = np.array([0,0,1])
    cosa = np.dot(n, l)/ (math.sqrt(n[0]**2 + n[1]**2 + n[2]**2))
    return cosa

def rotate_by_alplha_beta_gamma(v, ALPHA, BETA, GAMMA):
    R = np.dot(np.array([[1, 0, 0], [0, math.cos(ALPHA), math.sin(ALPHA)], [0, -math.sin(ALPHA), math.cos(ALPHA)]]), np.array([[math.cos(BETA), 0, math.sin(BETA)], [0,1,0], [-math.sin(BETA), 0, math.cos(BETA)]]))
    R = np.dot(R, np.array([[math.cos(GAMMA), math.sin(GAMMA), 0], [-math.sin(GAMMA), math.cos(GAMMA), 0], [0, 0, 1]]))
    tx = 0
    ty = 0.03
    tz = 0.1
    for i in range(len(v)):
        v[i] = np.dot(R, np.array(v[i])) + np.array([tx, ty, tz])
    
def create_image(H, W, filename, v, f):
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    image_array[:H, :W] = 255
    
    ALPHA = 160
    BETA = -0.5
    GAMMA = 0
    rotate_by_alplha_beta_gamma(v, ALPHA, BETA, GAMMA)
    for fi in range(len(f)):
        x0 = v[f[fi][0] - 1][0]
        y0 = v[f[fi][0] - 1][1]
        z0 = v[f[fi][0] - 1][2]
        x1 = v[f[fi][1] - 1][0]
        y1 = v[f[fi][1] - 1][1]
        z1 = v[f[fi][1] - 1][2]
        x2 = v[f[fi][2] - 1][0]
        y2 = v[f[fi][2] - 1][1]
        z2 = v[f[fi][2] - 1][2]
        cosA = cosa(normal(x0, y0, z0, x1, y1, z1, x2, y2, z2))
        if(cosA < 0):
            draw_triangle(image_array, x0, y0, z0, x1, y1, z1, x2, y2, z2, -cosA*255)
    image = Image.fromarray(image_array, mode="RGB")
    image.save(filename)


f = f_fromFile("model_1.obj")
v = v_fromFile("model_1.obj")
create_image(WIDTH, HEIGHT, 'image_z_buffer.png', v, f)