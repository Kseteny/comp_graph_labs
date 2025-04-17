import numpy as np
import math
import random
from PIL import Image, ImageOps

WIDTH = 1000
HEIGHT = 1000
Z_BUFFER = np.zeros((WIDTH, HEIGHT), dtype=np.float32)
Z_BUFFER[...] = np.inf

def f0_fromFile(path:str):
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] == 'f' and line[1] == ' ':
                line = line[2:-1]
                List.append(list(map(int, [line.split(' ')[0].split('/')[0], line.split(' ')[1].split('/')[0], line.split(' ')[2].split('/')[0]])))
    return List

def f1_fromFile(path:str):
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] == 'f' and line[1] == ' ':
                line = line[2:-1]
                List.append(list(map(int, [line.split(' ')[0].split('/')[1], line.split(' ')[1].split('/')[1], line.split(' ')[2].split('/')[1]])))
    return List

def v_fromFile(path:str):
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] == 'v' and line[1] == ' ':
                line = line[2:-1]
                List.append(list(map(float, line.split(' '))))
    return List
#---------------------------------------
def vt_fromFile(path:str):
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] == 'v' and line[1] == 't':
                line = line[3:-1]
                List.append(list(map(float, line.split(' '))))
    return List
#---------------------------------------
def vn_fromFile(path:str):
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] == 'v' and line[1] == 'n':
                line = line[3:-1]
                List.append(list(map(float, line.split(' '))))
    return List

def bar_coord(x, y, x0, y0, x1, y1, x2, y2)->tuple:
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)) 
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /  ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1 
    return (lambda0, lambda1, lambda2)
def draw_triangle(mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, color, texture_coords, image_text):
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
                    U = round(HEIGHT*(coords[0]*texture_coords[0] + coords[1]*texture_coords[2] + coords[2]*texture_coords[4]))
                    V = round(WIDTH*(coords[0]*texture_coords[1] + coords[1]*texture_coords[3] + coords[2]*texture_coords[5]))
                    mat[y, x] = image_text[V][U]

def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n = np.cross(np.array([x1 - x2, y1 - y2, z1 - z2]), np.array([x1 - x0, y1 - y0, z1 - z0]))
    return n/np.linalg.norm(n)
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

def create_image(H, W, filename, v, f0, f1, vn, vt, texture_image):
    print(len(v))
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    image_array[:H, :W] = 255
    
    ALPHA = 160
    BETA = -0.5
    GAMMA = 0
    rotate_by_alplha_beta_gamma(v, ALPHA, BETA, GAMMA)
    vn_calc = np.zeros((len(v), 3), dtype=np.float32)
    for fi in range(len(f0)):
        x0 = v[f0[fi][0] - 1][0]
        y0 = v[f0[fi][0] - 1][1]
        z0 = v[f0[fi][0] - 1][2]
        x1 = v[f0[fi][1] - 1][0]
        y1 = v[f0[fi][1] - 1][1]
        z1 = v[f0[fi][1] - 1][2]
        x2 = v[f0[fi][2] - 1][0]
        y2 = v[f0[fi][2] - 1][1]
        z2 = v[f0[fi][2] - 1][2]
        vn_calc[f0[fi][0] - 1] += normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        vn_calc[f0[fi][1] - 1] += normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        vn_calc[f0[fi][2] - 1] += normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    
    Ii = list()
    for vni in range(len(vn_calc)):
        vn_calc[vni] = vn_calc[vni]/np.linalg.norm(vn_calc[vni])
    for vni in range(len(vn_calc)):
        Ii.append(cosa(vn_calc[vni]))
    for fi in range(len(f0)):
        x0 = v[f0[fi][0] - 1][0]
        y0 = v[f0[fi][0] - 1][1]
        z0 = v[f0[fi][0] - 1][2]
        u0 = vt[f1[fi][0] - 1][0]
        v0 = vt[f1[fi][0] - 1][1]
        x1 = v[f0[fi][1] - 1][0]
        y1 = v[f0[fi][1] - 1][1]
        z1 = v[f0[fi][1] - 1][2]
        u1 = vt[f1[fi][1] - 1][0]
        v1 = vt[f1[fi][1] - 1][1]
        x2 = v[f0[fi][2] - 1][0]
        y2 = v[f0[fi][2] - 1][1]
        z2 = v[f0[fi][2] - 1][2]
        u2 = vt[f1[fi][2] - 1][0]
        v2 = vt[f1[fi][2] - 1][1]
        cosA = Ii[f0[fi][0] - 1]
        cosB = Ii[f0[fi][1] - 1]
        cosC = Ii[f0[fi][2] - 1]
        draw_triangle(image_array, x0, y0, z0, x1, y1, z1, x2, y2, z2, (cosA, cosB, cosC), (u0, v0, u1, v1, u2, v2), texture_image)

    image = Image.fromarray(image_array, mode="RGB")
    image.save(filename)


f0 = f0_fromFile("model_1.obj")
v = v_fromFile("model_1.obj")
#---------------------------------------
vt = vt_fromFile("model_1.obj")
vn = vn_fromFile("model_1.obj")
f1 = f1_fromFile("model_1.obj")
texture_image = np.array(ImageOps.flip(Image.open("bunny-atlas.jpg")))
#---------------------------------------
create_image(WIDTH, HEIGHT, 'image_z_buffer.png', v, f0, f1, vn, vt, texture_image)