import numpy as np
import math
from PIL import Image

def drawlnStandart(mat, x0, y0,  x1, y1, count, color):
    step = 1.0/count
    for t in np.arange (0, 1, step):
        x = round((1.0 - t)*x0 + t*x1)
        y = round((1.0 - t)*y0 + t*y1)
        mat[y, x] = color
def drawlnSmallFix(mat, x0, y0,  x1, y1, color):
    count = math.sqrt((x0 - x1)**2 + (y0 -  y1)**2)
    step = 1.0/count
    for t in np.arange (0, 1, step):
        x = round((1.0 - t)*x0 + t*x1)
        y = round((1.0 - t)*y0 + t*y1)
        mat[y, x] = color
def x_loop_drawln(mat, x0, y0,  x1, y1, color):
     for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        mat[x, y] = color
def x_loop_drawnlnf2(mat, x0, y0,  x1, y1, color):
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        if (x0 > x1):
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        mat[y, x] = color
def x_loop_drawnlnf3(mat, x0, y0,  x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        
        if (xchange):
            mat[x, y] = color
        else:
            mat[y, x] = color
def x_loop_drawnlnv1(mat, x0, y0,  x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        if (xchange):
            mat[x, y] = color
        else:
            mat[y, x] = color
def x_loop_drawnlnv2(mat, x0, y0,  x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = abs(y1 -y0)/(x1 -x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range (x0, x1):
        if (xchange):
            mat[x, y] = color
        else:
            mat[y, x] = color
        derror += dy
        if (derror > 0.5):
            derror-= 1.0
            y += y_update  
def x_loop_drawnlnv3(mat, x0, y0,  x1, y1, color):
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

def create_image(H, W, filename):
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    image_array[:H, :W] = 255
    for t in range(13):
        x0 = 100
        y0 = 100
        x1 = int(x0 + 95*math.cos(t*2*math.pi/13))
        y1 = int(y0 + 95*math.sin(t*2*math.pi/13))
        #drawlnStandart(image_array, x0, y0,  x1, y1, 1000, 0)
        #drawlnSmallFix(image_array, x0, y0,  x1, y1, 0)
        #x_loop_drawln(image_array, x0, y0,  x1, y1, 0)
        #x_loop_drawnlnf2(image_array, x0, y0,  x1, y1, 0)
        #x_loop_drawnlnf3(image_array, x0, y0, x1, y1, 0)
        #x_loop_drawnlnv1(image_array, x0, y0, x1, y1, 0)
        #x_loop_drawnlnv2(image_array, x0, y0, x1, y1, 0)
        x_loop_drawnlnv3(image_array, x0, y0, x1, y1, 0)
    image = Image.fromarray(image_array, mode="RGB")
    image.save(filename)

create_image(200, 200, 'image.png')