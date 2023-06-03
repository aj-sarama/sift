import sys
from PIL import Image, ImageOps, ImageChops
import numpy as np
import math
from time import sleep
import random
import matplotlib.pyplot as plt

def main():
    image_path = sys.argv[1]
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    matrix = np.array(image)

    # at what pixel do we want the descriptor?
    x = 425
    y = 550
    # and what is our sigma value?
    sigma = 64
    # draw circle
    angle_window_size = 24 # 16 from center in each direction
    # it makes sense that the angle window size would be smaller

    window = matrix[y - angle_window_size:y + angle_window_size, x - angle_window_size:x + angle_window_size].astype(float)
    large_window = np.array(Image.fromarray(window).resize((2*angle_window_size*8, 2*angle_window_size*8)))
    # calculate gradients inside the window
    gradient_y = np.zeros_like(window)
    gradient_x = np.zeros_like(window)
    for x in range(1, 2*angle_window_size - 1):
        for y in range(1, 2*angle_window_size - 1):
            gradient_y[y][x] = window[y-1][x] - window[y+1][x]
            gradient_x[y][x] = window[y][x+1] - window[y][x-1]

    # get the angle at each pixel
    angles = np.zeros((2*angle_window_size, 2*angle_window_size)).astype(float)
    for x in range(0, 2*angle_window_size):
        for y in range(0, 2*angle_window_size):
            if gradient_x[y][x] == 0:
                if gradient_y[y][x] == 0:
                    angles[y][x] = 0.0
                    continue
                elif gradient_y[y][x] > 0:
                    angles[y][x] = math.pi / 2
                    continue
                else:
                    angles[y][x] = -math.pi / 2
                    continue
            elif gradient_y[y][x] == 0:
                if gradient_x[y][x] == 0:
                    angles[y][x] = 0.0
                    continue
                elif gradient_x[y][x] > 0:
                    angles[y][x] = 0.0
                    continue
                else:
                    angles[y][x] = math.pi
                    continue

            angles[y][x] = math.atan(abs(gradient_y[y][x]) / abs(gradient_x[y][x]))
            if gradient_x[y][x] < 0 and gradient_y[y][x] > 0:
                angles[y][x] = math.pi - angles[y][x]
            elif gradient_x[y][x] < 0 and gradient_y[y][x] < 0:
                angles[y][x] += math.pi
            elif gradient_x[y][x] > 0 and gradient_y[y][x] < 0:
                angles[y][x] = -angles[y][x]

    bins = np.zeros((36)).astype(float)
    
    for x in range(1, 2*angle_window_size - 1):
        for y in range(1, 2*angle_window_size - 1):
            angle = angles[y][x]
            angle = angle * 180 / math.pi
            if angle < 0:
                angle += 360
            idx = int(angle / 10)
            bins[idx] += 1.0
            large_window[8*y][8*x] = 0
            for i in range(1,6):
                offset_x = math.cos(angles[y][x])*i
                offset_y = -math.sin(angles[y][x])*i
                coord_x = int(round(8*x + offset_x))
                coord_y = int(round(8*y + offset_y))
                if coord_x < 2*angle_window_size*8 and coord_y < 2*angle_window_size*8 and coord_x >= 0 and coord_y >= 0:
                    large_window[coord_y][coord_x] = 255

    # 6 times circular convolution
    for i in range(0,6):
        for x in range(0,36):
            if x == 0:
                bins[x] = (1/3) * (bins[35] + bins[0] + bins[1])
            elif x == 35:
                bins[x] = (1/3) * (bins[34] + bins[35] + bins[1])
            else:
                bins[x] = (1/3) * (bins[x - 1] + bins[x] + bins[x + 1])


    plt.bar(np.arange(36),bins)
    plt.show()


    def debug_points():
        x = random.randint(1,2*angle_window_size - 1)
        y = random.randint(1,2*angle_window_size - 1)
        window[y][x] = 0
        print("center: ", x, y)
        print("above: ", window[y-1][x])
        print("below: ", window[y+1][x])
        print("gradient_y", gradient_y[y][x])
        print("left: ", window[y][x-1])
        print("right: ", window[y][x+1])
        print("gradient_x", gradient_x[y][x])
        print("angle", angles[y][x]*180 / math.pi)
        for i in range(1,10):
            offset_x = math.cos(angles[y][x])*i
            offset_y = -math.sin(angles[y][x])*i
            window[int(round(y + offset_y))][int(round(x + offset_x))] = 255

    window_image = Image.fromarray(large_window)

    window_image.show()
    #Image.fromarray(gradient_y).show()
    #Image.fromarray(gradient_x).show()

if __name__ == "__main__":
    main()

