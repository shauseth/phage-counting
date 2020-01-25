# SH-I

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image as Image_

from fastai.vision import *

green_dict = {'background':0.0,
              'green_five':5.0,
              'green_four':4.0,
              'green_half':0.5,
              'green_half_half':1.0,
              'green_one':1.0,
              'green_one_half':1.5,
              'green_three':3.0,
              'green_three_half':3.5,
              'green_two':2.0,
              'green_two_half':2.5}

path = Path('./data/')
learn = load_learner(path, 'resnet18.pkl')

def count(img):
    pred_class,pred_idx,outputs = learn.predict(img)
    return green_dict[str(pred_class)]

def segment(arr, n, m, p=60):

    i_y = np.arange(arr.shape[0] // p) * p
    i_x = np.arange(arr.shape[1] // p) * p

    y = i_y[n]
    x = i_x[m]

    return arr[y:y+p, x:x+p, :]

def dim(arr, p=60):

    i_y = np.arange(arr.shape[0] // p) * p
    i_x = np.arange(arr.shape[1] // p) * p

    return len(i_y), len(i_x)

file = 'test.jpg'
img = Image_.open(path/file)
arr = np.array(img)
len_n, len_m = dim(arr)

i_list = []

for n in range(len_n):

    for m in range(len_m):

        seg = segment(arr, n, m)
        img = Image(pil2tensor(seg,np.float32).div_(255))
        i = count(img)
        i_list.append(i)

count = int(np.array(i_list).sum())

heat = np.array(i_list).reshape(dim(arr))
plt.imshow(heat, cmap='Greens')
plt.axis('off')
plt.title('Green Plaques: ' + str(count))
plt.show()
