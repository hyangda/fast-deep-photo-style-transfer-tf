#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:36:19 2018

@author: hyang
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# %% Function definitions
def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files


def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

# %% Test

path = '../fast_style_transfer/TEST_seg/'

image_paths = _get_files(path)

black = np.zeros(len(image_paths))
white = np.zeros(len(image_paths))

for j in range(len(image_paths)):
    print(j)
    print(image_paths[j])
    im = Image.open(image_paths[j])
    array = np.asarray(im)
    white[j] = np.sum(array > 0) / array.size
    black[j] = 1 - white[j]
    
# %% Plot
plt.figure
plt.title('Background/foreground area fraction')
plt.scatter(range(len(image_paths)), white, c='red')
plt.scatter(range(len(image_paths)), black, c='black')
plt.savefig('bg_fg.png')
plt.show()
