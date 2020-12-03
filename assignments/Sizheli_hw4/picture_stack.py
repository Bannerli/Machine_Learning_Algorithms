import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os

files = os.listdir()
picture_list = []

for file in files:
	if file[-3:] == 'jpg':
		picture_list.append(file)

picture_list.sort()
image_0 = np.array(Image.open(picture_list[0], 'r'))
print(picture_list)
print(image_0.shape)
# N
N = len(picture_list)
C = 3
H = image_0.shape[0]
W = image_0.shape[1]

total_matrix = np.empty((N, C, H, W))
for i, file in enumerate(picture_list, 0):
	image = Image.open(file, 'r')
	image_matrix = np.asarray(image, dtype = 'float32')
	picture = np.transpose(image_matrix,(2,0,1))
	total_matrix[i,:,:,:] = picture

print(total_matrix.shape)
