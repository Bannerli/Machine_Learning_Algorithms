from sklearn.datasets import fetch_openml
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
import datetime
import os
import torchvision.transforms.functional as TF
from PIL import Image


"""
x, y = fetch_openml("mnist_784", version = 1, return_X_y = True)
y = np.array([int(v) for v in y])

print(x.shape, y.shape)

x = torch.tensor(x, dtype = torch.float)
y = torch.tensor(y, dtype = torch.int64)
"""
kernel = torch.tensor([[0.03797616, 0.044863533, 0.03797616],
					   [0.044863533, 0.053, 0.04486353],
					   [0.03797616, 0.044863533, 0.03797616]])

print("sample_kernel :\t", sample_1)


# write a GaussianClass
class Operator(nn.Module):
	def __init__(self):
		super(Operator, self).__init__()

		kernel = torch.tensor([[0.03797616, 0.044863533, 0.03797616],
							   [0.044863533, 0.053, 0.04486353],
							   [0.03797616, 0.044863533, 0.03797616]])
		"""
		Gaussian_kernel = torch.FloatTensor(Gaussian_kernel).unsqueeze(0).unsqueeze(0)
		"""
		sample_1 = torch.rand(size = (3, 3, 3, 3), dtype = torch.float32)
		for i in range(sample_1.size(2)):
			sample_1[:, i] = kernel
		Gaussian_kernel = sample_1
		Vertical_Edge_detect = [[1, 1, 1],[0, 0, 0], [-1, -1, -1]]
		Horizontal_Edge_detect = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
		Edge_detectKernel_vertical = torch.FloatTensor(Vertical_Edge_detect).unsqueeze(0).unsqueeze(0)
		Edge_detectKernel_horizontal = torch.FloatTensor(Horizontal_Edge_detect).unsqueeze(0).unsqueeze(0)
		# why here we use the nn.parameter module?
		self.weight1 = nn.Parameter(data = Gaussian_kernel, requires_grad = False)
		self.weight2 = nn.Parameter(data = Edge_detectKernel_horizontal, requires_grad = False)
		self.weight3 = nn.Parameter(data = Edge_detectKernel_vertical, requires_grad = False)
	def GaussianBlur(self, x):
		self.weight = self.weight1
		"""
		x1 = x[:, 0]
		x2 = x[:, 1]
		x3 = x[:, 2]
		x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding = 1)
		x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding = 1)
		x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding = 1)
		x = torch.cat([x1, x2, x3], dim = 1)
		"""
		x = F.conv2d(x, self.weight, padding = 1).clamp(min = 0, max = 255)
		return x
	def EdgeDetect(self, image):
		x1 = image[:, 0]
		x2 = image[:, 1]
		x3 = image[:, 2]
		x1 = F.conv2d(x1.unsqueeze(1), self.weight2, padding = 1)
		x2 = F.conv2d(x2.unsqueeze(1), self.weight2, padding = 1)
		x3 = F.conv2d(x3.unsqueeze(1), self.weight2, padding = 1)
		x = torch.cat([x1, x2, x3], dim = 1)
		x = torch.clamp(x, 0, 255)
		y1 = image[:, 0]
		y2 = image[:, 1]
		y3 = image[:, 2]
		y1 = F.conv2d(y1.unsqueeze(1), self.weight3, padding = 1)
		y2 = F.conv2d(y2.unsqueeze(1), self.weight3, padding = 1)
		y3 = F.conv2d(y3.unsqueeze(1), self.weight3, padding = 1)
		y = torch.cat([y1, y2, y3], dim = 1)
		y = torch.clamp(y, 0, 255)
		return torch.sqrt(x**2 + y**2)

start_time = datetime.datetime.now()

files = os.listdir()
picture = None
for file in files:
	if file[-3:] == 'jpg':
		picture = file

image = Image.open(picture)
image = torch.unsqueeze(TF.to_tensor(image), dim = 0)
print(image.shape)
Gaussian = Operator()
processed = Gaussian.GaussianBlur(image)
print(processed.shape)
processed = TF.to_pil_image(processed[0], mode = 'RGB')
Image._show(processed)
#processed.save(os.path.join('/Users/lisizhe/Desktop/ml_homework/neural_network','data',"pikaqiu_processed.jpg"))


end_time = datetime.datetime.now()
print("total time consuming:\n", (end_time - start_time).microseconds)