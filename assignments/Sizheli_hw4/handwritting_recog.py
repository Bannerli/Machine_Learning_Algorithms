from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
import cv2 as cv
# import MNIST
"""
x, y = fetch_openml("mnist_784", version = 1, return_X_y = True)

# reshape input/output data

y = np.array([int(v) for v in y])[np.newaxis,:]
image_1 = x[0].reshape(28,28)
#total_picture = x.reshape(x.shape[0],int(x.shape[1]**0.5),int(x.shape[1]**0.5))
image1 = Image.fromarray(image_1)
image1.show()
image_2 = x[1].reshape(28,28)
image2 = Image.fromarray(image_2)
image2.show()
"""
#plt.imshow(total_picture[0])
#plt.show()

# difine the function of convolution operation
def conv(image, weight):
	height, width = image.shape
	h, w = weight.shape
	# the new size of after the conv scanning operation
	new_h = height - h + 1
	new_w = width - w + 1
	new_image = np.zeros((new_h, new_w), dtype = np.float)
	# do the conv operation
	for i in range(new_h):
		for j in range(new_w):
			new_image[i,j] = np.sum(image[i:i+h, j:j+w] * weight)
	# remove values smaller than 0 or larger than 255
	new_image = new_image.clip(0, 255)
	new_image = np.rint(new_image).astype("uint8")
	return new_image

if __name__ == "__main__":

	# read the data of the specific picture and transfer them into the target matrix
	#A = Image.open(".jpg","r")
	a = np.array(A)
	print(a.shape)
	image_h = a.shape[0]
	image_w = a.shape[1]

	# sobel算子,分别是水平方向,垂直方向和不分方向
	sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	sobel = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
	# prewitt各个方向上的算子
	prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
	prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
	prewitt = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
	# 拉普拉斯算子
	laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
	laplacian_2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
	weight_list = ("sobel_x", "sobel_y", "sobel", "prewitt_x", "prewitt_y", "prewitt", "laplacian", "laplacian_2")
	print("R\n")
	R = conv(a[:, :, 0], prewitt_x)
	print("G\n")
	G = conv(a[:, :, 1], prewitt_x)
	print("B\n")
	B = conv(a[:, :, 2], prewitt_x)
	Image1 = np.stack((R, G, B), axis = 2)
	image_1 = Image.fromarray(Image1)

	print("R\n")
	R = conv(a[:, :, 0], prewitt_y)
	print("G\n")
	G = conv(a[:, :, 1], prewitt_y)
	print("B\n")
	B = conv(a[:, :, 2], prewitt_y)
	Image2 = np.stack((R, G, B), axis = 2)
	image_2 = Image.fromarray(Image2)

	new_image = np.rint(Image1*0.5+Image2*0.5).astype("uint8")

	print(new_image)

	image_1.show()
	image_2.show()
	#image_3.show()
	image_3 = Image.fromarray(new_image)
	image_3.show()
"""
	print("Gridient detection\n")
	for w in weight_list:
		print("starting %s...." % w)
		print("weight:\n")
		print("R\n")
		R = conv(a[:, :, 0], eval(w))
		print("G\n")
		G = conv(a[:, :, 1], eval(w))
		print("B\n")
		B = conv(a[:, :, 2], eval(w))

		I = np.stack((R, G, B), axis = 2)
		Image.fromarray(I).save("%s.jpg" % (w))
"""

