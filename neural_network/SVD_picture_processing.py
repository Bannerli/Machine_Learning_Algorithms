import numpy as np
import PIL.Image as Image
import os
import matplotlib.pyplot as plt


files = os.listdir()
picture = None
for file in files:
	if file[-3:] == 'jpg':
		picture = file

image = Image.open(picture)

a = np.array(image)
print(a.dtype)
# print the shape of the image matrix
print("the shape of matrix:\n", a.shape)


# use the svd to decompose the picture
u, sigma, v = np.linalg.svd(a[:,:,0])
print(u.shape)
print(v.shape)

def rebuild_img(u, sigma, v, p):
	m = len(u)
	n = len(v)
	a = np.zeros((m,n))

	count = int(sum(sigma))
	curSum = 0
	k = 0

	while curSum <= count * p:
		uk = u[:, k].reshape(m,1)
		vk = v[k].reshape(1,n)
		# 每一个uk，vk相乘都是一个m*n的矩阵,所以是cur_Sum个m*n矩阵的相加
		a += sigma[k] * np.dot(uk, vk)
		curSum += sigma[k]
		k += 1

	a.clip(0,255)
	return np.rint(a).astype("uint8")

R = rebuild_img(u, sigma, v, 1)
print(R)

for i in np.arange(0.1, 1, 0.1):
	u, sigma, v = np.linalg.svd(a[:, :, 0])
	R = rebuild_img(u, sigma, v, i)

	u, sigma, v = np.linalg.svd(a[:, :, 1])
	G = rebuild_img(u, sigma, v, i)

	u, sigma, v = np.linalg.svd(a[:, :, 2])
	B = rebuild_img(u, sigma, v, i)

	I = np.stack((R, G, B), 2)
	plt.subplot(330 + int(i * 10))
	plt.title(i)
	plt.imshow(I)

plt.show()





