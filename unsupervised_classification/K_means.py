import numpy as np
import matplotlib.pylab as plt
# 这个类用于设置时间间隔
from matplotlib.pyplot import  MultipleLocator
import numpy as np

np.random.seed(46)

def randomPoint(dimension, num):
	# 随机生成矩阵，
	matrix = np.random.random((num, dimension))
	return matrix
def trainProduct(dimension, num):
	# 随机生成矩阵，矩阵每个元素扩大10倍
	matrix = np.random.rand(num,dimension)*10
	return matrix

def k_means(data, randompoint,iterations = 10):
	# 设置迭代次数为10
	ck = randompoint
	for j in range(iterations):
		distance = []
		for i in range(randompoint.shape[0]):
			vector = np.sum((data - ck[i]) ** 2, axis = 1)
			distance.append(vector)
			# 这时distance里面存储了各个点到三个随机点的欧氏距离
		result = np.asarray(distance).T
		label = np.argmin(result, axis = 1)
		# 此时一轮迭代已经结束， 对重新分布的点进行label标注和更新重心
		for k in range(ck.shape[0]):
			lab = np.array(label == k)
			ck[k] = np.mean(data[lab], axis = 0)
	return label,ck

def plotting(label,data,center_points):
	for i,c in zip(range(np.max(label)+1),["b","r","g"]):
		x1 = data[np.array(label == i)][:,0]
		y1 = data[np.array(label == i)][:,1]
		plt.scatter(x1, y1, label = "{}".format(i), color = c)
	plt.scatter(center_points[:,0],center_points[:,1],color = "k",label = "centers")
	plt.legend()
	plt.show()

# we set those assignment matrix for centroid points
def assignment_matrix(data):
	zeros = np.zeros(data.shape)
	index = np.argmin(data, axis = 1 )
	zeros[range(data.shape[0]), index] = 1
	return zeros

def plot_image(x, y):
	plt.figure()
	plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
	x_major_locator = MultipleLocator(1)
	y_major_locator = MultipleLocator(1)
	# get the current axis
	ax = plt.gca()
	plt.xlim(-10,10)
	plt.ylim(-10,10)
	"""
	ax.xaxis.set_major_locator(x_major_locator)
	ax.yaxis.set_major_locator(y_major_locator)
	"""
	plt.scatter(x, y, marker = 'o', color = "black")
	plt.show()

if __name__ == "__main__":
	np.random.seed(33)
	# we set 3 initial 2-dimensional points
	randompoint = randomPoint(2,3)
	#data = trainProduct(2,30)

	data = np.loadtxt('3cluster_2d_data.csv', delimiter = ',')
	# show random points

	print('initial random points are:\n',randompoint)
	label, center_points = k_means(data,randompoint)
	print(label)
	plotting(label, data,center_points)


