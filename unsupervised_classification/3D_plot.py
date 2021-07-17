import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
import os

rcParams["axes.unicode_minus"] = False # 用来正确显示负号


def data_plotsurface_3D():
	x = np.arange(-2, 1.5, 0.01)
	y = np.arange(-2, 1.5, 0.01)
	x, y = np.meshgrid(x, y)
	result = x + y + 0.5*(2 * x**2 +3 * y**2 + 2*x*y)
	figure = plt.figure()
	ax = Axes3D(figure)
	ax.plot_surface(x, y, result, cmap = "rainbow")
	ax.set_xlabel("w1")
	ax.set_ylabel("w2")
	ax.set_zlabel("g(w)")
	plt.show()

def data_plot_3D(data):
	y = data[:,0]
	x = data[:,1]
	z = data[:,2]
	figure = plt.figure()
	ax = Axes3D(figure)
	#ax.scatter(x,y,z, cmap = "rainbow")
	ax.set_zlabel("x value")
	ax.set_ylabel("y value")
	ax.set_zlabel("z value")
	plt.show()

if __name__ == "__main__":
	data = np.loadtxt('3d_span_data.csv', delimiter = ',')
	data_plot_3D(data.T)




