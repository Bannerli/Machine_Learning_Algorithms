import numpy as np
import matplotlib.pylab as plt

# define some classification
def step_function(x):
	return np.array(x > 0, dtype = np.int)

def sign(x):
	a_1 = np.array(x > 0, dtype = np.int)
	a_2 = np.array(x < 0, dtype = np.int)
	return a_1 * 1 + a_2 * (-1)

def relu(x):
	return np.maximum(0,x)

def model(x, w):
	a = w[0] + np.dot(x.T, w[1:])
	return a

def f1_plotting():
	x = np.linspace(-10,2,1000)
	y = np.log(1+np.exp(-x))
	plt.figure()
	plt.plot(x,y,linestyle = "solid")
	plt.show()
# now we generate some data to have a test
data = np.random.randn(4,1)
weight = np.random.randn(5,1)
model = model(data, weight)

data1 = np.array([1,0,1]).reshape(3,1)
data2 = np.array([1,2,4]).reshape(3,1)
f1_plotting()
"""
print(weight)h
ind = np.argwhere( weight >= 0)
print(weight[ind[:,0],ind[:,1]])
ones = np.ones((3,3))
print(step_function(weight))
print(sign(weight))
print(relu(weight))
"""



