# import autograd functionally
import autograd.numpy as np
from autograd.misc.flatten import flatten_func
from autograd import grad as compute_grad
import math

# import various other libraries
import copy
import matplotlib.pyplot as plt

# this is needed to compensate for %matplotl+ib notebook's tendancy lotted inline
from matplotlib import rcParams
rcParams["figure.autolayout"] = True

# gradient descent function
def gradient_descent(g, w ,alpha, max_its, beta):
	# flatten the input function, create gradient based on flat function
	g_flat, unflatten, w = flatten_func(g, w)
	grad = compute_grad(g_flat)
	# record history
	w_hist = []
	# push the first w
	w_hist.append(unflatten(w))
	# start gradient descent loop
	z = np.zeros(np.shape(w))  # momentum term

	# over the line
	for k in range(max_its):
		# plug in value into func and derivative
		grad_eval = grad(w)
		grad_eval.shape = np.shape(w)
		# take descent step with momentum
		z = beta * z + grad_eval
		w = w -alpha * z
		# record weight update
		w_hist.append(unflatten(w))

	return w_hist

def model(x,w):
	result = np.dot(x, w)
	return result

def boosting():
	alpha = 0.02
	max_its = 20
	beta = 0
	w_initial = np.random.randn(matrix.shape[0],1)
	w = w_initial
	w_hist = []
	b = 0
	def least_square(w):
		# compute ls cost for the selected x part
		cost = np.sum(b + (model(x_slice, w) - y)**2)
		return cost
	for i in range(matrix.shape[1]):
		# x_slice is progressed for one step every iterations
		x_slice = matrix[:,i:i+1]
		w_batch_hist = gradient_descent(least_square, w[i:i+1], alpha, max_its, beta = 0)
		w[i:i+1] = w_batch_hist[-1]
		b = model(matrix[:,:i+1],w[:i+1])
		w_hist.append(w_batch_hist[-1])
	return w

sin = lambda w: np.sin(w)
cos = lambda w: np.cos(w)
tan = lambda w: np.tan(w)

def function_vector(a,b,c):
	result = np.array((sin(a),cos(b),tan(c)))
	return result.reshape(-1,1)

def matrix_function(w, function_list):
	for i in range(w.shape[1]):
		w[:,i] = np.array(list(map(eval(function_list[i]),w[:,i])))
	return w


if __name__ == "__main__":
	matrix = np.random.randn(50,5)
	y = np.random.randn(50,1)
	input_number = matrix.shape[0]
	input_first = np.ones((input_number,1))
	#print(input_first)
	matrix = np.concatenate((input_first,matrix), axis = 1)
	print(boosting())

	#print(function_vector(1,2,math.pi*2))
	func_matrix = np.random.randn(10,3)
	#print(matrix_function(func_matrix,["sin","cos","tan"]))





	#print(matrix)


