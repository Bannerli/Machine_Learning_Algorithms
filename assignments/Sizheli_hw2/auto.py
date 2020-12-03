# import autograd functionally
import autograd.numpy as np
from autograd.misc.flatten import flatten_func
from autograd import grad as compute_grad

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

csvname = "boston_housing.csv"
data = np.loadtxt(csvname, delimiter = ',')
data = data.T
x = data[:,:-1]
y = data[:,-1:]

# we do the normalization of these data matrix
x_means = np.mean(x, axis = 0)
x_stds = np.std(x, axis = 0)

def normalize(data, data_mean, data_std):
	normalized = (data - data_mean)/data_std
	return normalized

# normalize the input data
x_normed = normalize(x, x_means, x_stds)

# our prediction function
def predict_normalized(x ,w):
	# feature transformations
	f = w[0] + np.dot(x, w[1:])
	return f

# define least square cost function
least_squares_normalized = lambda w: np.sum((predict_normalized(x_normed, w) - y)**2)

# initialize parameters
alpha = 10**-4
max_its = 100
beta = 0
w_init = np.random.randn(x.shape[1]+1, 1)
# run gradient descent, create cost function history
weight_history = gradient_descent(least_squares_normalized, w_init, alpha, max_its,beta)
cost_history = [least_squares_normalized(v) for v in weight_history]

csvname_ = "auto_data.csv"
data_ = np.loadtxt(csvname, delimiter = ',')
data_ = data.T
x_ = data[:,:-1]
y_ = data[:,-1:]
print(x_[:,2])