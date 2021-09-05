import matplotlib as plt
import numpy as np

# neural network feature transform
def feature_transforms(a, w):
	# loop through each layer
	for W in w:
		# compute inner-product with current layer weight
		a = W[0] + np.dot(a, w[1:])
		# pass through activation
		a = activation(a)
	return a

def activation(x):
	return np.tanh(x)

def model(x, theta):
	# compute feature transform
	f = feature_transforms(x, theta[0])
	# compute final linear combination
	a = theta[1][0] + np.dot(f, theta[1][1:])
	return a

# create initial weights for a neural network model
def network_initializer(layer_sizes, scale):
	# container for all tunable weights
	weights = []
	[]
	for k in range(len(layer_sizes)-1):
		# get layer sizes for current weight matrix
		U_K = layer_sizes[k]
		U_K_plus =  layer_sizes[k+1]
		# make weight matrix
		weight = np.random.randn(U_K,U_K_plus) * scale
		weights.append(weight)
		# repackage weights so that theta_init[0] contains all
		# weight matrices internal to the network, and theta_init[1]
		# contains final linear combination weights
	theta_init = [weights[:-1], weights[-1]]
	return theta_init


	return theta_init
if __name__ == "__main__":
	layers = [2,3,4,5]
	#print(network_initializer(layers,0.5))
	# have a test about
	list_1 = np.asarray([-1,2,3,4,10])
	print(list_1.clip(0,5))