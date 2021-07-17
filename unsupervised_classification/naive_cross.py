import numpy as np

def polynomial_matrix(x, w):
	matrix = x
	for i in range(2, w.shape[0]):
		matrix = np.concatenate((matrix, x**i), axis = 1)
	return matrix

def gradually_complexity(x, polynum):
	poly_matrix = []
	for i in range(1,polynum):
		w = np.random.randn(1+i,1)
		x_processed = polynomial_matrix(x, w)
		poly_matrix.append(x_processed)

	return poly_matrix
training = np.random.randint(1,5,(10,1))
weight = np.random.randn(5,1)
print(gradually_complexity(training,5))
