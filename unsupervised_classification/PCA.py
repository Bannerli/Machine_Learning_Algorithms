import numpy as np
import matplotlib.pyplot as plt
import os
import K_means
import copy

# we write those functions for N*P matirx

# center an input dataset X
def center(X):
	X_means = np.mean(X, axis = 1)[:, np.newaxis]
	X_centered = X - X_means
	return X_centered

def compute_pca(X,lam):
	# create the data covariance matrix
	p = float(X.shape[1])
	Cov = 1/p * np.dot(X, X.T) + lam * np.eye(X.shape[0])

	# use numpy function to compute eigenvectors / eigenvalues
	eigenvalues, eigenvectors = np.linalg.eig(Cov)
	# wo do the rank of the eigenvalues
	eigenvalue_order = np.argsort(eigenvalues)
	# we need the order from high to low
	#eigenvalue_order = eigenvalue_order[::-1]
	# so we have the spanning matrix
	C_matrix = eigenvectors.T[eigenvalue_order]
	return C_matrix

if __name__ == "__main__":
	dirlist = os.listdir()
	text = None
	for file in dirlist:
		if '2d' in file:
			text = file
	data = np.loadtxt(text, delimiter = ',')
	x_origin = data[0]
	y_origin = data[1]
	# plot the origin picture
	K_means.plot_image(x_origin,y_origin)
	centered_data = center(data)
	C_matrix = -1*compute_pca(centered_data, 10**-5)
	print("The spanning matrix:\n",C_matrix)
	# plot the vector arrow
	x_vector = np.zeros(3)
	y_vector = np.zeros(3)
	x_vector[:2] = C_matrix.T[0]
	y_vector[:2] = C_matrix.T[1]
	K_means.plot_image(x_vector,y_vector)
	# Now we transform the origin data
	data_transformed = np.dot(C_matrix, data)
	x_transformed = data_transformed[0]
	y_transformed = data_transformed[1]
	K_means.plot_image(x_transformed,y_transformed)



