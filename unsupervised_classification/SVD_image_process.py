from torch import  nn
import numpy as np


class LinearRegression(nn.Module):
	def __init__(self):
		super(LinearRegression, self).__init__()
		self.linear = nn.Linear(in_features = 1, out_features = 1)
		self.parameters = nn.Parameter(1,1)

	def forward(self, x):
		out = self.linear(x)
		return out

def SVD_transform():
	matrix = np.random.randint(low = 0, high = 10, size = (4,3))
	U,Sigma,V_T = np.linalg.svd(matrix)
	return matrix, U, Sigma, V_T

if __name__ == '__main__':
	"""
	model = LinearRegression()
	for layer, param in model.state_dict().items():
		print(layer, param)
	"""
	matrix, U, Sigma, V_T = SVD_transform()

	m = matrix.shape[0]
	n = matrix.shape[1]
	sigma = np.zeros(shape = (m,n))
	for i in range(len(Sigma)):
		sigma[i][i] = Sigma[i]
	print(np.sum(U**2, axis = 0))
	print(sigma)
	print("original matrix:\n", matrix)
	result = np.dot(np.dot(U, sigma), V_T)
	print("redo the matrix:\n", result)

