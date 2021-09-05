import torch
import numpy as np
import matplotlib.pyplot as plt

class TwolayerNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		"""
		we assign the follwoing paramenters:
		:param D_in:
		:param H:
		:param D_out:
		"""
		super(TwolayerNet,self).__init__()
		self.linear1 = torch.nn.Linear(D_in, H)
		self.relu = torch.nn.ReLU()
		self.linear2 = torch.nn.Linear(H, D_out)
	def forward(self, x):
		"""
		in the forward function we accept a Tensor of input datand we must return a tensor of
		output data. We can use Modules defined in the constructor
		:param x:
		:return:
		"""
		# nn.Linear is a module to compute the matrix multiply
		h_relu = self.linear1(x).clamp(min = 0)
		y_pred = self.linear2(h_relu)
		return y_pred

if __name__ == "__main__":
	# N is batch size; D_in is input dimension;
	# H is hidden dimension; D_out is output dimension.
	N, D_in, H, D_out = 64, 1000, 100, 10

	# Create random Tensors to hold inputs and outputs
	x = torch.randn(N, D_in)
	y = torch.randn(N, D_out)

	# Construct our model by instantiating the class defined above
	model = TwolayerNet(D_in, H, D_out)
	# Construct our loss function and an Optimizer. The call to model.parameters()
	# in the SGD constructor will contain the learnable parameters of the two
	# nn.Linear modules which are members of the model.
	criterion = torch.nn.MSELoss(reduction = 'sum')
	optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)
	for t in range(500):
		# Forward pass: Compute predicted y by passing x to the model
		# y_pred = model(x)
		y_pred = model.forward(x)

		# Compute and print loss
		loss = criterion(y_pred, y)
		if t % 100 == 99:
			print(t, loss.item(),type(loss))
		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	y_prediction = model.forward(x)
	print(y_prediction[0])
	print(y[0])
