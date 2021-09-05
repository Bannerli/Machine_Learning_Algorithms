# fit the curve

import numpy as np
import torch
import torch.nn.functional as TF

# Assuming we know that the desired function is polynomial of 2nd degree, we
# allocate a vector of size 3 to hold the coefficients and initialize it with
# random noise.

w = torch.tensor(torch.randn(3, 1), requires_grad = True)

# We use the Adam optimizer with learning rate set to 0.1 to minimize the loss.
opt = torch.optim.Adam([w], 0.1)

# we should design a
def model(x):
	# We define yhat to be your estimate of y.
	f = torch.stack((x * x, x, torch.ones_like(x)), dim = 1)
	yhat = torch.squeeze(f @ w, dim = 1)
	return yhat
def compute_loss(y, yhat):
	# The loss is defined to be the mean squared error distance between our
	# estimate of y and its true value
	loss = TF.mse_loss(yhat, y)
	return loss
def generate_data():
	# Generate some training data based on the true function
	x = torch.rand(100) * 20 - 10
	y = 5 * x * x + 3
	return x, y
def train_step():
	x, y = generate_data()
	yhat = model(x)
	loss = compute_loss(y, yhat)
	opt.zero_grad()
	loss.backward()
	opt.step()

# if we want to realize a linear model
class Net(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.a = torch.nn.Parameter(torch.rand(1), requires_grad = True)
		self.b = torch.nn.Parameter(torch.rand(1), requires_grad = True)
	def forward(self, x):
		yhat = self.a * x + self.b
		return yhat

x = torch.arange(100, dtype = torch.float32)/100
y = 5 * x + 3 + torch.rand(100) * 0.3
net = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)

for i in range(10000):
	yhat = net(x)
	loss = criterion(yhat, y)
	net.zero_grad()
	loss.backward()
	optimizer.step()

print(net.a, net.b)

