import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# load data
data = np.loadtxt('sinusoid_example_data.csv', delimiter = ',')
data = torch.from_numpy(data)

x = data[:,:-1]
y = data[:,-1:]
dtype = torch.double
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = x.size()[0]
D_in = x.size()[1]
H = 20
D_out = y.size()[1]
#N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
#x = torch.randn(N, D_in, device=device, dtype=dtype)
#y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 0.02
for t in range(500):
	# Forward pass: compute predicted y
	h = x.mm(w1)
	h_relu = h.clamp(min=0)
	y_pred = h_relu.mm(w2)
	# Compute and print loss
	loss = (y_pred - y).pow(2).sum().item()
	if t % 100 == 99:
		print(t, loss)

	# Backprop to compute gradients of w1 and w2 with respect to loss
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(w2.t())
	grad_h = grad_h_relu.clone()
	grad_h[h < 0] = 0
	grad_w1 = x.t().mm(grad_h)

	# Update weights using gradient descent
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2

print("the w1:\n", w1)
print("the w2:\n", w2)

# we compute the prediction values
y_prediction = x.mm(w1).mm(w2)
print(y_prediction)
plt.figure()
plt.scatter(x,y)
plt.scatter(x,y_prediction)
plt.show()