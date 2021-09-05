import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# load data
data = np.loadtxt('sinusoid_example_data.csv', delimiter = ',')
data = torch.from_numpy(data).float()
print(data.dtype)
# Create random tensors to hold inputs and outputs
x = data[:,:-1]
y = data[:,-1:]

N = x.size()[0]
D_in = x.size()[1]
H_1 = 10
H_2 = 20
H_3 = 20
D_out = y.size()[1]

# we may use the sequential function to organize those specific layers
model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H_1),
	torch.nn.ReLU(),
	torch.nn.Linear(H_1, H_2),
	torch.nn.ReLU(),
	torch.nn.Linear(H_2, H_3),
	torch.nn.ReLU(),
	torch.nn.Linear(H_3, D_out),
)

# the nn package also contains definitions of populat loss functions;
# in this case we choose the MSE as our loss function
loss_function = torch.nn.MSELoss(reduction = 'sum')
# choose one optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

for t in range(500):
	# forward pass: compute predicted y passing x to the model
	y_pred = model(x)
	loss = loss_function(y_pred, y)
	if t % 100 == 99:
		print(t, loss)

	# zeros the gradients before running the backward pass.
	optimizer.zero_grad()

	# backword pass: compute the gradient of the loss with respect to all the learnable paramenters.
	loss.backward()
	# upadate weights using gradient descent
	optimizer.step()

print(model.parameters())
y_prediction = model(x)


# product a interval
x_space = torch.linspace(x.min(), x.max(), steps = 100).reshape(100,1)
y_space_predict = model(x_space)
print(x_space)
plt.figure()
plt.plot(x_space, y_space_predict.detach().numpy())
plt.scatter(x,y)
plt.show()