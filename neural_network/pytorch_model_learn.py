import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# load data
data = np.loadtxt('sinusoid_example_data.csv', delimiter = ',')
data = torch.from_numpy(data).float()

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
learning_rate = 0.01
opt = torch.optim.SGD(model.parameters(), lr = learning_rate)


for t in range(500):
	# forward pass: compute predicted y passing x to the model
	y_pred = model(x)
	loss = loss_function(y_pred, y)
	if t % 100 == 99:
		print(t, loss)

	# zeros the gradients before running the backward pass.
	model.zero_grad()

	# backword pass: compute the gradient of the loss with respect to all the learnable paramenters.
	loss.backward()
	# upadate weights using gradient descent
	opt.step()
	opt.zero_grad()

# print(model.parameters())
y_prediction = model(x)
print("the prediction of that:\n", y_prediction)
print("loss_function:\n", loss_function(y_prediction, y))

plt.figure()
plt.scatter(x,y_prediction.detach().numpy())
plt.scatter(x,y)
plt.show()