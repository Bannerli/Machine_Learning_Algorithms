import numpy as np
import torch
import torch.nn as nn
import os

all_files = os.listdir()
data_file = None
for file in all_files:
	if '4class' in file:
		data_file = file


data = np.loadtxt(data_file, delimiter = ',')

x_train = data[:-1].T
y_train = data[-1]

# transform those data into the tensor embedded by pytorch
x_train = torch.tensor(x_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.int64)

print(x_train.size())
print(y_train.size())

P = x_train.size()[0]
D_in = x_train.size()[1]
D_out = int(y_train.max()+1)
print(D_out)

# we may use the sequential function to organize those specific layers
model = torch.nn.Sequential(
	torch.nn.Linear(D_in, D_out),
)

# in this case we choose the MSE as our loss function
loss_function = torch.nn.CrossEntropyLoss(reduction = 'sum')
# choose one optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)

for t in range(800):
	# forward pass: compute predicted y passing x to the model
	y_pred = model(x_train)
	loss = loss_function(y_pred, y_train)
	if t % 100 == 99:
		print(t, loss)

	# zeros the gradients before running the backward pass.
	optimizer.zero_grad()

	# backword pass: compute the gradient of the loss with respect to all the learnable paramenters.
	loss.backward()
	# upadate weights using gradient descent
	optimizer.step()

def predict(x, y):
	y_prediction = model(x)
	result = torch.argmax(y_prediction, dim = 1)
	return (result == y).float().mean()

print(predict(x_train, y_train))
print(y_train)
