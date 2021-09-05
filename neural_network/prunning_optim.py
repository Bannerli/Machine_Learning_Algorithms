import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		# 1 input image channel, 6 output channels, 3*3 square conv kernel
		self.conv1 = nn.Conv2d(1,6,3,)
		# 6 input image channel, 16 output channels, 3*3 square conv kernel
		self.conv2 = nn.Conv2d(6,16,3)
		self.pool = nn.MaxPool2d(2,2)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc1 = nn.ReLU()
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)


	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, int(x.nelement()/x.shape[0]))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

model = LeNet().to(device = device)
module = model.conv1
print(list(module.named_parameters()))
#print("weight of the module:", module.weight)
#print("bias of the module:", module.bias)

prune.random_unstructured(module, name = 'weight',
						  amount = 0.3)
print(list(module.named_parameters()))

