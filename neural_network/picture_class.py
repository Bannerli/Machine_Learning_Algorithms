import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 载入训练数据
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)
# 载入测试数据集
testset = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


dataiter = iter(trainloader)
images, labels = dataiter.next()
picture_1 = images[0]
print(labels.dtype)


print(images.shape)
print('真实标签: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))
"""
pic = np.transpose(picture_1, (1, 2, 0))
plt.imshow(pic/2 + 0.5)
plt.show()
"""

# define a conv net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # transform two dimensions into one dimension
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


net = Net()
# write the loss function as the criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# test the data form

result = net(images)
print(result)
print(result.data)

# iterate twice

for epoch in range(2):
    running_loss = 0.
    for i, data in enumerate(trainloader, 0):
        # get access to the data
        inputs, labels = data
        # bound variable to the inputs and labels
        inputs, labels = Variable(inputs), Variable(labels)
        # initial gradient parameters
        optimizer.zero_grad()
        # forward propagation
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# test the performance of the testing set
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    values, predicted_index = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted_index == labels).sum()
print("accuracy: %d %%"%(100*correct//total))





