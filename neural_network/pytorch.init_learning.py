import torch.nn as nn
import torch
gain = nn.init.calculate_gain('leaky_relu', 0.2)
print("gain:\n", gain)
# product a empty matirx
tensor = torch.empty(3,5)
gain = nn.init.uniform_(tensor,a = 0.0,b = 1.0)
print("gain:\n", gain)
gain = nn.init.normal_(tensor, mean = 0, std = 1)
print("gain:\n", gain)
gian = nn.init.ones_(tensor)
print("gain:\n", gain)
w = nn.init.eye_(tensor)
print("w:\n", w)
