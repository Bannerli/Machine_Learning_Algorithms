import torch
from torch.autograd.functional import hessian

def pow_addr_reducer(x, y):
	return (2 * x.pow(2) + 3 * y.pow(2)).sum()

inputs = (torch.FloatTensor(1),torch.FloatTensor(1))
print(hessian(pow_addr_reducer, inputs))