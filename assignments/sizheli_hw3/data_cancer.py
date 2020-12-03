import numpy as np

file = np.loadtxt("breast_cancer_data.csv",delimiter = ",")
data = file.T
x = data[:,:-1]

matrix = (np.random.randn(10,5)*10).astype(int)
index = np.arange(5)
y = (np.random.rand(10)*10).astype(int)%5

b = matrix[np.arange(np.size(y)),y.astype(int).flatten()]

def normalization(w):
	slice = w[1:]
	norm = np.sum(np.power(slice,2),axis = 0)**0.5
	result = w/norm
	return result,norm

def onehot(w):
	maxnum = np.array(np.max(w, axis = 1))
	maxnum = np.array(maxnum).reshape(10,1)
	encode = np.array(w >= maxnum, dtype = int)
	return encode*1.0

def multi_class_softmax(matrix):
	all_evals = matrix
	# compute the exp result
	init_result = np.exp(all_evals)
	total = np.sum(init_result, axis = 1)
	# select the corresponding column
	b = init_result[np.arange(np.size(y)),y.astype(int).flatten()]
	#cost = np.sum(b/total)
	return b

def encode_y(y, categories):
	number_y = np.size(y)
	a = np.zeros((number_y,np.max(y)+1))
	a[np.arange(number_y), y.astype(int).flatten()] = 1
	return a
#print(normalization(matrix))
print(onehot(matrix))
print(y)
print(encode_y(y,5))
print(multi_class_softmax(matrix))

import os
#print(os.getcwd())