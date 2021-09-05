import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

# data
def create_data():
	iris = load_iris()
	keys = iris.keys()
	data = iris.data
	target = iris.target
	return np.array(data)[:100], np.array(target)[:100]

X, y = create_data()
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

class NaiveBayes():
	def __init__(self):
		self.model = None
	# the expectation
	@staticmethod
	def mean(X):
		return np.sum(X, axis = 0)/float(len(X))
	def stdev(self, X):
		avg = self.mean(X)
		return np.sqrt(np.mean((X - avg)**2, axis = 0))
	def gaussian_probability(self, X, mean, stdev):
		exponent = np.exp(-(X - mean)**2/(2 * stdev**2))
		return (1/(np.sqrt(2 * np.pi) * stdev))* exponent

	# 处理一下train data
	def summmarize(self, train_data):
		summarize = []
		for i in range(train_data.shape[1]):
			summarize.append((self.mean(train_data)[i], self.stdev(train_data)[i]))
		return summarize

	# 分类计算出数学期望和标准差
	def fit(self, X, y):
		labels = list(set(y))
		data = {label: [] for label in labels}
		for f, label in zip(X, y):
			data[label].append(f)
		for key in data:
			data[key] = np.array(data[key])
		self.model = {label: self.summmarize(value) for
					  label, value in data.items()
					  }
		return self.model

	def calculate_probabilities(self, input_data):
		# summaries:
		# input_data:
		probabilities = {}
		for label, value in self.model.items():
			probabilities[label] = 1
			for i in range(len(value)):
				mean, stdev = value[i]
				probabilities[label] *= self.gaussian_probability(input_data[:,i],
																  mean, stdev)
		return probabilities
		# 类别
	def predict(self, X_test):
			# {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
			probab = self.calculate_probabilities(X_test)
			label0 = probab[0]
			label1 = probab[1]
			result = np.array(label1 > label0, dtype = int)
			return result


model = NaiveBayes()
mean = model.mean(X_train)
stdva = model.stdev(X_train)
gaussian = model.gaussian_probability(X_train, mean, stdva)
summarize = model.summmarize(X_train)
data = model.fit(X_train, y_train)
print(model.predict(X_test))

