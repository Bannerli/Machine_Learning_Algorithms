import numpy as np
import tensorflow as tf
import csv


# write some general functions
def sin(x):
	return np.sin(x)

def cos(x):
	return np.cos(x)

def tan(x):
	return np.tan(x)

# product the sample matrix
with open('kleibers_law_data.csv') as data:
	file = csv.reader(data, delimiter = ',')
	feature_list = []
	target_list = []
	for row in file:
		feature_list.append(row[:-1])
		target_list.append(row[-1])
	feature_items = np.array(feature_list)
	target_items = np.array(target_list)
print("features:\n",feature_list)
print("targets:\n",target_items)



