import numpy as np
import copy
import matplotlib.pyplot as plt
import K_means


#data = np.loadtxt('circle_data.csv',delimiter = ',')
data = np.loadtxt('2d_span_data_centered.csv', delimiter = ',')
randompoints = K_means.randomPoint(2,2)
np.random.seed(123)
print(data.shape)
data = data
x = data[0]
y = data[1]
K_means.plot_image(x, y)

#label, center_points = K_means.k_means(data,randompoints)
#K_means.plotting(label,data,center_points)
#print(label)

