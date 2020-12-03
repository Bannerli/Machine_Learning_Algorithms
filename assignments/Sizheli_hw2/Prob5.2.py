# import autograd functionality
import autograd.numpy as np
from autograd.util import quick_grad_check
from autograd import grad as compute_grad
from autograd.misc.flatten import flatten_func
import matplotlib.pylab as plt
# this is needed to compensate for %matplotl+ib notebook's tendancy lotted inline
from matplotlib import rcParams
rcParams["figure.autolayout"] = True

# gradient descent function
def gradient_descent(g, alpha, max_its, w, beta):
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = compute_grad(g_flat)

    # record history
    w_hist = []
    w_hist.append(unflatten(w))

    # start gradient descent loop
    z = np.zeros((np.shape(w)))  # momentum term

    # over the line
    for k in range(max_its):
        # plug in value into func and derivative
        grad_eval = grad(w)
        grad_eval.shape = np.shape(w)
        """ normalized or unnormalized descent step? """
        # take descent step with momentum
        z = beta * z + grad_eval
        w = w - alpha * z

        # record weight update
        w_hist.append(unflatten(w))

csvname = "kleibers_law_data.csv"
data = np.loadtxt(csvname, delimiter = ',')
data = data.T
x = data[:,0]
y = data[:,-1]
x = np.log(x)
y = np.log(y)
print(x.shape)

# define least square cost function
def least_squares(w):
    cost = 0
    for p in range(len(y)):
        cost += (w[0] + w[1]*x[p] - y[p])**2
    return cost

# initialize parameters
alpha = 0.0005
max_its = 500
w_init = np.asarray([2,0.75])

# run gradient descent, create cost function history
weight_history = gradient_descent(least_squares, alpha, max_its, w_init, beta = 0)
cost_history = [least_squares(v) for v in weight_history]
best_weight = weight_history[-1]
# scatter plot the input data
fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.scatter(x,y,color = 'k',edgecolor = 'w')
x_vals = np.linspace(np.min(x),np.max(y),200)
y_vals = best_weight[0] + best_weight[1]*x_vals
ax.plot(x_vals,y_vals,color = 'r')
plt.show()