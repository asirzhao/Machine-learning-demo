import numpy as np
from matplotlib import pyplot as plt
from liner_regression import *


data = np.genfromtxt("./data/liner_regression_data.csv", delimiter=',')
plt.figure(1)
plt.subplot(121)
plt.scatter(data[:, 0], data[:, 1])
     
x_array = np.ones(data.shape)
x_array[:, 1] = data[:, 0]
y_array = data[:, 1:]

# choose one grad_type to see the optimzation result. Batch_size is needed only if you choose MBGD.
lr = LineRegression(x=x_array, y=y_array, grad_type='BGD', alpha=0.0001, alpha_rate=0.9465, iterator_num=20)
loss, gradient, theta = lr.regression() 
print("Loss of each iteration:")
print(loss)
print("Gradient value of each iteration:")
print(gradient)
   
x_min=20
x_max=80
plt.plot([x_min, x_max], [lr.regression_fuc(theta, x_min), lr.regression_fuc(theta, x_max)], color='r')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Liner regression")
    
plt.subplot(122)
plt.plot(loss)
plt.xlabel("iterators")
plt.ylabel("loss value")
plt.title("Liner regression loss")
    
plt.show()

