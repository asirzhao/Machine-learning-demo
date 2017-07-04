import numpy as np
from matplotlib import pyplot as plt

data = np.genfromtxt("C:/Users/zhaoyao/Desktop/data/liner_regression_data.csv", delimiter=',')
plt.figure(1)
plt.subplot(121)
plt.scatter(data[:, 0], data[:, 1])


class LineRegression:
    """
    Class for liner regression
    """
    def __init__(self, x, y):
        """
        Init the value of class
            x_array: array of x data
            y_array: array of y data
            theta: array of weight and bias
            alpha: learning rate
            alpha_rate: alpha decay rate
            iterator_num: iterator times of gradient descent
            batch: size of batch
        """
        self.x_array = x.T
        self.y_array = y.T
        self.theta = np.random.random((self.x_array.shape[0], 1))
        self.alpha = 0.0001
        self.alpha_rate = 0.9465
        self.iterator_num = 100
        self.batch = 10

    def regression(self, grad_type):
        """
         Liner regression
        :param grad_type:
            BGD for batch gradient descent
            SGD for stochastic gradient descent
            MBGD for mini-batch gradient descent
        :return: loss, gradient and theta
        """
        loss_array = list()
        gradient_array = list()
        theta_new = self.theta
        gradient = np.zeros((self.iterator_num, self.x_array.shape[0]))
        global_loss = self.cost_fuc(theta_new)
        loss_array.append(global_loss)
        alpha = self.alpha
        for iterator in range(self.iterator_num):
            if grad_type == 'BGD':
                theta_new, gradient = self.batch_gradient_descent(theta_new)
            elif grad_type == 'SGD':
                theta_new, gradient = self.stochastic_gradient_descent(theta_new, alpha)
            elif grad_type == 'MBGD':
                theta_new, gradient = self.mini_batch_gradient_descent(theta_new, self.batch)
            global_loss = self.cost_fuc(theta_new)
            loss_array.append(global_loss)
            gradient_array.append(gradient)
            alpha *= self.alpha_rate

        return np.array(loss_array), np.array(gradient_array), theta_new

    def hypothesis_fuc(self, theta):
        """
        Hypothesis function for liner regression
        :param theta:
        :return: hypothesis value
        """
        return theta.T.dot(self.x_array)

    def cost_fuc(self, theta):
        """
        Cost function for liner function
        :param theta:
        :return: cost(loss) value
        """
        loss_mat = ((self.hypothesis_fuc(theta) - self.y_array)**2).sum(axis=1)/(2 * self.x_array.shape[1])
        return loss_mat[0]

    def batch_gradient_descent(self, theta):
        """
        Batch gradient descent
        :param theta:
        :return: new theta value
        """
        x_array = self.x_array
        h_y_min = theta.T.dot(x_array) - self.y_array
        gradient = h_y_min.dot(x_array.T).T/x_array.shape[1]
        return theta - self.alpha * gradient, gradient[:, 0].T

    def stochastic_gradient_descent(self, theta, alpha):
        """
        Stochastic gradient descent
        :param theta:
        :return: new theta value
        """
        x_array = self.x_array
        h_y_min = theta.T.dot(x_array) - self.y_array
        num = np.random.choice(np.arange(x_array.shape[1]), 1)
        gradient = h_y_min[0, num] * x_array.T[num, :].T
        return theta - alpha * gradient, gradient[:, 0].T

    def mini_batch_gradient_descent(self, theta, batch):
        """
        Mini-batch gradient descent
        :param theta:
        :param batch:
        :return: new theta value
        """
        x_array = self.x_array
        h_y_min = theta.T.dot(x_array) - self.y_array
        num = np.random.choice(np.arange(x_array.shape[1]), batch)
        gradient = h_y_min[0, num] * x_array.T[num, :].T
        return theta - self.alpha * gradient, gradient[:, 0].T

    def regression_fuc(self, theta, x):
        return theta[0, 0] + x * theta[1, 0]

if __name__ == '__main__':

    x_array = np.ones(data.shape)
    x_array[:, 1] = data[:, 0]
    y_array = data[:, 1:]

    lr = LineRegression(x_array, y_array)
    loss, gradient, theta = lr.regression(grad_type='MBGD')
    print(loss)
    print(gradient)
    plt.plot([20, 80], [lr.regression_fuc(theta, 20), lr.regression_fuc(theta, 80)], color='r')

    plt.subplot(122)
    plt.plot(loss)

    plt.show()
