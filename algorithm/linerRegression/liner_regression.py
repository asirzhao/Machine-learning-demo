import numpy as np
from matplotlib import pyplot as plt


class LineRegression:
    """
    Class for liner regression
        the loss or cost function of liner regression is minmiziate square error(MSE), to optimize MSE, batch gradient descent, stochastic gradient
      descent and mini-batch gradient descent method are provided.
        See more infomation about gradient decent method, http://sebastianruder.com/optimizing-gradient-descent/
    """
    def __init__(self, x, y, grad_type, alpha, alpha_rate, iterator_num, batch_size=20):
        """
        Init the value of class
            x_array: array of x data
            y_array: array of y data
            theta: array of weight and bias
            grad_type:
              BGD for batch gradient descent
              SGD for stochastic gradient descent
              MBGD for mini-batch gradient descent
            alpha: learning rate
            alpha_rate: alpha decay rate
            iterator_num: iterator times of gradient descent. For BGD, 10 to 30 times could make it, while for SGD, more than 100 times may make it.
            batch: size of batch
        """
        self.x_array = x.T
        self.y_array = y.T
        self.__theta = np.random.random((self.x_array.shape[0], 1))
        self.grad_type = grad_type
        self.alpha = alpha
        self.alpha_rate = alpha_rate
        self.iterator_num = iterator_num
        self.batch = batch_size

    def regression(self):
        """
         Liner regression
         return: loss, gradient and theta
        """
        grad_type = self.grad_type
        loss_array = list()
        gradient_array = list()
        theta_new = self.__theta
        gradient = np.zeros((self.iterator_num, self.x_array.shape[0]))
        global_loss = self.__cost_fuc(theta_new)
        loss_array.append(global_loss)
        alpha = self.alpha
        for iterator in range(self.iterator_num):
            if grad_type == 'BGD':
                theta_new, gradient = self.__batch_gradient_descent(theta_new)
            elif grad_type == 'SGD':
                theta_new, gradient = self.__stochastic_gradient_descent(theta_new, alpha)
            elif grad_type == 'MBGD':
                theta_new, gradient = self.__mini___batch_gradient_descent(theta_new, self.batch)
            global_loss = self.__cost_fuc(theta_new)
            loss_array.append(global_loss)
            gradient_array.append(gradient)
            alpha *= self.alpha_rate

        return np.array(loss_array), np.array(gradient_array), theta_new

    def __hypothesis_fuc(self, theta):
        """
        Hypothesis function for liner regression
        :param theta:
        :return: hypothesis value
        """
        return theta.T.dot(self.x_array)

    def __cost_fuc(self, theta):
        """
        Cost function for liner function
        :param theta:
        :return: cost(loss) value
        """
        loss_mat = ((self.__hypothesis_fuc(theta) - self.y_array)**2).sum(axis=1)/(2 * self.x_array.shape[1])
        return loss_mat[0]

    def __batch_gradient_descent(self, theta):
        """
        Batch gradient descent
        :param theta:
        :return: new theta value
        """
        x_array = self.x_array
        h_y_min = theta.T.dot(x_array) - self.y_array
        gradient = h_y_min.dot(x_array.T).T/x_array.shape[1]
        return theta - self.alpha * gradient, gradient[:, 0].T

    def __stochastic_gradient_descent(self, theta, alpha):
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

    def __mini___batch_gradient_descent(self, theta, batch):
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

