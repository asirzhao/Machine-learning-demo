import numpy as np
import numpy.linalg as lg


class LogisticRegression:
    """
    Logistic regression
        Mini-batch gradient descent and Newton's method are provided to optimze the log cost function. The result
      presents that Newton's method converge much faster than gradient descent.
        See more information about Newton'method and convex function optimzation:
	1.https://en.wikipedia.org/wiki/Newton%27s_method
	2.tps://see.stanford.edu/materials/lsocoee364a/03ConvexFunctions.pdf
    """
    def __init__(self, feature_train, label_train, grad_type='GD', alpha=0.001, alpha_rate=1, batch_size=10, iterator_num=100000):
        """
        :param feature_train: input x
        :param label_train: input y
        :param grad_type: GD for mini-batch gradient descent and NM for Newton's method
        :param alpha: learning rate
        :param alpha_rate: learning rate decay
        :param batch_size: batch size for input data
        :param iterator_num: iterator numbers
       """
        self.x = feature_train.T
        self.y = label_train.T
        self.grad_type = grad_type
        self.__theta = np.random.random((self.x.shape[0], 1))
        self.alpha = alpha
        self.alpha_rate = alpha_rate
        self.batch = batch_size
        self.iterator_num = iterator_num

    def regression(self, x_array, y_array):
        """
        Logistic regression iterator
        :param x_array:
        :param y_array:
        :return: loss array & gradient array & theta value
        """
        # initialize optimization parameters
        loss_array = list()
        grad_array = list()
        alpha = self.alpha
        theta_new = self.__theta
        batch = self.batch
        loss_array.append(self.__cost_func(theta_new, x_array, y_array))
        grad_type = self.grad_type
        # iterator for regression
        for iterator in range(self.iterator_num):
            if grad_type == 'GD':
                theta_new, gradient = self.__mini_batch_gradient_descent(theta_new, x_array, y_array, alpha, batch)
                alpha *= self.alpha_rate
            elif grad_type == 'NM':
                theta_new, gradient = self.__newton_method(theta_new, x_array, y_array)
            grad_array.append(gradient)
            loss_array.append(self.__cost_func(theta_new, x_array, y_array))
        return np.array(loss_array), np.array(grad_array), theta_new

    def __hypothesis_fuc(self, theta, x_array):
        """
        Sigmoid function for logistic regression
        :param theta: weight and bias
        :param x_array: feature arrays
        :return: hypothesis value
        """
        hypothesis = 1/(1 + np.exp(-1 * (theta.T.dot(x_array))))
        return hypothesis

    def __cost_func(self, theta, x_array, y_array):
        """
        Cost function
        :param theta: weight and bias
        :param x_array: feature arrays
        :param y_array: label arrays
        :return: loss value
        """
        hypothesis = self.__hypothesis_fuc(theta, x_array).T
        loss_mat = -1 * (y_array.dot(np.log(hypothesis)) + (1 - y_array).dot(np.log(1 - hypothesis))) / (x_array.shape[1])
        return loss_mat[0, 0]

    def __mini_batch_gradient_descent(self, theta, x_array, y_array, alpha, batch):
        """
        Batch gradient descent
        :param theta: previous theta value
        :param x_array: feature arrays
        :param y_array: label arrays
        :param alpha: learning rate
        :param batch: batch_size
        :return: new theta value & gradient array
        """
        y_min_h = y_array - self.__hypothesis_fuc(theta, x_array)
        num = np.random.choice(np.arange(x_array.shape[1]), batch)
        gradient = -1*(x_array[:, num].dot(y_min_h[:, num].T))/(x_array.shape[1])
        return theta - alpha * gradient, gradient[:, 0].T

    def __newton_method(self, theta, x_array, y_array,):
        """
        Newton's method
        :param theta: previous theta value
        :param x_array: feature arrays
        :param y_array: label arrays
        :return: new theta value
        """
        hypothesis = self.__hypothesis_fuc(theta, x_array)
        _h = hypothesis * (1 - hypothesis)
        _h_dig = np.diag(_h.reshape(_h.shape[1],))
        hessian = (x_array.dot(_h_dig)).dot(x_array.T)
        y_min_h = y_array - hypothesis
        gradient =x_array.dot(y_min_h.T)
        return theta + (gradient.T.dot(lg.inv(hessian))).T, gradient[:, 0].T

