import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as lg


def regression_fuc(theta, x):
    """
    Generate the liner function
    :param theta:
    :param x:
    :return: liner function
    """
    return -1 * (theta[0, 0] / theta[2, 0]) - (theta[1, 0] / theta[2, 0]) * x


def z_score_nor_fuc(x):
    """
    z-score normalization
    :param x:
    :return: normalized data
    """
    miu = np.mean(x, axis=0)
    delta = np.std(x, axis=0)
    return (x - miu)/delta


def min_max_nor_fuc(x):
    """
    min-max normalization
    :param x:
    :return:
    """
    min = np.min(x, axis=0)
    max = np.max(x, axis=0)
    return (x - min)/(max - min)

data = np.genfromtxt("C:/Users/zhaoyao/Desktop/data/logistic_regression_2.txt", delimiter=',')
data_new = z_score_nor_fuc(data[:, :-1])
data_training = np.empty(data.shape)
data_training[:, :-1] = data_new[:, :]
data_training[:, -1] = data[:, -1]
# np.random.shuffle(data)

# data_training, data_testing, empty = np.split(data, [70, 100])
data_testing = np.empty(data.shape)

train_positive = data_training[np.where(data_training[:, -1] == 1)]
train_negative = data_training[np.where(data_training[:, -1] == 0)]
plt.figure(1)
plt.subplot(121)
plt.scatter(train_positive[:, 0], train_positive[:, 1], color='r')
plt.scatter(train_negative[:, 0], train_negative[:, 1], color='b')


class LogisticRegression:
    """
    Logistic regression
    """
    def __init__(self, feature_train, label_train, feature_test, label_test):
        """
        alpha = 0.001 & iterator_num = 100000 for all batches
        iterator_num = 5 for Newton's Method
        :param feature_train:
        :param label_train:
        :param feature_test:
        :param label_test:
        """
        self.x_train = feature_train.T
        self.y_train = label_train.T
        self.x_test = feature_test.T
        self.y_test = label_test.T
        self.theta = np.random.random((self.x_train.shape[0], 1))
        self.alpha = 0.01
        self.alpha_rate = 1
        self.batch = 1
        self.iterator_num = 5

    def regression(self, x_array, y_array, opt_type):
        """
        Logistic regression iterator
        :param x_array:
        :param y_array:
        :param opt_type:
                GD represents gradient descent
                NM represents Newton's Method
        :return: loss array & gradient array & theta value
        """
        # initialize optimization parameters
        loss_array = list()
        grad_array = list()
        alpha = self.alpha
        theta_new = self.theta
        batch = self.batch
        loss_array.append(self.cost_func(theta_new, x_array, y_array))
        # iterator for regression,
        for iterator in range(self.iterator_num):
            if opt_type == 'GD':
                theta_new, gradient = self.mini_batch_gradient_descent(theta_new, x_array, y_array, alpha, batch)
                alpha *= self.alpha_rate
            elif opt_type == 'NM':
                theta_new, gradient = self.newton_method(theta_new, x_array, y_array)
            grad_array.append(gradient)
            loss_array.append(self.cost_func(theta_new, x_array, y_array))
        return np.array(loss_array), np.array(grad_array), theta_new

    def hypothesis_fuc(self, theta, x_array):
        """
        Sigmoid function for logistic regression
        :param theta: weight and bias
        :param x_array: feature arrays
        :return: hypothesis value
        """
        hypothesis = 1/(1 + np.exp(-1 * (theta.T.dot(x_array))))
        return hypothesis

    def cost_func(self, theta, x_array, y_array):
        """
        Cost function
        :param theta: weight and bias
        :param x_array: feature arrays
        :param y_array: label arrays
        :return: loss value
        """
        hypothesis = self.hypothesis_fuc(theta, x_array).T
        loss_mat = -1 * (y_array.dot(np.log(hypothesis)) + (1 - y_array).dot(np.log(1 - hypothesis))) / (x_array.shape[1])
        return loss_mat[0, 0]

    def mini_batch_gradient_descent(self, theta, x_array, y_array, alpha, batch):
        """
        Batch gradient descent
        :param theta: previous theta value
        :param x_array: feature arrays
        :param y_array: label arrays
        :param alpha: learning rate
        :param batch: batch_size
        :return: new theta value & gradient array
        """
        y_min_h = y_array - self.hypothesis_fuc(theta, x_array)
        num = np.random.choice(np.arange(x_array.shape[1]), batch)
        gradient = -1*(x_array[:, num].dot(y_min_h[:, num].T))/(x_array.shape[1])
        return theta - alpha * gradient, gradient[:, 0].T

    def newton_method(self, theta, x_array, y_array,):
        """
        Newton's method
        :param theta: previous theta value
        :param x_array: feature arrays
        :param y_array: label arrays
        :return: new theta value
        """
        hypothesis = self.hypothesis_fuc(theta, x_array)
        _h = hypothesis * (1 - hypothesis)
        _h_dig = np.diag(_h.reshape(_h.shape[1],))
        hessian = (x_array.dot(_h_dig)).dot(x_array.T)
        y_min_h = y_array - hypothesis
        gradient =x_array.dot(y_min_h.T)
        return theta + (gradient.T.dot(lg.inv(hessian))).T, gradient[:, 0].T


if __name__ == '__main__':
    # Training set initialization
    x_train = np.ones(data_training.shape)
    x_train[:, 1:] = data_training[:, :-1]
    y_train = data_training[:, -1:]

    # Testing set initialization
    x_test = np.ones(data_testing.shape)
    x_test[:, 1:] = data_testing[:, :-1]
    y_test = data_testing[:, -1:]

    # Training stage
    lr = LogisticRegression(x_train, y_train, x_test, y_test)
    loss, gradient, theta = lr.regression(x_train.T, y_train.T, opt_type='NM')
    print(loss)
    print(gradient)
    print(theta)
    plt.plot([-2, 2], [regression_fuc(theta, -2), regression_fuc(theta, 2)], color='green')
    plt.subplot(122)
    plt.plot(loss)
    plt.show()


