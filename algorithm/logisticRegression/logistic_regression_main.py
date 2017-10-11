import numpy as np
from matplotlib import pyplot as plt
from logistic_regression import *

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


if __name__ == '__main__':
    # read data
    data = np.genfromtxt("./data/logistic_regression_2.txt", delimiter=',')
   
    # normalize input data
    data_new = z_score_nor_fuc(data[:, :-1])
    data_training = np.empty(data.shape)
    data_training[:, :-1] = data_new[:, :]
    data_training[:, -1] = data[:, -1]
    # np.random.shuffle(data)

    train_positive = data_training[np.where(data_training[:, -1] == 1)]
    train_negative = data_training[np.where(data_training[:, -1] == 0)]
    plt.figure(1)
    plt.subplot(121)
    plt.scatter(train_positive[:, 0], train_positive[:, 1], color='r')
    plt.scatter(train_negative[:, 0], train_negative[:, 1], color='b')

    # Training set initialization
    x_train = np.ones(data_training.shape)
    x_train[:, 1:] = data_training[:, :-1]
    y_train = data_training[:, -1:]

    # logisitc regression class
    lr = LogisticRegression(x_train, y_train, grad_type='NM', iterator_num=5)
    loss, gradient, theta = lr.regression(x_train.T, y_train.T)

    print("Loss value for each iterator:")
    print(loss)
    print("Gradient value for each iterator:")
    print(gradient)
    plt.plot([-2, 2], [regression_fuc(theta, -2), regression_fuc(theta, 2)], color='green')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Logistic regression result")

    plt.subplot(122)
    plt.plot(loss)
    plt.xlabel("iterators")
    plt.ylabel("loss value")
    plt.title("Logistic cost function loss")

    plt.show()

