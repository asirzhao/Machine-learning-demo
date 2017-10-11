import numpy as np
from matplotlib import pyplot as plt
from svm import *


def z_score_nor_fuc(x):
    """
    z-score normalization
    :param x:
    :return: normalized data
    """
    miu = np.mean(x, axis=0)
    delta = np.std(x, axis=0)
    return (x - miu)/delta


def liner_classify_func(w, b, x):
    """
    classify line for liner SVM
    :param w: weights
    :param b: bias
    :param x: x1
    :return: x2
    """
    return -1*(b+w[0, 0]*x)/w[0, 1]

if __name__ == '__main__':
    data = np.genfromtxt("./data/svm_rbf.csv", delimiter=',')
    data_new = z_score_nor_fuc(data[:, :-1])
    data[:, :-1] = data_new

    data_pos = data[np.where(data[:, -1] == 1)]
    data_neg = data[np.where(data[:, -1] == -1)]

    plt.scatter(data_pos[:, 0], data_pos[:, 1], color='r')
    plt.scatter(data_neg[:, 0], data_neg[:, 1], color='b')
    x = data[:, :-1]
    y = data[:, -1:]
    svm = SupportVectorMachineClassify(x.T, y.T, k_type='rbf')
    weight, bias = svm.support_vector_machine()
    print(svm.alpha)
    support_vector = svm.get_support_vector()
   
    # print the classify hyperplane only when the type is liner(Actually,I have no idea how to print hyperplane when the param is rbf krenel)
    # plt.plot([-1, 1], [liner_classify_func(weight, bias, -1), liner_classify_func(weight, bias, 1)], color='green')
    plt.scatter(support_vector.T[:, 0], support_vector.T[:, 1], color='green') #lighten up support vector
    plt.xlabel("x1")
    plt.ylabel("y2")
    plt.title("SVM result")
    plt.show()

