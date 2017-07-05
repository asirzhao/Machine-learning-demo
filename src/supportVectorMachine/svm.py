import numpy as np
import random
from matplotlib import pyplot as plt


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


class SupportVectorMachineClassify:
    """
    Class of Support Vector Machine
        SVM is a classic machine learning algorithm, liner and RBF kernel are provided in this demo. This demo is total
     based on SMO algorithm, which is a good method to solve QP problem.
        See more information about SVM and SMO:
        1.https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
        2.http://cs229.stanford.edu/notes/cs229-notes3.pdf
        3.http://cs229.stanford.edu/materials/smo.pdf
        4.https://en.wikipedia.org/wiki/Support_vector_machine
    """
    def __init__(self, x_array, y_array):
        """
        _penalty: penalty parameter C
        _delta: parameter of delta in Gaussian kernel
        m: length of x_array
        kernel_unit: kernel unit value
        kernel_matrix: kernel matrix
        bias: bias of SVM
        tol: tolerance parameter to determine continue or break the loop
        max_passes: times of alpha without changing in iterator
        l_value: L value of SMO
        h_value: H value of SMO
        k_type: kernel type
        step: threshold parameter of alpha in two iterators
        """
        self.x_arr = x_array
        self.y_arr = y_array
        self._penalty = 200
        self._delta = 1
        self.m = x_array.shape[1]
        self.kernel_unit = 0
        self.kernel_matrix = np.zeros((self.m, self.m))
        self.alpha = np.zeros((1, self.m))
        self.bias = 0
        self.tol = 0.001
        self.max_passes = 25
        self.l_value = 0
        self.h_value = 0
        self.k_type = "liner"
        self.step = 0.0001

    def support_vector_machine(self):
        f_alpha, f_bias = self.sequential_min_opt()
        f_weight = (f_alpha*self.y_arr).dot(self.x_arr.T)
        return f_weight, f_bias

    def get_support_vector(self):
        """
        Get support vector
        :return:
        """
        alpha = self.alpha
        sub_0 = np.where(alpha > 0)
        sub_p = np.where(alpha < self._penalty)
        """both sub_0 and sub_p has two dimensionality, we cannot pick up the first dimensionality"""
        sub = np.intersect1d(sub_0[1], sub_p[1])
        return self.x_arr[:, sub]

    def sequential_min_opt(self):
        """
        Sequential Minimal Optimization(SMO) function
        :return: Lagrange multipliers & bias
        """
        pass_value = 0
        while pass_value < self.max_passes:
            num_changed_alpha = 0
            for i in range(self.m):
                error_value_i = self.error_func(i)
                s_i = self.y_arr[0, i] * error_value_i
                if (s_i < -self.tol and self.alpha[0, i] < self._penalty) or (s_i > self.tol and self.alpha[0, i] > 0):
                    j = self.random_process(i)
                    error_value_j = self.error_func(j)
                    alpha_old_i = self.alpha[0, i]
                    alpha_old_j = self.alpha[0, j]
                    self.l_value, self.h_value = self.calculate_l_h(i, j)
                    if self.l_value == self.h_value:
                        continue
                    eta = 2*self.x_arr[:, i].T.dot(self.x_arr[:, j])-self.x_arr[:, i].T.dot(self.x_arr[:, i])-self.x_arr[:, j].T.dot(self.x_arr[:, j])
                    if eta >= 0:
                        continue
                    alpha_new_j = self.alpha[0, j]-self.y_arr[0, j]*(error_value_i-error_value_j)/eta
                    self.alpha[0, j] = self.calculate_alpha2(alpha_new_j)
                    if abs(alpha_old_j-self.alpha[0, j]) < self.step:
                        continue
                    self.alpha[0, i] = alpha_old_i+self.y_arr[0, i]*self.y_arr[0, j]*(alpha_old_j-self.alpha[0, j])
                    self.bias = self.calculate_bias(error_value_i, error_value_j, alpha_old_i, alpha_old_j, i, j)
                    num_changed_alpha += 1
                    print(alpha_old_i, self.alpha[0, i], alpha_old_j, self.alpha[0, j])
            if num_changed_alpha == 0:
                pass_value += 1
            else:
                pass_value = 0
        return self.alpha, self.bias

    def calculate_bias(self, error_value_i, error_value_j, alpha_old_i, alpha_old_j, i, j):
        b1 = self.bias-error_value_i-self.y_arr[0, i]*(self.alpha[0, i]-alpha_old_i)*(self.x_arr[:, i].T.dot(self.x_arr[:, i]))-self.y_arr[0, j]*(self.alpha[0, j]-alpha_old_j)*(self.x_arr[:, i].T.dot(self.x_arr[:, j]))
        b2 = self.bias-error_value_j-self.y_arr[0, i]*(self.alpha[0, i]-alpha_old_i)*(self.x_arr[:, i].T.dot(self.x_arr[:, j]))-self.y_arr[0, j]*(self.alpha[0, j]-alpha_old_j)*(self.x_arr[:, j].T.dot(self.x_arr[:, j]))
        if 0 < self.alpha[0, i] < self._penalty:
            return b1
        elif 0 < self.alpha[0, j] < self._penalty:
            return b2
        else:
            return 0.5 * (b1 + b2)

    def calculate_alpha2(self, alpha_new):
        if alpha_new > self.h_value:
            alpha = self.h_value
        elif alpha_new < self.l_value:
            alpha = self.l_value
        else:
            alpha = alpha_new
        return alpha

    def calculate_l_h(self, i, j):
        if self.y_arr[0, i] == self.y_arr[0, j]:
            l_value = max(0, self.alpha[0, i] + self.alpha[0, j] - self._penalty)
            h_value = min(self._penalty, self.alpha[0, i] + self.alpha[0, j])
        else:
            l_value = max(0, self.alpha[0, j] - self.alpha[0, i])
            h_value = min(self._penalty, self._penalty + self.alpha[0, j] - self.alpha[0, i])
        return l_value, h_value

    def random_process(self, i):
        """
        Generate random number j, which is not equal to input number i
        :param i: input number i
        :return: random number j
        """
        j = random.randint(0, self.m-1)
        while j == i:
            j = random.randint(0, self.m - 1)
        return j

    def kernel_func(self, x_array, z_array, k_type):
        """
        Kernel unit value function.
        Function support liner and rbf kernel so far
        :param x_array: x
        :param z_array: z
        :param k_type: kernel type
        :return: kernel matrix unit value
        """
        if k_type == "rbf":
            min_a_z = x_array - z_array
            self.kernel_unit = np.exp(min_a_z.dot(min_a_z.T) / (-2.0 * self._delta ** 2))
        elif k_type == "liner":
            self.kernel_unit = x_array.dot(z_array.T)
        else:
            raise Exception("The kernel type %s is not supported so far..." % k_type)
        return self.kernel_unit

    def kernel_matrix_gen(self, k_type):
        """
        Kernel matrix generator. Shape of matrix is m*m, m represents amount of training samples
        :param k_type: kernel type
        :return: kernel matrix
        """
        for i in range(self.m):
            for j in range(self.m):
                self.kernel_matrix[i, j] = self.kernel_func(self.x_arr[:, i], self.x_arr[:, j], k_type)
        return self.kernel_matrix

    def error_func(self, i):
        """
        ith iterator error of SVM
        :param i: iterator time
        :return: error value
        """
        g_xi = (self.alpha * self.y_arr).dot(self.kernel_matrix_gen(self.k_type)[:, i]) + self.bias
        error = g_xi - self.y_arr[0, i]
        return error

if __name__ == '__main__':
    data = np.genfromtxt("./svm.csv", delimiter=',')
    data_new = z_score_nor_fuc(data[:, :-1])
    data[:, :-1] = data_new

    data_pos = data[np.where(data[:, -1] == 1)]
    data_neg = data[np.where(data[:, -1] == -1)]

    plt.scatter(data_pos[:, 0], data_pos[:, 1], color='r')
    plt.scatter(data_neg[:, 0], data_neg[:, 1], color='b')
    x = data[:, :-1]
    y = data[:, -1:]
    svm = SupportVectorMachineClassify(x.T, y.T)
    weight, bias = svm.support_vector_machine()
    print(svm.alpha)
    support_vector = svm.get_support_vector()
    
    plt.plot([-1, 1], [liner_classify_func(weight, bias, -1), liner_classify_func(weight, bias, 1)], color='green')
    plt.scatter(support_vector.T[:, 0], support_vector.T[:, 1], color='green') #lighten up support vector
    plt.xlabel("x1")
    plt.ylabel("y2")
    plt.title("SVM result")
    plt.show()

