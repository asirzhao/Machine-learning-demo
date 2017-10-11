import numpy as np
import random
from matplotlib import pyplot as plt


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
    def __init__(self, x, y, k_type='liner', penalty=200, delta=1, tol=0.001, max_passes=25):
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
        self.x_arr = x
        self.y_arr = y
        self.k_type = k_type
        self.__penalty = penalty
        self.__delta = delta
        self.m = x.shape[1]
        self.kernel_unit = 0
        self.kernel_matrix = np.zeros((self.m, self.m))
        self.alpha = np.zeros((1, self.m))
        self.bias = 0
        self.tol = tol
        self.max_passes = max_passes
        self.l_value = 0
        self.h_value = 0
        self.step = 0.001

    def support_vector_machine(self):
        f_alpha, f_bias = self.__sequential_min_opt()
        f_weight = (f_alpha*self.y_arr).dot(self.x_arr.T)
        return f_weight, f_bias

    def get_support_vector(self):
        """
        Get support vector
        :return:
        """
        alpha = self.alpha
        sub_0 = np.where(alpha > 0)
        sub_p = np.where(alpha < self.__penalty)
        """both sub_0 and sub_p has two dimensionality, we cannot pick up the first dimensionality"""
        sub = np.intersect1d(sub_0[1], sub_p[1])
        return self.x_arr[:, sub]

    def __sequential_min_opt(self):
        """
        Sequential Minimal Optimization(SMO) function
        :return: Lagrange multipliers & bias
        """
        pass_value = 0
        while pass_value < self.max_passes:
            num_changed_alpha = 0
            for i in range(self.m):
                error_value_i = self.__error_func(i)
                s_i = self.y_arr[0, i] * error_value_i
                if (s_i < -self.tol and self.alpha[0, i] < self.__penalty) or (s_i > self.tol and self.alpha[0, i] > 0):
                    j = self.__random_process(i)
                    error_value_j = self.__error_func(j)
                    alpha_old_i = self.alpha[0, i]
                    alpha_old_j = self.alpha[0, j]
                    self.l_value, self.h_value = self.__calculate_l_h(i, j)
                    if self.l_value == self.h_value:
                        continue
                    eta = 2*self.x_arr[:, i].T.dot(self.x_arr[:, j])-self.x_arr[:, i].T.dot(self.x_arr[:, i])-self.x_arr[:, j].T.dot(self.x_arr[:, j])
                    if eta >= 0:
                        continue
                    alpha_new_j = self.alpha[0, j]-self.y_arr[0, j]*(error_value_i-error_value_j)/eta
                    self.alpha[0, j] = self.__calculate_alpha2(alpha_new_j)
                    if abs(alpha_old_j-self.alpha[0, j]) < self.step:
                        continue
                    self.alpha[0, i] = alpha_old_i+self.y_arr[0, i]*self.y_arr[0, j]*(alpha_old_j-self.alpha[0, j])
                    self.bias = self.__calculate_bias(error_value_i, error_value_j, alpha_old_i, alpha_old_j, i, j)
                    num_changed_alpha += 1
                    print(alpha_old_i, self.alpha[0, i], alpha_old_j, self.alpha[0, j])
            if num_changed_alpha == 0:
                pass_value += 1
            else:
                pass_value = 0
        return self.alpha, self.bias

    def __calculate_bias(self, error_value_i, error_value_j, alpha_old_i, alpha_old_j, i, j):
        b1 = self.bias-error_value_i-self.y_arr[0, i]*(self.alpha[0, i]-alpha_old_i)*(self.x_arr[:, i].T.dot(self.x_arr[:, i]))-self.y_arr[0, j]*(self.alpha[0, j]-alpha_old_j)*(self.x_arr[:, i].T.dot(self.x_arr[:, j]))
        b2 = self.bias-error_value_j-self.y_arr[0, i]*(self.alpha[0, i]-alpha_old_i)*(self.x_arr[:, i].T.dot(self.x_arr[:, j]))-self.y_arr[0, j]*(self.alpha[0, j]-alpha_old_j)*(self.x_arr[:, j].T.dot(self.x_arr[:, j]))
        if 0 < self.alpha[0, i] < self.__penalty:
            return b1
        elif 0 < self.alpha[0, j] < self.__penalty:
            return b2
        else:
            return 0.5 * (b1 + b2)

    def __calculate_alpha2(self, alpha_new):
        if alpha_new > self.h_value:
            alpha = self.h_value
        elif alpha_new < self.l_value:
            alpha = self.l_value
        else:
            alpha = alpha_new
        return alpha

    def __calculate_l_h(self, i, j):
        if self.y_arr[0, i] == self.y_arr[0, j]:
            l_value = max(0, self.alpha[0, i] + self.alpha[0, j] - self.__penalty)
            h_value = min(self.__penalty, self.alpha[0, i] + self.alpha[0, j])
        else:
            l_value = max(0, self.alpha[0, j] - self.alpha[0, i])
            h_value = min(self.__penalty, self.__penalty + self.alpha[0, j] - self.alpha[0, i])
        return l_value, h_value

    def __random_process(self, i):
        """
        Generate random number j, which is not equal to input number i
        :param i: input number i
        :return: random number j
        """
        j = random.randint(0, self.m-1)
        while j == i:
            j = random.randint(0, self.m - 1)
        return j

    def __kernel_func(self, x_array, z_array, k_type):
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
            self.kernel_unit = np.exp(min_a_z.dot(min_a_z.T) / (-2.0 * self.__delta ** 2))
        elif k_type == "liner":
            self.kernel_unit = x_array.dot(z_array.T)
        else:
            raise Exception("The kernel type %s is not supported so far..." % k_type)
        return self.kernel_unit

    def __kernel_matrix_gen(self, k_type):
        """
        Kernel matrix generator. Shape of matrix is m*m, m represents amount of training samples
        :param k_type: kernel type
        :return: kernel matrix
        """
        for i in range(self.m):
            for j in range(self.m):
                self.kernel_matrix[i, j] = self.__kernel_func(self.x_arr[:, i], self.x_arr[:, j], k_type)
        return self.kernel_matrix

    def __error_func(self, i):
        """
        ith iterator error of SVM
        :param i: iterator time
        :return: error value
        """
        g_xi = (self.alpha * self.y_arr).dot(self.__kernel_matrix_gen(self.k_type)[:, i]) + self.bias
        error = g_xi - self.y_arr[0, i]
        return error


