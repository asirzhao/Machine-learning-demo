import numpy as np
import pandas as pd
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing.data import OneHotEncoder
from sklearn import metrics
from matplotlib import pyplot as plt


class GradientBoostingWithLogisticRegression:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def gradient_boosting_stage(self):
        """
        Gradient Boosting stage
        :return: gbt model and generated new features
        """
        gbt = GradientBoostingClassifier(n_estimators=50, max_depth=5)
        gbt.fit(self.x_train, self.y_train)
        return gbt, gbt.apply(self.x_train)[:, :, 0]

    def logistic_regression_stage(self, x_array, y_array):
        """
        Logistic stage
        :param x_array: input x_array
        :param y_array: input y_array
        :return: logistic regression model
        """
        lr = LogisticRegression()
        lr.fit(x_array, y_array)
        return lr

    def feature_assemble_train(self, x_gen):
        """
        Assemble features, with one-hot encoding
        :param x_gen: generated x_array
        :return: assembled features
        """
        enc = OneHotEncoder()
        x_gen_enc = enc.fit_transform(x_gen)
        return x_gen_enc

    # def feature_assemble_test(self, x_gen):


    def roc_curve(self, pro, label):
        """
        present ROC curve and AUC value
        :return: ROC curve and AUC value
        """
        fpr, tpr, threshold = metrics.roc_curve(y_true=label, y_score=pro, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.plot(fpr, tpr, color='r', label="AUC = " + str(auc))
        plt.title("ROC-curve")
        plt.grid(True)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc='lower right')
        plt.show()
        return

    def train_stage(self):
        """
        Train stage
        :return: gbt model and lr model
        """
        gbt, x_gen_train = self.gradient_boosting_stage()
        feature_arr_max = []
        feature_arr_min = []

        # 保存每个feature的最大和最小value
        for i in range(x_gen_train.shape[1]):
            feature_arr_max.append(max(x_gen_train[:, i]))
            feature_arr_min.append(min(x_gen_train[:, i]))
            # print(min(x_gen_train[:, i]))
        # print(x_gen_train[0, :])
        # tmp = list(x_gen_train[:, 1])
        # tmp = list(set(tmp))
        # tmp.sort()
        # print(tuple(tmp))
        x_fin_train = self.feature_assemble_train(x_gen_train)
        lr = self.logistic_regression_stage(x_fin_train, self.y_train)
        return gbt, lr

    def test_stage(self, gbt, lr):
        """
        Test stage
        :param gbt: gbt model
        :param lr: lr model
        :return: ROC curve and AUC value
        """
        x_gen_test = gbt.apply(self.x_test)[:, :, 0]
        """"
        TODO:
        """
        # x_fin_test = self.feature_assemble(x_gen_test)
        pro = lr.predict_proba(x_fin_test)
        self.roc_curve(pro, self.y_test)
        return


if __name__ == '__main__':
    file_train = "./data/gbt_train.csv"
    file_test = "./data/gbt_test.csv"
    train = pd.read_csv(file_train, encoding="utf-8")
    test = pd.read_csv(file_test, encoding="utf-8")
    grad_with_log = GradientBoostingWithLogisticRegression(train.iloc[:, 2:-9], train.iloc[:, -2], test.iloc[:, 2:-9], test.iloc[:, -2])
    gbt, lr = grad_with_log.train_stage()
    # grad_with_log.test_stage(gbt, lr)
