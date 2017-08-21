import numpy as np
import pandas as pd
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing.data import OneHotEncoder
from sklearn import metrics
from matplotlib import pyplot as plt


class GradientBoostingWithLogisticRegression:
    """
    This class reproduces the paper of Facebook which combine gbdt and logistic regression to solve CRT problem.
    See more details:
    1.He X, Pan J, Jin O, et al. Practical lessons from predicting clicks on ads at facebook[C].
        Proceedings of 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. ACM, 2014: 1-9.
    """
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
        enc = OneHotEncoder().fit(x_gen)
        x_gen_enc = enc.transform(x_gen)
        return x_gen_enc, enc

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
        :return: gbt model, lr model and one-hot encoder model
        """
        gbt, x_gen_train = self.gradient_boosting_stage()
        x_fin_train, enc = self.feature_assemble_train(x_gen_train)
        lr = self.logistic_regression_stage(x_fin_train, self.y_train)
        return gbt, lr, enc

    def test_stage(self, gbt, lr, enc):
        """
        Test stage
        :param gbt: gbt model
        :param lr: lr model
        :param enc: one-hot encoder model
        :return: ROC curve and AUC value
        """
        x_gen_test = gbt.apply(self.x_test)[:, :, 0]
        x_fin_test = enc.transform(x_gen_test)
        pro = lr.predict_proba(x_fin_test)
        self.roc_curve(pro[:, 1], self.y_test)
        return

