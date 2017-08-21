import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


class BinaryEvaluation:
    """
    This class provides some methods to evaluate binary classification. All the samples should be in probability format,
    which is available to generate PR-curve, ROC-curve and so on.In this class, dataframe is the input data, column
    label is the labels and column result, which is the predicted probability,of samples.
    See more information about binary evaluation:
    1.https://en.wikipedia.org/wiki/Precision_and_recall
    2.https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """
    def __init__(self):
        self.file1 = "./data/data1.csv"
        # self.file2 = "./data/data2.csv"

    def probability_distribution(self, df):
        """
        Present the distribution of both positive and negative labels
        :param df: input data
        :return:fig show
        """
        pos_sample = df[(df['label'] == 1.0)]
        neg_sample = df[(df['label'] == 0.0)]
        # Draw the fig
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        # Draw the two histograms
        ax1.hist(pos_sample['result'], bins=100, alpha=0.3, range=[0, 1], color='r')
        ax2.hist(neg_sample['result'], bins=100, alpha=0.3, range=[0, 1], color='b')

        plt.title("Probability distribution histogram")
        plt.grid(True)
        ax1.set_ylabel('Positive label count', color='r')
        ax2.set_ylabel('Negative label count', color='b')
        ax1.set_xlabel("Probability")
        plt.show()

    def pr_curve_present(self, df):
        """
        Present the PR-curve of the model
        :param df: input data
        :return: fig show
        """
        label = df['label']
        probability = df['result']
        precision, recall, thresholds = metrics.precision_recall_curve(y_true=label, probas_pred=probability, pos_label=1)
        plt.plot(recall, precision, color='r')
        plt.title("P-R curve")
        plt.grid(True)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.show()

    def p_r_sep_curve(self, df):
        """
        Present separate curve of precision and recall
        Personally suspicious metric of model. :-)
        :param df: input data
        :return: fig show
        """
        label = df['label']
        probability = df['result']

        precision, recall, thresholds = metrics.precision_recall_curve(y_true=label, probas_pred=probability, pos_label=1)
        length = len(thresholds)
        new_p = precision[0:length]
        new_r = recall[0:length]

        plt.plot(thresholds, new_p, color='r')
        plt.plot(thresholds, new_r, color='b')

        plt.legend(['precision', 'recall'], loc='lower left')
        plt.title("P-R curve")
        plt.grid(True)
        plt.show()

    def roc_curve(self, df):
        """
        Present the ROC-curve and calculate AUC value
        :param data_frame:
        :return:
        """
        label = df['label']
        probability = df['result']
        fpr, tpr, threshold = metrics.roc_curve(y_true=label, y_score=probability, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.plot(fpr, tpr, color='r', label="AUC = "+str('%.3f'% auc))
        plt.title("ROC-curve")
        plt.grid(True)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc='lower right')
        plt.show()

    def roc_curve_compare(self, df1, df2):
        """
        Comparision of two models' ROC curve
        :param df1: dataframe 1
        :param df2: dataframe 2
        :return: fig show
        """
        label_1 = df1['label']
        probability_1 = df1['result']
        fpr_1, tpr_1, threshold_1 = metrics.roc_curve(y_true=label_1, y_score=probability_1, pos_label=1)
        auc_1 = metrics.auc(fpr_1, tpr_1)
        plt.plot(fpr_1, tpr_1, color='r', label="model 1, AUC = "+str('%.3f'% auc_1))

        label_2 = df2['label']
        probability_2 = df2['result']
        fpr_2, tpr_2, threshold_2 = metrics.roc_curve(y_true=label_2, y_score=probability_2, pos_label=1)
        auc_2 = metrics.auc(fpr_2, tpr_2)
        plt.plot(fpr_2, tpr_2, color='b', label="model 2, AUC = "+str('%.3f'% auc_2))

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.title("ROC-curve")
        plt.grid(True)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc='lower right')
        plt.show()


if __name__ == '__main__':
    binary_evaluation = BinaryEvaluation()

    # import the data, encoding can be "ISO-8859-1", "utf-8" or "gbk" to fit Chinese characters
    data1 = pd.read_csv(binary_evaluation.file1, encoding="utf-8")
    df1 = data1.loc[:, ['result', 'label']]
    
    # data2 = pd.read_csv(binary_evaluation.file2, encoding="utf-8")
    # df2 = data2.loc[:, ['result', 'label']]

    # evaluation process
    binary_evaluation.probability_distribution(df1)
    binary_evaluation.pr_curve_present(df1)
    binary_evaluation.p_r_sep_curve(df1)
    binary_evaluation.roc_curve(df1)
    # binary_evaluation.roc_curve_compare(df1, df2)
