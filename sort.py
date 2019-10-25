import pickle
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import math


def sort_matrix(score_matrix,interact_matrix):
    """
    sort the score matrix, and change the association matrix correspondingly.
    :param scor_matrix:
    :param interact_matrix:
    :return:
    """
    sort_index = np.argsort(-score_matrix,axis=0)  # 排序好的打分矩阵的索引
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:,i] = score_matrix[:,i][sort_index[:,i]]
        y_sorted[:,i] = interact_matrix[:,i][sort_index[:,i]]
    return y_sorted,score_sorted

