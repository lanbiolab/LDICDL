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
    将打分矩阵进行排序，并把对应的关系矩阵同步变化
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

def read_csv_matrix(fileName):
    """
    读取csv格式保存的矩阵
    :param fileName:文件路径
    :return: 返回一个dataframe
    """
    df = pd.read_csv(fileName)
    index = np.array(df['Unnamed: 0'])  #用于设置的索引列表必须转成numpy数组
    del df['Unnamed: 0']  #删除第一列名字
    df2 = df.set_index(index)  #重新设置索引
    return df2

def BiRW(ks,kk,ss):
    '''
    bi-random walk
    :param ks: 激酶-底物关系矩阵
    :param kk: 激酶相似性矩阵
    :param ss: 底物相似性矩阵
    :return:
    '''
    ks = np.mat(ks).transpose()
    alpha = 0.3
    l = 2
    r = 2

    # 矩阵每行的和
    row_sum_kk = kk.sum(axis=1)
    row_sum_ss = ss.sum(axis=1)

    # Reachable probability matrix
    norm_kk = np.zeros(kk.shape)
    norm_ss = np.zeros(ss.shape)
    for i in range(kk.shape[0]):
        for j in range(kk.shape[1]):
            norm_kk[i,j] = kk[i,j]/row_sum_kk[i] # normalized the above two matrixes along the row vector
    for i in range(ss.shape[0]):
        for j in range(ss.shape[1]):
            norm_ss[i,j] = ss[i,j]/row_sum_ss[i]

    T_kk = np.mat(norm_kk)
    T_ss = np.mat(norm_ss)

    R0 = ks/np.sum(ks.ravel())  #R0维度为（724,216）
    Rt = R0
    #bi-random walk
    for t in range(max(l,r)):
        ftl = 0
        ftr = 0
        if t<=l:
            nRtleft = alpha * T_ss*Rt + (1-alpha)*R0
            ftl = 1
        if t<=r:
            nRtright = alpha * Rt * T_kk + (1-alpha)*R0
            ftr = 1
        Rt =  (ftl*nRtleft + ftr*nRtright)/(ftl + ftr)
    return Rt


#程序运行例子
if __name__ == '__main__':
    #读取数据
    ks = read_csv_matrix('E:\kinase_alinged\PhosD\kinase_substrate_phos_interaction_matrix.csv').as_matrix()
    kk = read_csv_matrix('E:\kinase_alinged\激酶和底物的所有的相似性矩阵\kinase_conf_score_matrix.csv').as_matrix()
    ss = read_csv_matrix('E:\kinase_alinged\激酶和底物的所有的相似性矩阵\substrate_conf_score_matrix.csv').as_matrix()

    #bi-random walk算法
    score_matrix = BiRW(ks,kk,ss)
    print(score_matrix[:,0])
