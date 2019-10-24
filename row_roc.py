import pickle
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import math
import BiRW
import h5py
import mf_auto_lncrnadisease as mald
import time


with h5py.File('need_lncrna_disease.h5', 'r') as hf:
      lncrna_disease_matrix = hf['rating'][:]
      lncrna_disease_matrix_val =  lncrna_disease_matrix.copy()
index_tuple = (np.where( lncrna_disease_matrix == 1))
one_list = list(zip(index_tuple[0],index_tuple[1]))
random.shuffle(one_list)
split = math.ceil(len(one_list)/10)
all_tpr = []
all_fpr = []
#pr曲线参数
all_recall = []
all_precision = []
#将一个大list划分为多个小list
for i in range(0,len(one_list),split):

    train_index = one_list[i:i+split]
    new_lncrna_disease_matrix =  lncrna_disease_matrix.copy()

    for index in train_index:
        new_lncrna_disease_matrix[index[0],index[1]] = 0  #将测试集置0
        # if new_kinase_substrate_matrix[index[0],:].sum() == 0:
        #     new_kinase_substrate_matrix[index[0],index[1]] = 1
    train_matrix_file = str(i) + "times_need_lncrna_disease_tr.h5"
    with h5py.File(train_matrix_file, 'w') as hf:
         hf.create_dataset("rating",  data=new_lncrna_disease_matrix)
    roc_lncrna_disease_matrix = new_lncrna_disease_matrix + lncrna_disease_matrix
    score_matrix=  mald.deeplearing_start(new_lncrna_disease_matrix,lncrna_disease_matrix_val,i)
    aa=score_matrix.shape
    ###score_matrix = BiRW.BiRW(new_lncrna_disease_matrix,kk,ss)
    #转置
   ### roc_lncrna_disease_matrix = roc_lncrna_disease_matrix.transpose()
    bb=roc_lncrna_disease_matrix.shape
    zero_matrix = np.zeros((score_matrix.shape[0], score_matrix.shape[1])).astype('int64')
    print(score_matrix.shape)
    print(roc_lncrna_disease_matrix.shape)
    #把原来没变的1对应的关系矩阵和打分矩阵的元素变为0

    score_matrix_temp=score_matrix.copy()
    score_matrix=score_matrix_temp+zero_matrix
    minvalue=np.min(score_matrix)
    score_matrix[np.where(roc_lncrna_disease_matrix == 2)] = minvalue-10
    fileName1 = str(i) + "times.txt"
    file=open(fileName1,'w')

    for i in score_matrix:
          k='\t'.join([str(j) for j in i])
          file.write(k+"\n")
    file.close()

    print(score_matrix.shape)
    print(roc_lncrna_disease_matrix.shape)
    sorted_lncrna_disease_matrix,sorted_score_Matrix = BiRW.sort_matrix(score_matrix,roc_lncrna_disease_matrix)

   #tpr,fpr
    tpr_list = []
    fpr_list = []
    # pr曲线参数
    recall_list = []
    precision_list = []
    #每循环一次获得一次tpr和fpr的值
    for cutoff in range(sorted_lncrna_disease_matrix.shape[0]):  #每个阈值
        P_matrix = sorted_lncrna_disease_matrix[0:cutoff+1,:]
        N_matrix = sorted_lncrna_disease_matrix[cutoff+1:sorted_lncrna_disease_matrix .shape[0]+1,:]
        # print(P_matrix.shape)
        # print(N_matrix.shape)
        TP = np.sum(P_matrix == 1)
        FP = np.sum(P_matrix == 0)
        TN = np.sum(N_matrix == 0)
        FN = np.sum(N_matrix == 1)
        tpr = TP/(TP+FN)
        fpr = FP/(FP+TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

        # pr曲线参数
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        recall_list.append(recall)
        precision_list.append(precision)
    all_tpr.append(tpr_list)
    all_fpr.append(fpr_list)
    #pr曲线参数
    all_recall.append(recall_list)
    all_precision.append(precision_list)

tpr_arr = np.array(all_tpr)
fpr_arr = np.array(all_fpr)
#pr曲线参数
recall_arr = np.array(all_recall)
precision_arr = np.array(all_precision)

mean_tpr = np.mean(tpr_arr,axis=0)  # axis=0，计算每一列的均值
mean_fpr = np.mean(fpr_arr,axis=0)

file=open('mean_tpr_cross_lai.txt','w')
for i in mean_tpr:
      file.write(str(i)+"\n")
file.close()

file=open('mean_fpr_cross_lai.txt','w')
for i in mean_fpr:
      file.write(str(i)+"\n")
file.close()

roc_auc = metrics.auc(mean_fpr,mean_tpr)
plt.plot(mean_fpr,mean_tpr, label='mean ROC=%0.3f'%roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 0)  #要加上这一句才能显示label
plt.savefig("roc3.png")
print("runtime over, now is :")
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
plt.show()

#pr曲线
