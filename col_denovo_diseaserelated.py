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
import mf_auto_lncrnadisease as  mald



with h5py.File('need_lncrna_disease.h5', 'r') as hf:
      lncrna_disease_matrix = hf['rating'][:]
      lncrna_disease_matrix_val =  lncrna_disease_matrix.copy()

all_tpr = []
all_fpr = []

#pr曲线参数
all_recall = []
all_precision = []

for i in range(412):
    new_lncrna_disease_matrix = lncrna_disease_matrix.copy()
    roc_lncrna_disease_matrix = lncrna_disease_matrix.copy()
    #将第i行全部置为0
    if ((False in (new_lncrna_disease_matrix[:,i]==0))==False):
        continue
    new_lncrna_disease_matrix[:,i] = 0
    print(new_lncrna_disease_matrix.shape)

    score_matrix=  mald.deeplearing_start(new_lncrna_disease_matrix,lncrna_disease_matrix_val, i)
    #roc_lncrna_disease_matrix = np.array(roc_lncrna_disease_matrix.transpose())
    # sorted_kinase_substrate_matrix = BiRW.sort_matrix(score_matrix[i,:],roc_kinase_substrate_matrix[i,:])
    sort_index = np.argsort(-score_matrix[:,i],axis=0)  # 排序好的打分矩阵的索引
    sorted_lncrna_disease_row = roc_lncrna_disease_matrix[:,i][sort_index]

    fileName1 = str(i) + "times.txt"
    file = open(fileName1, 'w')

    for p in score_matrix:
        k = '\t'.join([str(j) for j in p])
        file.write(k + "\n")
    file.close()
    tpr_list = []
    fpr_list = []

    #pr曲线参数
    recall_list = []
    precision_list = []

    #每循环一次获得一次tpr和fpr的值
    for cutoff in range(1,241):  #每个阈值
        P_vector = sorted_lncrna_disease_row[0:cutoff]
        N_vector = sorted_lncrna_disease_row[cutoff:]
        TP = np.sum(P_vector == 1)
        FP = np.sum(P_vector == 0)
        TN = np.sum(N_vector == 0)
        FN = np.sum(N_vector == 1)
        tpr = TP/(TP+FN)
        fpr = FP/(FP+TN)
        # #fpr = FP/(FP+TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        #pr曲线参数
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        recall_list.append(recall)
        precision_list.append(precision)

    all_tpr.append(tpr_list)
    all_fpr.append(fpr_list)
    fileName2 = str(i) + "times_recall.txt"
    file = open(fileName2, 'w')
    for h in recall_list:
        file.write(str(h) + '\n')  # \r\n为换行符
    file.close()
    fileName3 = str(i) + "times_precision.txt"
    file = open(fileName3, 'w')
    for s in recall_list:
        file.write(str(s) + '\n')  # \r\n为换行符
    file.close()
    #pr曲线参数
    all_recall.append(recall_list)
    all_precision.append(precision_list)


tpr_arr = np.array(all_tpr)
fpr_arr = np.array(all_fpr)

#pr曲线参数
recall_arr = np.array(all_recall)
precision_arr = np.array(all_precision)

mean_denovo_recall = np.mean(recall_arr,axis=0)
mean_denovo_precision = np.mean(precision_arr,axis=0)
file=open('mean_denovo_recall_holl_diseaserelated.txt','w')
for i in mean_denovo_recall:
    file.write(str(i)+'\n')  #\r\n为换行符
file.close()
file=open('mean_denovo_precision_holl_dieaserelated.txt','w')
for i in mean_denovo_precision:
    file.write(str(i)+'\n')  #\r\n为换行符
file.close()

mean_denovo_tpr = np.mean(tpr_arr,axis=0)  # axis=0，计算每一列的均值
mean_denovo_fpr = np.mean(fpr_arr,axis=0)
file=open('mean_denovo_tpr_holl_diseaserelated.txt','w')
for i in mean_denovo_tpr:
    file.write(str(i)+'\n')  #\r\n为换行符
file.close()
file=open('mean_denovo_fpr_holl_dieaserelated.txt','w')
for i in mean_denovo_fpr:
    file.write(str(i)+'\n')  #\r\n为换行符
file.close()
roc_auc = metrics.auc(mean_fpr, mean_tpr)
roc_auc = np.trapz(mean_denovo_tpr,mean_denovo_fpr)
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
plt.plot(mean_denovo_fpr,mean_denovo_tpr, label='mean ROC=%0.3f'%roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()  #要加上这一句才能显示label
plt.show()

#改用PR曲线画图
pr_auc = metrics.auc(mean_denovo_recall,mean_denovo_precision)
plt.plot(mean_denovo_recall,mean_denovo_precision, label='mean PR_AUC=%0.3f'%pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()  #要加上这一句才能显示label
plt.savefig("pr1.png")
plt.show()