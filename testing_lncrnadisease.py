
import numpy as np
import h5py
from scipy import stats
import auto_fun as auto
import pandas


def testing_lncrnadisease():
     a=1
     return a


def protect_lncrnadiseas(lncdis_tr,lncdis_val,i):
     alpha=100

     #load training and validation subset

     rating_tr = lncdis_tr
     aaa=rating_tr.shape[0]
     bbb=rating_tr.shape[1]


     rating_val =lncdis_val



     #load u and v
     with h5py.File('u_40_lncdis_40+100_auto.h5', 'r') as hf:

            u_lncrna = hf['u'][:]
            ccc=u_lncrna.shape[0]
            ddd=u_lncrna.shape[1]
     """   
     with h5py.File('v_40_lncdis_40+100_auto.h5', 'r') as hf:

           v = hf['v'][:]
    """
     with h5py.File('v_40_lncdis_40+100_auto.h5', 'r') as hf:

            v_lncrna = hf['v'][:]
     with h5py.File('u_40_disease_40+100_auto.h5', 'r') as hf:

           u_disease = hf['u'][:]
     with h5py.File('v_40_disease_40+100_auto.h5', 'r') as hf:

           v_disease = hf['v'][:]

     #preference and confidence
     p=np.zeros(rating_tr.shape)
     p[rating_tr>0]=1
     c=np.zeros(rating_tr.shape)
     c=1+alpha*rating_tr

     print(np.linalg.norm(p-np.dot(u_lncrna,v_lncrna.T)))
     print(np.linalg.norm(p-np.dot(u_disease,v_disease.T)))


     #print(p-np.dot(u,v.T))

     #only retain the new choices
     rating_val[rating_tr>0]=0

     r_pred_lncrna=np.dot(u_lncrna,v_lncrna.T)
     r_pred_disease=np.dot(u_disease,v_disease.T)

     #set mask
     m=(p>0)
     score_lncrna=r_pred_lncrna
     score_disease=r_pred_disease
     print(score_lncrna.shape)
     print(score_disease.shape)
     fileName1 = str(i) + "times_score_lncrna.txt"
     file = open(fileName1, 'w')

     for g in score_lncrna:
          k = '\t'.join([str(j) for j in g])
          file.write(k + "\n")
     file.close()
     fileName2 = str(i) + "times_score_disease.txt"
     file = open(fileName2, 'w')

     for b in score_disease:
          k = '\t'.join([str(j) for j in b])
          file.write(k + "\n")
     file.close()
     score_finall=add(score_lncrna,score_disease)
     print(score_finall.shape)
     return score_finall

def predicte(r_pred,rating_val,m):
     rank=0
     total=0
     content_list=[]
     content_list2=[]
     content_list_temp=[]
     for i in range(rating_val.shape[0]):
          prod=rating_val[i]
          prod_predict=np.ma.masked_array(r_pred[i],mask=m[i])
          content_list_temp=prod_predict.tolist()
          list_zero = []
          aaa=len(list_zero)
          bbb=len(content_list_temp)
          for i in range(len(content_list_temp)):

              if content_list_temp[i] is None:
                   i_temp=i
                   list_zero.append(0)
              else:
                   list_zero.append(content_list_temp[i])

                   print(content_list_temp[i])
          content_list.append(list_zero)

          content_list2.append(prod_predict.tolist())

          if np.sum(prod)>0:
             for j in range(prod.size):
                  if prod[j]>0:
                     total=total+1
                     rank=rank+stats.percentileofscore(prod_predict[~prod_predict.mask],prod_predict[j])

     print(len(content_list))
     print(len(content_list[0]))
     file=open('prediction_result_lncrnadisease.txt','w')

     for i in content_list2:
          k='\t'.join([str(j) for j in i])
          file.write(k+"\n")
     file.close()
     print(total)
     ##print(100-rank/total)
     score_matrix=np.array(content_list)
     ttt=score_matrix.shape[0]
     zzz=score_matrix.shape[1]
     return score_matrix

def add(a, b):
     c = [[(a[i][j]+b[i][j])/2 for j in range(len(a[i]))] for i in range(len(a))]
     score_c=np.array(c)

     return score_c