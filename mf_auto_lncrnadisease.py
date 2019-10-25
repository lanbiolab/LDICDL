#hybride model of SDAE and MF. One hidden layer is used in the SDAE.

import h5py
import numpy as np
import random
import auto_fun as auto
import testing_lncrnadisease as testing


timecounts = [0]
def deeplearing_start(lncdis_tr,lncdis_val,i):
    lncrnadisease_tr=lncdis_tr
    lncrnadisease_val=lncdis_val
    with h5py.File('need_lncrna_disease_tr.h5', 'w') as hf:
         hf.create_dataset("rating",  data=lncrnadisease_tr)
    main1(denoise = True)
    main2(denoise = True)
    score_matrix=testing.protect_lncrnadiseas(lncrnadisease_tr,lncrnadisease_val,i)
    print("count time is:")
    print(timecounts[0])
    return score_matrix
def main1(denoise = True):
    INPUT_LAYER = 6066 #lncrna feature sizes
    HIDDEN_UNIT1 = 130
    HIDDEN_UNIT2 = 100
    LEARNING_RATE = 0.001/100
    EPOCH_NUM = 100
    #randomSeed = np.random.RandomState(42)
    mu, sigma = 0, 0.1
    l=100
    #alpha=100
    #l2_u=200
    #l2_v=200
    alpha=100
    l2_u=200 #lambda
    l2_v=200
    batch=60
    # batch = 60
    ratio_l=500
    ratio_u=1.0
    #allMatrix, xtrain, xval, xtest,lenList,accList = getData()
    diction = [('ind_empleado', 5), ('pais_residencia', 6066), ('sexo', 3), ('ind_nuevo', 2), ('indrel', 2), ('indrel_1mes', 4), ('tiprel_1mes', 4), ('indresi', 2), ('indext', 2), ('conyuemp', 3), ('canal_entrada', 158), ('indfall', 2), ('cod_prov', 53), ('ind_actividad_cliente', 2), ('segmento', 4), ('antiguedad_binned', 10), ('age_binned', 6066), ('renta_binned', 10)]
    lenList = []
    for tuppl in diction:
        val = tuppl[1]
        lenList.append(val)
    accList = []
    for i in range(len(lenList)):
        if i ==0:
            accList.append(lenList[i])
        else:
            accList.append(accList[i-1]+lenList[i])
    #read lncrna infor
    with h5py.File('need_lncrna_gene_micrna_go.h5', 'r') as hf:
        xtrain = hf['infor'][:]
    #read rating matrix
    with h5py.File('need_lncrna_disease_tr.h5', 'r') as hf:
        rating_mat = hf['rating'][:]

    W1,W2,b1,b2,c1,c2 = auto.initialization(INPUT_LAYER,HIDDEN_UNIT1,HIDDEN_UNIT2,mu,sigma)
    #define lncrna and disease matrices
    u=np.random.rand(rating_mat.shape[0],l)
    v=np.random.rand(rating_mat.shape[1],l)

    #define preference and confidence matrices
    p=np.zeros(rating_mat.shape)
    p[rating_mat>0]=1
    c=np.zeros(rating_mat.shape)
    c=1+alpha*rating_mat  #confidence matrices

    iteration=30

    print('start')
    for iterate in range(iteration):
        #update lncrna
        for i in range(rating_mat.shape[0]):
            c_diag=np.diag(c[i,:])
            temp_u=np.dot(np.dot(p[i,:],c_diag),v)
            u[i,:]=np.dot(temp_u,np.linalg.pinv(l2_u*np.identity(l)+np.dot(np.dot(v.T,c_diag),v)))
        print('u complete')

        #update disease
        for j in range(rating_mat.shape[1]):
            #print(j)
            c_diag=np.diag(c[:,j])
            temp_v=np.dot(np.dot(p[:,j],c_diag),u)
            v[j,:]=np.dot(temp_v,np.linalg.pinv(l2_v*np.identity(l)+np.dot(np.dot(u.T,c_diag),u)))
        print('v complete')
        print(np.linalg.norm(p-np.dot(u,v.T)))
        timecounts[0] += 1
       # W1,b1,c1 = auto.autoEncoder_mono(ratio_l,ratio_u,batch,W1,xtrain,u,b1,c1,accList,EPOCH_NUM,LEARNING_RATE,denoise = True)
        #autoEncoder(ratio_l,ratio_u,batch,W1,W2,xtrain,u,b1,b2,c1,c2,accList,EPOCH_NUM,LEARNING_RATE,denoise = True):
        hiddenlayer3 = True
        W1, W2, b1, b2, c1, c2 = auto.autoEncoder(ratio_l, ratio_u, batch, W1, W2, xtrain, u, b1, b2, c1, c2, accList,
                                                  EPOCH_NUM,
                                                  LEARNING_RATE, denoise=True)
        # getoutPut(W1,W2,b1,b2,x,accList):
        hidden = auto.getoutPut(W1, W2, b1, b2, xtrain, accList)
        u=hidden
        print(np.linalg.norm(p-np.dot(u,v.T)))

    with h5py.File('u_40_lncdis_40+100_auto.h5', 'w') as hf:
        hf.create_dataset("u",  data=u)
    with h5py.File('v_40_lncdis_40+100_auto.h5', 'w') as hf:
        hf.create_dataset("v",  data=v)
    with h5py.File('W1_40_lncdis_40+100.h5', 'w') as hf:
        hf.create_dataset("W1",  data=W1)
    with h5py.File('b1_40_lncdis_40+100.h5', 'w') as hf:
        hf.create_dataset("b1",  data=b1)
    with h5py.File('c1_40_lncdis_40+100.h5', 'w') as hf:
        hf.create_dataset("c1",  data=c1)
    if hiddenlayer3:
        with h5py.File('W2_40_disease_40+100.h5', 'w') as hf:
            hf.create_dataset("W2", data=W2)
        with h5py.File('b2_40_disease_40+100.h5', 'w') as hf:
            hf.create_dataset("b2", data=b2)

    return hidden

def main2(denoise = True):
    INPUT_LAYER = 10621 # disease feature sizes
    HIDDEN_UNIT1 = 130
    HIDDEN_UNIT2 = 100
    LEARNING_RATE = 0.001/100
    EPOCH_NUM = 100
    #randomSeed = np.random.RandomState(42)
    mu, sigma = 0, 0.1
    l=100
    alpha=100
    l2_u=200
    l2_v=200
    # batch=60
    batch = 60
    ratio_l=500
    ratio_u=1.0
    #allMatrix, xtrain, xval, xtest,lenList,accList = getData()
    diction = [('ind_empleado', 5), ('pais_residencia', 10621), ('sexo', 3), ('ind_nuevo', 2), ('indrel', 2), ('indrel_1mes', 4), ('tiprel_1mes', 4), ('indresi', 2), ('indext', 2), ('conyuemp', 3), ('canal_entrada', 158), ('indfall', 2), ('cod_prov', 53), ('ind_actividad_cliente', 2), ('segmento', 4), ('antiguedad_binned', 10), ('age_binned', 10621), ('renta_binned', 10)]
    lenList = []
    for tuppl in diction:
        val = tuppl[1]
        lenList.append(val)
    accList = []
    for i in range(len(lenList)):
        if i ==0:
            accList.append(lenList[i])
        else:
            accList.append(accList[i-1]+lenList[i])
    with h5py.File('need_disease_micrna_gene.h5', 'r') as hf:
        xtrain = hf['infor'][:]
        xtrain = xtrain
    #read rating matrix
    with h5py.File('need_lncrna_disease_tr.h5', 'r') as hf:
        rating_mat = hf['rating'][:]
        rating_mat =  rating_mat.transpose()

    W1,W2,b1,b2,c1,c2 = auto.initialization(INPUT_LAYER,HIDDEN_UNIT1,HIDDEN_UNIT2,mu,sigma)
    u=np.random.rand(rating_mat.shape[0],l)
    v=np.random.rand(rating_mat.shape[1],l)

    #define preference and confidence matrices
    p=np.zeros(rating_mat.shape)
    p[rating_mat>0]=1
    c=np.zeros(rating_mat.shape)
    c=1+alpha*rating_mat  #confidence matrices

    iteration=30

    print('start')
    for iterate in range(iteration):
        #update
        for i in range(rating_mat.shape[0]):
            c_diag=np.diag(c[i,:])
            temp_u=np.dot(np.dot(p[i,:],c_diag),v)
            u[i,:]=np.dot(temp_u,np.linalg.pinv(l2_u*np.identity(l)+np.dot(np.dot(v.T,c_diag),v)))
        print('u complete')

        #update
        for j in range(rating_mat.shape[1]):
            #print(j)
            c_diag=np.diag(c[:,j])
            temp_v=np.dot(np.dot(p[:,j],c_diag),u)
            v[j,:]=np.dot(temp_v,np.linalg.pinv(l2_v*np.identity(l)+np.dot(np.dot(u.T,c_diag),u)))
        print('v complete')
        print(np.linalg.norm(p-np.dot(u,v.T)))
        timecounts[0] += 1

        hiddenlayer3 = True
        W1, W2, b1, b2, c1, c2 = auto.autoEncoder(ratio_l, ratio_u, batch, W1, W2, xtrain, u, b1, b2, c1, c2, accList, EPOCH_NUM,
                                      LEARNING_RATE, denoise=True)
        #getoutPut(W1,W2,b1,b2,x,accList):
        hidden = auto.getoutPut(W1, W2, b1, b2, xtrain, accList)
        u=hidden
        print(np.linalg.norm(p-np.dot(u,v.T)))

    with h5py.File('v_40_disease_40+100_auto.h5', 'w') as hf:
        hf.create_dataset("v",  data=u)
    with h5py.File('u_40_disease_40+100_auto.h5', 'w') as hf:
        hf.create_dataset("u",  data=v)
    with h5py.File('W1_40_disease_40+100.h5', 'w') as hf:
        hf.create_dataset("W1",  data=W1)
    with h5py.File('b1_40_disease_40+100.h5', 'w') as hf:
        hf.create_dataset("b1",  data=b1)
    with h5py.File('c1_40_disease_40+100.h5', 'w') as hf:
        hf.create_dataset("c1",  data=c1)

    if hiddenlayer3:
        with h5py.File('W2_40_disease_40+100.h5', 'w') as hf:
            hf.create_dataset("W2", data=W2)
        with h5py.File('b2_40_disease_40+100.h5', 'w') as hf:
            hf.create_dataset("b2", data=b2)

    return hidden


