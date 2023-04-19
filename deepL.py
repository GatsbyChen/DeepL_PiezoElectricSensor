#!/usr/bin/env python
# coding: utf-8

# In[10]:


import tensorflow.compat.v1 as tf 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from deepLSetting import DeepLSetting 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import keras.optimizers as opt
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from bayes_opt import BayesianOptimization

#csvを受け取って、トレーニングを行う。
def deepL_keras(csv: pd.DataFrame, dls: DeepLSetting, num_epoch, batch, plot=True, k_fold=False):
    #データの定義
    X = csv.iloc[:, 0:dls.num_featureValue]
    y = csv.iloc[:, dls.num_featureValue:len(csv.columns)]
    X_train = X[dls.trainRow[0]:dls.trainRow[1]]
    X_test = X[dls.trainRow[1]:len(X)]
    y_train = y[dls.trainRow[0]:dls.trainRow[1]]
    y_test = y[dls.trainRow[1]:len(X)]
    #k-foldがTrueなら、K分割交差検証を行う。
    if k_fold:
        print(k_foldCrossValidation(k_fold, X_train, y_train, dls.model, num_epoch, batch))
        return
   
    #データで訓練
    #model = Sequential()
    #model.add(Dense(512, input_shape=(19,)))
    #model.add(Dropout(0.2))
    #model.add(Dense(512))
    #model.add(Dropout(0.2))
    #model.add(Dense(2))
    #model.compile(loss='mean_squared_error', optimizer=opt.Adam(), metrics=["mae"])
    history = dls.model.fit(X_train, y_train, batch_size=batch, epochs=num_epoch, verbose=1)
    score = dls.model.evaluate(X_test, y_test, verbose=0)
    
    if plot:
        plt.plot(history.history['loss'][10:])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid()
        plt.legend(['Train'], loc='upper left')
        plt.show()
        print(f"テストデータのMSE: {score[0]*40000}, MAE:{score[1]*40000}")
        y_test = y_test*200
        print(f"トレーニングデータ{y_train*200}")
        print(dls.model.predict(X_train)*200)
        print(f"テストデータ{y_test}")
        print(dls.model.predict(X_test)*200)
        
    return -40000*score[0]
    
#K分割交差検証
def k_foldCrossValidation(k, X_train, y_train, model, num_epoch, batch):
    kf = KFold(n_splits=k, shuffle=True)
    all_loss=[]
    all_val_loss=[]
    all_acc=[]
    all_val_acc=[]
    for train_index, val_index in kf.split(X_train,y_train):
        print(X_train)
        train_data=X_train.iloc[train_index,:]
        train_label=y_train.iloc[train_index,:]
        val_data=X_train.iloc[val_index,:]
        val_label=y_train.iloc[val_index,:]

        history=model.fit(train_data,
                          train_label,
                          epochs=num_epoch,
                          batch_size=batch,
                          validation_data=(val_data,val_label))

        loss=history.history['loss']
        val_loss=history.history['val_loss']

        all_loss.append(loss)
        all_val_loss.append(val_loss)
        print(f"{val_index}のMSE: {val_loss[num_epoch-1]*40000}")
    ave_all_loss=[
        np.mean([x[i] for x in all_loss]) for i in range(num_epoch)]
    ave_all_val_loss=[
        np.mean([x[i] for x in all_val_loss]) for i in range(num_epoch)]
    return ave_all_val_loss

"""
#ベイズ最適化
def func(x,y,z):
    data = pd.read_csv("out6.csv")
    dls = DeepLSetting()
    dls.set_initial(19,2,[0,8])
    dls.set_modelLayerAndNode([19,int(x),int(y),2], dropout=z)
    dls.model_compile()
    return deepL_keras(data, dls, 3000, 4, plot=False)
"""
dls = DeepLSetting()
dls.set_initial(19,2,[0,8])
data = pd.read_csv("out6.csv")
#データを正規化
data["H"] = data["H"]/10
data["H/A'B'"] = data["H/A'B'"]/100
data["SBP"] = data["SBP"]/200
data["DBP"] = data["DBP"]/200
pbounds = {
        'num_layer': (3,7),
        'num_node': (1,1024),
        'dropout': (0.1,0.4),
        'batch': (1,30)}
dls.bayesOpt(data, pbounds, n_iter=100)
"""
dls = DeepLSetting()
dls.set_initial(19,2,[0,8])
dls.set_modelLayerAndNode([19,512,512,2], dropout=0.3)
dls.model_compile()
dls.model.summary()
data = pd.read_csv("out6.csv")
#データを正規化
data["H"] = data["H"]/10
data["H/A'B'"] = data["H/A'B'"]/100
data["SBP"] = data["SBP"]/200
data["DBP"] = data["DBP"]/200
deepL_keras(data, dls, 3000, 4)#, k_fold=5)
"""
#subprocess.run(['jupyter', 'nbconvert', '--to', 'script', 'deepL.ipynb'])
#"A'G'","A'E'/A'G'","E'G'/A'G'","A'C'/A'G'","C'E'/A'G'","A'B'/A'C'","B'C'/A'C'","C'D'/C'E'","D'E'/C'E'","E'F'/E'G'","F'G'/E'G'","H","f/H","g/H","i/H","H/A'B'","S","S_sys/S","S_dia/S"


# In[ ]:





# In[ ]:




