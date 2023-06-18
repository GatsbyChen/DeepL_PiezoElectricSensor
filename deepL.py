#テスト
#import tensorflow.compat.v1 as tf 
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
        print(f"テストデータのMSE: {score[0]*40000}, MAE:{score[1]*200}")
        y_test = y_test*200
        print(f"実際のトレーニングデータ_y\n{y_train*200}")
        print(f"予測トレーニングデータ_y\n{dls.model.predict(X_train, verbose=0)*200}")
        print(f"実際のテストデータ_y\n{y_test}")
        print(f"予測テストデータ_y\n{dls.model.predict(X_test, verbose=0)*200}")
        
    return -40000*score[0]
    

"""
dls = DeepLSetting()
dls.set_initial(39,2,[0,20])
data = pd.read_csv("out0524.csv")
#データを正規化
data["H"] = data["H"]/10
data["H/A'B'"] = data["H/A'B'"]/100
data["SBP"] = data["SBP"]/200
data["DBP"] = data["DBP"]/200
pbounds = {
        'num_layer': (3,7),
        'num_node': (1,2048),
        'dropout': (0.1,0.4),
        'batch': (2,20)}
options = {'c1': 0.8, 'c2': 0.8, 'w': 0.2, 'k': 3, 'p': 2}
dls.bayesOpt(data , pbounds=pbounds, num_epoch=5000)
"""


dls = DeepLSetting()
dls.set_initial(39,2,[0,39])
dls.set_modelLayerAndNode([39,929,929,929,929,2], dropout=0.3)
dls.model_compile()
dls.model.summary()
data = pd.read_csv("out0619.csv")
#データを正規化
data["H"] = data["H"]/10
data["H/A'B'"] = data["H/A'B'"]/100
data["SBP"] = data["SBP"]/200
data["DBP"] = data["DBP"]/200
deepL_keras(data, dls, 5000, 10, plot=True)#k_fold=5)




