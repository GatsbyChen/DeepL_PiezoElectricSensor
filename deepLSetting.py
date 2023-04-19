#!/usr/bin/env python
# coding: utf-8

# In[30]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import keras.optimizers as opt
import subprocess
from bayes_opt import BayesianOptimization
#機械学習のハイパーパラメータ、学習方法などをまとめたクラス。
class DeepLSetting:
    
    def __init__(self):
        self.num_featureValue = None #特徴量の数
        self.num_output = None #目的変数の数
        self.trainRow = None #学習データの割合
        self.model = Sequential()#学習モデル。
        self.num_nodeList = None #レイヤーの配列。
    
    #説明変数の数、目的変数の数、学習データの割合を設定する。
    def set_initial(self,  num_featureValue, num_output, trainRow):
        self.num_featureValue = num_featureValue
        self.num_output = num_output
        self.trainRow = trainRow
    
        
    #modelにNN構造を手動で設定する
    def set_modelLayerAndNode(self,num_nodeList: list, activation="relu", dropout=0.2):
        self.num_nodeList = num_nodeList.copy()
        self.model.add(Dense(num_nodeList[1], input_shape=(num_nodeList[0],)))
        #self.model.add(Activation(activation))
        self.model.add(Dropout(dropout))
        if(len(num_nodeList) == 2):
            return
        for i in range(2, len(num_nodeList)):
            self.model.add(Dense(num_nodeList[i]))
            #self.model.add(Activation(activation))
            if i < len(num_nodeList)-1:
                self.model.add(Dropout(dropout))

    #モデルをコンパイルする。
    def model_compile(self, loss_tmp='mean_squared_error', optimizer_tmp=opt.Adam(), metrics_tmp=["mae"]):
        self.model.compile(
        loss = loss_tmp,
        optimizer=optimizer_tmp,
        metrics=metrics_tmp)
    
    #ベイズ最適化用の関数
    def func(self, num_layer, num_node, dropout, batch, data=None, num_epoch=3000, loss_tmp='mean_squared_error', optimizer_tmp=opt.Adam()):
        data_tmp = data.copy()
        nodeList = [self.num_featureValue]
        for i in range(int(num_layer)):
            if(i == int(num_layer)-1):
                nodeList.append(self.num_output)
                break
            nodeList.append(int(num_node))
        self.model = Sequential() #モデルの初期化
        self.set_modelLayerAndNode(nodeList, dropout=dropout)
        self.model_compile(loss_tmp=loss_tmp, optimizer_tmp=optimizer_tmp)
        #データの準備
        X = data_tmp.iloc[:, 0:self.num_featureValue]
        y = data_tmp.iloc[:, self.num_featureValue:len(data.columns)]
        X_train = X[self.trainRow[0]:self.trainRow[1]]
        X_test = X[self.trainRow[1]:len(X)]
        y_train = y[self.trainRow[0]:self.trainRow[1]]
        y_test = y[self.trainRow[1]:len(X)]
        #学習
        history = self.model.fit(X_train, y_train, batch_size=int(batch), epochs=num_epoch, verbose=0)
        score = self.model.evaluate(X_test, y_test, verbose=0)
        return -40000*score[0]
    
     #NN構造をレイヤー数、ノード数、ドロップアウト、バッチ数を最適化する。
    def bayesOpt(self, data, pbounds,num_epoch=3000, n_iter=25, loss_tmp='mean_squared_error', optimizer_tmp=opt.Adam()):
        if(self.num_featureValue == None):
            raise Exception("先にset_initial関数で初期化してください。")
        data_tmp = data.copy()
        print(self.func.__defaults__)
        self.func.__func__.__defaults__ = data_tmp, num_epoch, loss_tmp, optimizer_tmp
        optimizer = BayesianOptimization(f=self.func, pbounds=pbounds)
        optimizer.maximize(init_points=5, n_iter=n_iter)
        
        


# In[ ]:




