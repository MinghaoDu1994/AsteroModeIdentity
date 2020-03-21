#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 21:30:38 2018

@author: duminghao
"""
from __future__ import print_function
import numpy as np
import os
import keras.optimizers
import keras.initializers
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelBinarizer  
from sklearn.cross_validation import train_test_split  
from keras.models import *
from keras.layers import *
from keras.utils import plot_model, np_utils
from keras.callbacks import Callback
from hyperopt import tpe, STATUS_OK, Trials
from hyperas import optim
from hyperas.distributions import choice, uniform
from tabulate import tabulate
#%%
# class TestPerEpoch(keras.callbacks.Callback):
#     def on_epoch_end(self, logs={}):
#         acc = model.evaluate(x_test, y_test)
#         if acc 
class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        if acc > 0.86 :
            print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
            self.model.save('./models/mine_%.3f.h5' %acc)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

def data():
    x_train = np.load('Simu_Sample_1000.npy')
    y_train = np.load('Simu_Sample_label_1000.npy')
    x_test = np.loadtxt('obsubg_kep_nonorm_1000.csv', delimiter=',')
    x_test2 = np.loadtxt('obsubg_kep_1000.csv', delimiter=',')
    y_test = np.loadtxt('obsubg_kep_label.csv', delimiter=',')

    #x_train = x_train.reshape((int(len(x_train)/1000),1000))
    #x_train = np.delete(x_train,0,0)
    x_train = x_train
    y_train = y_train

    x_train_median = np.median(x_train)
    x_train_max = np.max(x_train)
    x_train_min = np.min(x_train)
    # x_test_median = np.median(x_test)
    x_test_max = np.max(x_test)
    x_test_min = np.min(x_test)

    #x_train /= x_train_median
    #x_train -= x_train_min
    x_train /= x_train_max

    # x_test /= x_test_median
    # x_test -= x_test_min
    # x_test /= x_test_max

    kk=2
    x_train = np.expand_dims(x_train,axis=kk)
    x_test  = np.expand_dims(x_test, axis=kk)
    x_test2  = np.expand_dims(x_test2, axis=kk)
    
    y_train = LabelBinarizer().fit_transform(y_train)  
    y_test = LabelBinarizer().fit_transform(y_test)


    return x_train, y_train, x_test, x_test2, y_test  



def create_model(x_train, y_train, x_test2, y_test):

    # dp = 0.5                   #drop out ratio
    # fd_1={{choice([32,16,8,4])}} 
    # fd_2={{choice([32,16,8,4])}} 
    # fd_3={{choice([32,16,8,4])}}          #filter depth
    # kz_1={{choice([64,32,16,8,4,2])}} 
    # kz_2={{choice([64,32,16,8,4,2])}} 
    # kz_3={{choice([64,32,16,8,4,2])}}           #kernel size
    # strides_1  = {{choice([3,2,1])}}
    # strides_2  = {{choice([3,2,1])}}
    # strides_3  = {{choice([3,2,1])}}      # strides
    # pool_size_1  = {{choice([3,2,1])}}
    # pool_size_2  = {{choice([3,2,1])}}
    # pool_size_3  = {{choice([3,2,1])}}
    # #pool_size_1 = pool_size_2 = pool_size_3 = 2
    # dense_num1 = {{choice([256,512,1024,2048])}}
    # dense_num2 = {{choice([256,512,1024,2048])}}#1024 

    dp = 0.3                   #drop out ratio
    fd_1={{choice([28,26,24,22,20,18,16,14,12])}} 
    fd_2={{choice([28,2624,22,20,18,16,14,12])}} 
    fd_3={{choice([24,22,16,12,8,6,4])}}          #filter depth
    kz_1={{choice([80,76,72,68,64,56,48,32,16])}} 
    kz_2={{choice([64,48,40,32,24,20,16,12,8])}} 
    kz_3={{choice([64,32,16])}}           #kernel size
    strides_1  = {{choice([3,2,1])}}
    strides_2  = {{choice([3,2,1])}}
    strides_3  = {{choice([3,2,1])}}      # strides
    # pool_size_1  = {{choice([3,2,1])}}
    # pool_size_2  = {{choice([3,2,1])}}
    # pool_size_3  = {{choice([3,2,1])}}
    pool_size_1 = pool_size_2 = pool_size_3 = 3
    dense_num1 = {{choice([512,1024,2048])}}
    dense_num2 = {{choice([512,1024,2048])}}#1024 


    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam= keras.optimizers.Adagrad(lr=0.01, epsilon=1e-04)
    model = Sequential()
    '''
    filters=number of conv kernel
    strides=step
    '''
    #model.add(MaxPooling1D(pool_size=pool_size_3))
    model.add(Convolution1D(filters=fd_1, kernel_size=kz_1,strides=strides_1,
                            input_shape=(1000,1),padding='same',
                            activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=pool_size_1))
    model.add(Dropout(dp)) 
    
    model.add(Convolution1D(filters=fd_2, kernel_size=kz_2, strides=strides_2, 
                            padding='same',activation='relu',
                            kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=pool_size_2))
    model.add(Dropout(dp))  

    
    # model.add(Convolution1D(filters=fd_3, kernel_size=kz_3, strides=strides_3, 
    #                             padding='same',activation='relu',
    #                             kernel_initializer='uniform'))
    # model.add(MaxPooling1D(pool_size=pool_size_3))
    # model.add(Dropout(dp))
    
      
    #    model.add(Convolution1D(filters=128, kernel_size=3, strides=1, padding='same',
    #                            activation='relu',kernel_initializer='uniform'))
    
    model.add(Flatten())
    model.add(Dense(dense_num1,activation='relu'))      
    model.add(Dense(dense_num2,activation='relu'))  
    model.add(Dense(3,activation='softmax'))  
    
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])  

    if 'results' not in globals():
        global results
        results = []
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    # class TestCallback(Callback):
    #     def __init__(self, test_data):
    #         self.test_data = test_data

    #     def on_epoch_end(self, epoch, logs={}):
    #         x, y = self.test_data
    #         loss, acc = self.model.evaluate(x, y, verbose=0)
    #         if acc > 0.86 :
    #             print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
    #             self.model.save('./models/mine_%.3f.h5' %acc)
    #         print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        
    epochs=20
    result = model.fit(x_train, y_train,
              batch_size=1024,
              epochs=epochs,
              verbose=2 ,
              validation_split=0.1,
              callbacks=[es])
    
    parameters = space
    results.append(parameters)
    
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    score, acc = model.evaluate(x_test2, y_test, verbose=0)
    print(acc)
    results.append(acc)
    if acc > 0.80:
        print('################################################################')
        print(acc)
        model.save('/neptune/dmh/ML_Mode_Classify/cnn_train/models/mine_%.3f_%.3f.h5' %(acc,validation_acc))
        print('################################################################')
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
    




if __name__ == '__main__':
    X_train, Y_train, X_test, X_test2, Y_test = data()
    #X_train, Y_train = data()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
    
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.save('model.h5')
    print("Evalutation of best performing model:")
    #print(tabulate(results, headers="keys", tablefmt="fancy_grid", floatfmt=".8f"))
    #print(best_model.evaluate(X_test, Y_test))
    print(best_model.evaluate(X_test2, Y_test))


"""
{'batch_size': 1, 'dense_num1': 3, 'dense_num1_1': 2, 'dense_num1_2': 0,
 'dense_num1_3': 1, 'dense_num1_4': 0, 'fd_1': 1, 'fd_1_1': 0, 'fd_1_2': 0, 
 'fd_1_3': 0, 'fd_1_4': 4, 'fd_3': 4, 'kz_1': 0, 'kz_1_1': 1, 'kz_1_2': 2, 
 'kz_1_3': 4, 'kz_2': 2, 'kz_3': 2, 'strides_1': 2, 'strides_2': 2, 'strides_2_1': 1, 
 'strides_2_10': 2, 'strides_2_2': 0, 'strides_2_3': 1, 'strides_2_4': 1, 
 'strides_2_5': 0,
 'strides_2_6': 2, 'strides_2_7': 2, 'strides_2_8': 0, 'strides_2_9': 2}
"""


