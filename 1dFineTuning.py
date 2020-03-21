#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 21:30:38 2018

@author: duminghao

# Fast load model 
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
#%%
def data():
    data = np.loadtxt('obsubg_kep_1000.csv', delimiter=',')
    label = np.loadtxt('obsubg_kep_label.csv', delimiter=',')
    x_train = data[:500]
    x_test = data[500:]
    y_train = label[:500]
    y_test = label[500:]    
    kk=2
    x_train = np.expand_dims(x_train,axis=kk)
    x_test  = np.expand_dims(x_test, axis=kk)
    
    y_train = LabelBinarizer().fit_transform(y_train)  
    y_test = LabelBinarizer().fit_transform(y_test)


    return x_train, y_train, x_test, y_test  



def create_model(x_train, y_train, x_test, y_test):

    sgd=keras.optimizers.SGD(lr={{choice([0.1,0.01,0.001])}},decay={{choice([1e5,1e6])}},
        momentum={{choice([0.9,0.7])}},nesterov=True)
    adam= keras.optimizers.Adagrad(lr=0.01, epsilon=1e-04)
    model=load_model('mine_0.850.h5')
    for layer in model.layers[:-3]:
        layer.trainable = False
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])  
    result = model.fit(x_train, y_train,
              batch_size=256,
              epochs=30,
              verbose=2,
              validation_split=0.1,
              callbacks=[TestCallback])
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        if acc > 0.94 :
            print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
            self.model.save('./models/mine_%.3f.h5' %acc)
        


# if __name__ == '__main__':
#     X_train, Y_train, X_test, Y_test = data()
#     best_run, best_model = optim.minimize(model=create_model,
#                                           data=data,
#                                           algo=tpe.suggest,
#                                           max_evals=20,
#                                           trials=Trials())
    
#     print("Evalutation of best performing model:")
#     print(best_model.evaluate(X_test, Y_test))
#     print("Best performing model chosen hyper-parameters:")
#     print(best_run)
X_train, Y_train, X_test, Y_test = data()
sgd=keras.optimizers.SGD(lr=0.01,decay=1e-5,
    momentum=0.9,nesterov=True)
adam= keras.optimizers.Adagrad(lr=0.005, epsilon=1e-04)
model=load_model('kick_ass_model87')
for layer in model.layers[:-3]:
    layer.trainable = False
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])  
result = model.fit(X_train, Y_train,
              batch_size=128,
              epochs=1000,
              verbose=2,
              validation_split=0.1,
              callbacks=[TestCallback([X_test,Y_test])])


