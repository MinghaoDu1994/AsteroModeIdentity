import numpy as np
import os
import keras.optimizers
import keras.initializers
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelBinarizer  
from sklearn.model_selection import train_test_split  
from keras.models import *
from keras.layers import *
from keras.utils import plot_model, np_utils

# from hyperopt import tpe, STATUS_OK, Trials
# from hyperas import optim
# from hyperas.distributions import choice, uniform

model = load_model('kick_ass_model87')
#plot_model(model, to_file='model87.eps', show_shapes=True)
#model.load_model('model')
x_train = np.load('Simu_Sample_1000.npy')
y_train = np.load('Simu_Sample_label_1000.npy')
# x_test = np.loadtxt('obsubg_kep_nonorm_1000.csv', delimiter=',')
# x_test_max = np.max(x_test)
# x_test_min = np.min(x_test)
# x_test -= x_test_min
# x_test /= x_test_max
x_test = np.loadtxt('obsubg_kep_1000.csv', delimiter=',')
y_test = np.loadtxt('obsubg_kep_label.csv', delimiter=',')

# x_train_median = np.median(x_train)
# x_test_median = np.median(x_test)
#x_train /= x_train_median
# x_test /= x_test_median



x_train_max = np.max(x_train)
x_train_min = np.min(x_train)
x_train -= x_train_min
x_train /= x_train_max



kk=2
x_train = np.expand_dims(x_train,axis=kk)
x_test  = np.expand_dims(x_test, axis=kk)

y_train = LabelBinarizer().fit_transform(y_train)  
y_test = LabelBinarizer().fit_transform(y_test)

sgd=keras.optimizers.SGD(lr=0.1,decay=1e5,
        momentum=0.7,nesterov=True)

adam= keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])  
model.summary()
result = model.fit(x_train, y_train,
					batch_size=2048,
          			epochs=4,
          			verbose=1,
          			validation_split=0.2)
print('result', result)

acc = model.evaluate(x_test, y_test)
print(acc)

save_model(model,'model_train_with_all_simu_data')




# Best performing model chosen hyper-parameters:
# {'batch_size': 0, 'dense_num1': 0, 'dense_num1_1': 0, 'dense_num1_2': 1, 
#'dp': 1, 'fd1': 0, 'fd1_1': 1, 'fd1_2': 1, 'fd1_3': 0, 'fd1_4': 2, 'fd1_5': 0, 
#'pool_size': 1, 'strides1': 3, 'strides2': 1, 'strides2_1': 2}





