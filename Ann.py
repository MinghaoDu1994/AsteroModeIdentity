# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:15:22 2018
Sklearn Neural Network
@author: Administrator
"""
#%%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report  
from sklearn.preprocessing import LabelBinarizer  
from sklearn.model_selection import train_test_split  
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

#%%
#actf=['identity', 'logistic', 'tanh', 'relu']
#solvf=['lbfgs','sgd','adam']
#%%
sample=np.loadtxt('/neptune/dmh/ML_Mode_Classify/data/Sample_interP1000.csv',delimiter=',')
label =np.loadtxt('/neptune/dmh/ML_Mode_Classify/data/Sample_label(l).csv',delimiter=',')

pixel_num=1000
type_num=3


#%%
sample=np.transpose(sample)
sample -= sample.min()  
sample /= sample.max() 

X_train, X_test, y_train, y_test = train_test_split(sample, label,test_size=0.2,random_state=40)
labels_train = LabelBinarizer().fit_transform(y_train)  
labels_test = LabelBinarizer().fit_transform(y_test)


#%%

parameters={'hidden_layer_sizes':(100,300,500,700,1000,1500,2000),'alpha':(0.0001,0.001,0.01,0.1,0.3,0.5)}
mlp=MLPClassifier(activation='relu',solver='sgd',
                  batch_size='auto',max_iter=3000)
clf=GridSearchCV(mlp,parameters,scoring='accuracy',n_jobs=10)
clf.fit(X_train,y_train)
aa=clf.score(X_test,y_test)
bb=clf.predict(X_test)
cv_result=pd.DataFrame.from_dict(clf.cv_results_)
with open('cv_results.csv','w') as f:
	cv_result.to_csv(f)













