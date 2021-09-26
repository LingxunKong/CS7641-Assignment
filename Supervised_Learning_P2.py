# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 20:06:17 2021

@author: Lingxun Kong
"""
 
"""


"""

import numpy as np
import math
import pandas as pd

num_init_sample = 5000

#Underlying function
def feasibility(x):
    if sum(x) > 3.2:
        return False
    elif x[0]*x[1] < 0.2:
        return False
    elif math.exp(x[0]*x[1]*x[2]) > 2*x[3]:
        return False
    elif x[2]*x[3] > x[1]:
        return False
    else:
        return True

np.random.seed(2)
data = {}
label = {}
truelabel = 0
falselabel = 0
#Sample the dataset
for i in range(num_init_sample):
    x = np.random.random_sample(size = (4))
    fea = feasibility(x)
    if fea == True: 
        data[i] =x
        label[i] = 1
        truelabel+=1
    else:
        if np.random.random() > 0.85:
            data[i] = x
            label[i] = 0
            falselabel += 1
            
print("# of true labels =",truelabel)
print("# of false labels =",falselabel)


#Create features (list of edges) and lables
df_data = pd.DataFrame.from_dict(data,orient='index')
df_label = pd.DataFrame.from_dict(label,orient='index')


import matplotlib.pyplot as plt
import time
from sklearn import svm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

y =df_label.values.ravel()
X = df_data.values

"""
Try fitting the data set with different classifiers
"""

#ANN



def create_NN_model():
    global activation
    global neuron
    model = Sequential()
    model.add(Dense(neuron[0],activation=activation,input_shape=(X.shape[1],)))
    model.add(Dense(neuron[1],activation=activation))
    # model.add(Dense(16,activation='relu'))
    model.add(Dense(neuron[2],activation=activation))
    
    model.add(Dense(1,activation = 'sigmoid'))
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy']
                  
        )
    return model


from keras.wrappers.scikit_learn import KerasClassifier


train_sizes =np.arange(round(df_data.shape[0]*0.2),round(df_data.shape[0]*0.79),80)

train_sizes = np.insert(train_sizes,0,100)

def classify(estimator,X,y,train_sizes,cv,title,plot=False):
    start = time.time() 
    train_sizes, train_scores, validation_scores=learning_curve(estimator,X,y,train_sizes=train_sizes,cv = cv)
    end = time.time()
    avg_train_scores = np.average(train_scores,axis = 1)
    avg_validation_scores = np.average(validation_scores,axis=1)
    if plot:
        plt.figure()
        plt.plot(train_sizes,avg_train_scores)
        plt.plot(train_sizes,avg_validation_scores)
        plt.legend(["Train accuracy","Test accuracy"])
        plt.title(title)
        plt.xlabel('Train size')
        plt.ylabel('Accuracy')
        wall_clock_time = end - start
        print("Wall clock time for {0} = {1:4f} secs".format(title,wall_clock_time))
    return train_sizes, avg_train_scores, avg_validation_scores
    
 
"""
Decision Tree
"""   
 
#Decision Tree with Pruning
alpha = [0.0,0.005,0.01,0.05,0.1,0.5]
DT = {}
fig1, ax1 = plt.subplots()
for i in alpha:
    clf=tree.DecisionTreeClassifier(ccp_alpha=i)
    DT[i]=classify(clf,X,y,train_sizes,5,"Decision Tree pruning comparison")
    ax1.plot(DT[i][0],DT[i][2])
ax1.legend(["Pruning rate ="+str(j) for j in alpha])
ax1.set_xlabel('Training Sizes')
ax1.set_ylabel('Validation Accuracy')

DTclf=tree.DecisionTreeClassifier(ccp_alpha=0.005)
[train_sizes, avg_train_scores, avg_validation_scores]=classify(DTclf,X,y,train_sizes,5,"Decision Tree Pruning Rate = 0.005",True)
print("Decision trees validation accuracy = ",avg_validation_scores[-1])


"""
kNN
"""   
#Test different k
K = [3,4,5,6]
kNN = {}
fig2, ax2 = plt.subplots()
for k in K:
    clf=KNeighborsClassifier(n_neighbors=k)
    kNN[k]=classify(clf,X,y,train_sizes,5,"kNN k comparison")
    ax2.plot(kNN[k][0],kNN[k][2])
ax2.legend(["k ="+str(k) for k in K])
ax2.set_xlabel('Training Sizes')
ax2.set_ylabel('Validation Accuracy')

kNN4 = KNeighborsClassifier(n_neighbors=4)
[train_sizes, avg_train_scores, avg_validation_scores]=classify(kNN4,X,y,train_sizes,5,"kNN k=4",True)
print("kNN validation accuracy = ",avg_validation_scores[-1])


"""
SVM
"""   

#SVM with linear kernel
SVML = svm.SVC(C = 100,kernel='linear',gamma = 'auto')

# SVM with rbf kernel
SVMR = svm.SVC(C = 100,kernel='rbf',gamma = 'auto')

classify(SVML,X,y,train_sizes,5,"SVM linear kernel",True)

[train_sizes, avg_train_scores, avg_validation_scores]=classify(SVMR,X,y,train_sizes,5,"SVM rbf kernel",True)
print("SVM validation accuracy = ",avg_validation_scores[-1])

C = [1,10,50,100]
SVMlist = {}
fig5, ax5 = plt.subplots()
for k in C:
    clf=svm.SVC(C = k,kernel='rbf',gamma = 'auto')
    SVMlist[k]=classify(clf,X,y,train_sizes,5,"SVM C comparison")
    ax5.plot(SVMlist[k][0],SVMlist[k][2])
ax5.legend(["C ="+str(k) for k in C])
ax5.set_xlabel('Training Sizes')
ax5.set_ylabel('Validation Accuracy')


"""
Boosting
"""   
#Test different learning rate
LR = [0.05,0.1,0.5,1]
Boost = {}
fig3, ax3 = plt.subplots()
for k in LR:
    clf=GradientBoostingClassifier(n_estimators=100, learning_rate=k, max_depth=3, random_state=0,ccp_alpha=0.005)
    Boost[k]=classify(clf,X,y,train_sizes,5,"Boosting learning rate comparison")
    ax3.plot(Boost[k][0],Boost[k][2])
ax3.legend(["Learning rate ="+str(k) for k in LR])
ax3.set_xlabel('Training Sizes')
ax3.set_ylabel('Validation Accuracy')

Bosstclf=GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=0,ccp_alpha=0.005)

[train_sizes, avg_train_scores, avg_validation_scores]=classify(Bosstclf,X,y,train_sizes,5,"Gradient Boosting Learning Rate = 0.05",True)
print("Boosting validation accuracy = ",avg_validation_scores[-1])


"""
Nueral Network
"""
neu = [[32,16,16],[16,8,4],[32,16,8],[16,8,8]]
activation = 'relu'
ANN= {}
fig4, ax4 = plt.subplots()
for k in neu:
    global neuron
    neuron = k
    clf=KerasClassifier(build_fn = create_NN_model,verbose=0,epochs =50)
    ANN[tuple(k)]=classify(clf,X,y,train_sizes,5,"ANN neurons comparison")
    ax4.plot(ANN[tuple(k)][0],ANN[tuple(k)][2])
ax4.legend(["# of neurons = "+str(k) for k in ANN])
ax4.set_xlabel('Training Sizes')
ax4.set_ylabel('Validation Accuracy')



model_NN = KerasClassifier(build_fn = create_NN_model,verbose=0,epochs =100)

[train_sizes, avg_train_scores, avg_validation_scores]=classify(model_NN,X,y,train_sizes,5,'ANN',True)
print("ANN validation accuracy = ",avg_validation_scores[-1])

history = model_NN.fit(X,y,validation_split=0.2,verbose=0,epochs =100)

# summarize history for accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ANN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ANN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()