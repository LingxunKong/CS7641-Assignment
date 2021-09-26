# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 22:03:30 2021

@author: Lingxun Kong
"""

import pandas as pd
import matplotlib.pyplot as plt
import time
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection,svm
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingClassifier

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('SL_Data.csv')

def preprocessDescription(df):
    
    df["Description"] = df["Description"].str.lower() #make description text lower case
    df["Label"] = df["Label"].str.lower() # make Label strings lower case
    
    #make sure the description and label are string type
    df["Description"] = df["Description"].astype("str")
    df["Label"] = df["Label"].astype("str")
    
    #Tokenize
    import nltk
    from nltk.tokenize import word_tokenize
    df["Text_final"] = [word_tokenize(i) for i in df["Description"]]
    
    for idx,entry in enumerate(df["Text_final"]):
        df.loc[idx,'Text_final'] = str(df["Text_final"][idx])
   
    return df

try:    
    df = preprocessDescription(df) #preprocess text data (description)
except:
    df = pd.read_csv('PreprocessedData.csv')

#Split the data set into training and testing sets
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df["Text_final"],df["Label"],test_size = 0.2,random_state = 101)

#Encode the Label

Encoder = LabelEncoder()

Encoder.fit(df["Label"])

Train_Y = Encoder.transform(Train_Y)

Test_Y = Encoder.transform(Test_Y)

max_features = 1000

Tfidf_vect = TfidfVectorizer(max_features = max_features,min_df = 1)
Tfidf_vect.fit(df["Text_final"])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
X = Tfidf_vect.transform(df["Text_final"])
y = Encoder.transform(df["Label"])



"""
Try fitting the data set with different classifiers
"""

#ANN

scipy.sparse.csr_matrix.sort_indices(X)

y_NN = keras.utils.to_categorical(y,4)

def create_NN_model():
    global activation
    model = Sequential()
    model.add(Dense(32,activation=activation,input_shape=(X.shape[1],)))
    model.add(Dense(16,activation=activation))
    model.add(Dense(16,activation=activation))
    model.add(Dense(4,activation = 'sigmoid'))
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy']
                  
        )
    return model


from keras.wrappers.scikit_learn import KerasClassifier


train_sizes =np.arange(40,321,20)


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
alpha = [0,0.005,0.01,0.05,0.1,0.5]
DT = {}
fig1, ax1 = plt.subplots()
for i in alpha:
    clf=tree.DecisionTreeClassifier(ccp_alpha=i)
    DT[i]=classify(clf,X,y,train_sizes,5,"Decision Tree")
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
    kNN[k]=classify(clf,X,y,train_sizes,5,"kNN")
    ax2.plot(kNN[k][0],kNN[k][2])
ax2.legend(["k ="+str(k) for k in K])
ax2.set_xlabel('Training Sizes')
ax2.set_ylabel('Validation Accuracy')

kNN3 = KNeighborsClassifier(n_neighbors=3)

[train_sizes, avg_train_scores, avg_validation_scores]=classify(kNN3,X,y,train_sizes,5,"kNN k=3",True)
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
    clf=GradientBoostingClassifier(n_estimators=100, learning_rate=k, max_depth=3, random_state=0,ccp_alpha=0.01)
    Boost[k]=classify(clf,X,y,train_sizes,5,"Boosting")
    ax3.plot(Boost[k][0],Boost[k][2])
ax3.legend(["Learning rate ="+str(k) for k in LR])
ax3.set_xlabel('Training Sizes')
ax3.set_ylabel('Validation Accuracy')

Bosstclf=GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=0,ccp_alpha=0.01)

[train_sizes, avg_train_scores, avg_validation_scores]=classify(Bosstclf,X,y,train_sizes,5,"Gradient Boosting Learning Rate = 0.05",True)
print("Boosting validation accuracy = ",avg_validation_scores[-1])


"""
Nueral Network
"""
act= ['sigmoid','relu','softmax']
ANN= {}
fig4, ax4 = plt.subplots()
for k in act:
    global activation
    activation = k
    clf=KerasClassifier(build_fn = create_NN_model,verbose=0,epochs =50)
    ANN[k]=classify(clf,X,y_NN,train_sizes,5,"ANN")
    ax4.plot(ANN[k][0],ANN[k][2])
ax4.legend(["activation of hidden ="+str(k) for k in ANN])
ax4.set_xlabel('Training Sizes')
ax4.set_ylabel('Validation Accuracy')


activation = 'relu'
model_NN = KerasClassifier(build_fn = create_NN_model,verbose=0,epochs =50)

[train_sizes, avg_train_scores, avg_validation_scores]=classify(model_NN,X,y_NN,train_sizes,5,'ANN',True)
print("ANN validation accuracy = ",avg_validation_scores[-1])

model = create_NN_model()
scipy.sparse.csr_matrix.sort_indices(Train_X_Tfidf)
scipy.sparse.csr_matrix.sort_indices(Test_X_Tfidf)
NN_Train_y = keras.utils.to_categorical(Train_Y,4)
NN_Test_y = keras.utils.to_categorical(Test_Y,4)
history = model.fit(Train_X_Tfidf ,NN_Train_y,verbose=0,epochs =100,validation_data = (Test_X_Tfidf ,NN_Test_y))

# summarize history for ANN accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ANN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for ANN loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ANN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




