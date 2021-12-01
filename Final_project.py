#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential 


# In[2]:


#Loading the dataset
df = pd.read_csv("wine.csv")


# In[3]:


df.head()


# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[5]:


corr = df.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,linewidths=.5)


# In[6]:


label_encoder =LabelEncoder()
df['quality']= label_encoder.fit_transform(df['quality'])
print(label_encoder.inverse_transform((0,1)))
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[7]:


#Implementing cross validation 
n=10
kf = KFold(n_splits=n, random_state=None)


# In[8]:


def evaluation_metrics(tp,tn,fp,fn):
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    precision = tp / (tp+fp)
    recall= tp / (tp+fn)
    tp_rate = tp/(tp+fn)
    tn_rate = tn/(tn+fp)
    fp_rate = fp/(tn+fp)
    fn_rate = fn/(tp+fn)
    f1_measure = 2*(precision*recall)/(precision+recall)
    error_rate = (fp+fn)/(tp+tn+fp+fn)
    balance_accuracy = (tp_rate + tn_rate)/2
    TSS = (tp/(tp+fn))-(fp/(fp+tn))
    HSS = (2*((tp*tn)-(fp*fn)))/((tp+fn)*(fn+tn)+(fp+tp)*(fp+tn))
    return tp_rate,tn_rate,fp_rate,fn_rate,accuracy,precision,recall,f1_measure,error_rate,balance_accuracy,TSS,HSS
          



def confusion_matrix(truth,predicted):
    
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    for true,pred in zip(truth,predicted):
        if true == 1:
            if pred == true:
                true_positive += 1
            elif pred != true:
                false_negative += 1

        elif true == 0:
            if pred == true:
                true_negative += 1
            elif pred != true:
                false_positive += 1
    positive = true_positive + false_negative
    negative = true_negative + false_positive
    print("Fold Size = ",positive+negative)
    print("The total positive = ",positive)
    print("The total negative = ",negative)
    confusion_matrix_res = [[true_negative, false_negative],[false_positive,true_positive]]
    print("Confusion Matrix : [[true_negative, false_negative],[false_positive,true_positive]] = ",confusion_matrix_res)
    return true_positive,true_negative,false_positive,false_negative


# ## KNN

# In[9]:


modelknn = KNeighborsClassifier(n_neighbors=2)
count=0
accuracy = [None] * 10
precision = [None] * 10
recall= [None] * 10
tp_rate = [None] * 10
tn_rate = [None] * 10
fp_rate = [None] * 10
fn_rate = [None] * 10
f1_measure = [None] * 10
error_rate = [None] * 10
balance_accuracy = [None] * 10
TSS = [None] * 10
HSS = [None] * 10
    
for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    modelknn.fit(X_train,y_train)
    pred_values = modelknn.predict(X_test)
    print("Fold = ",count+1)
    tp,tn,fp,fn=confusion_matrix(y_test,pred_values)
    tp_rate[count],tn_rate[count],fp_rate[count],fn_rate[count],accuracy[count],precision[count],recall[count],f1_measure[count],error_rate[count],balance_accuracy[count],TSS[count],HSS[count] = evaluation_metrics(tp,tn,fp,fn)
    print("True Positive rate = ",tp_rate[count])
    print("True Negative rate = ",tn_rate[count])
    print("False Positive rate = ",fp_rate[count])
    print("False negative rate = ",fn_rate[count])
    print("Accuracy = ",accuracy[count])
    print("Precision = ",precision[count])
    print("Recall = ",recall[count])
    print("F1 Measure = ",f1_measure[count])
    print("Error rate = ",error_rate[count])
    print("Balanced Accuracy = ",balance_accuracy[count])
    print("True Skill statistics = ",TSS[count])
    print("Heidke Skill Score = ",HSS[count])
 
    print("-"*100)
    count+=1
    
print("*"*100)
print("Average True Positive rate = ",sum(tp_rate)/n)
print("Average True Negative rate = ",sum(tn_rate)/n)
print("Average False Positive rate = ",sum(fp_rate)/n)
print("Average False negative rate = ",sum(fn_rate)/n)
print("Average Accuracy = ", sum(accuracy)/n)
print("Average Precision = ",sum(precision)/n)
print("Average Recall = ",sum(recall)/n)
print("Average F1 Measure = ",sum(f1_measure)/n)
print("Average Error rate = ",sum(error_rate)/n)
print("Average Balanced Accuracy = ",sum(balance_accuracy)/n)
print("Average True Skill statistics = ",sum(TSS)/n)
print("Average Heidke Skill Score = ",sum(HSS)/n)
print("*"*100)


# ## Random forest

# In[10]:


#Random forest algorithm
modelrf = RandomForestClassifier()
count=0
accuracy = [None] * 10
precision = [None] * 10
recall= [None] * 10
tp_rate = [None] * 10
tn_rate = [None] * 10
fp_rate = [None] * 10
fn_rate = [None] * 10
f1_measure = [None] * 10
error_rate = [None] * 10
balance_accuracy = [None] * 10
TSS = [None] * 10
HSS = [None] * 10

for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    modelrf.fit(X_train,y_train)
    pred_values = modelrf.predict(X_test)
    
    print("Fold = ",count+1)
    tp,tn,fp,fn=confusion_matrix(y_test,pred_values)
    tp_rate[count],tn_rate[count],fp_rate[count],fn_rate[count],accuracy[count],precision[count],recall[count],f1_measure[count],error_rate[count],balance_accuracy[count],TSS[count],HSS[count] = evaluation_metrics(tp,tn,fp,fn)
    print("True Positive rate = ",tp_rate[count])
    print("True Negative rate = ",tn_rate[count])
    print("False Positive rate = ",fp_rate[count])
    print("False negative rate = ",fn_rate[count])
    print("Accuracy = ",accuracy[count])
    print("Precision = ",precision[count])
    print("Recall = ",recall[count])
    print("F1 Measure = ",f1_measure[count])
    print("Error rate = ",error_rate[count])
    print("Balanced Accuracy = ",balance_accuracy[count])
    print("True Skill statistics = ",TSS[count])
    print("Heidke Skill Score = ",HSS[count])
 
    print("-"*100)
    count+=1
    
print("*"*100)
print("Average True Positive rate = ",sum(tp_rate)/n)
print("Average True Negative rate = ",sum(tn_rate)/n)
print("Average False Positive rate = ",sum(fp_rate)/n)
print("Average False negative rate = ",sum(fn_rate)/n)
print("Average Accuracy = ", sum(accuracy)/n)
print("Average Precision = ",sum(precision)/n)
print("Average Recall = ",sum(recall)/n)
print("Average F1 Measure = ",sum(f1_measure)/n)
print("Average Error rate = ",sum(error_rate)/n)
print("Average Balanced Accuracy = ",sum(balance_accuracy)/n)
print("Average True Skill statistics = ",sum(TSS)/n)
print("Average Heidke Skill Score = ",sum(HSS)/n)
print("*"*100)


# ## LSTM    

# In[11]:


count=0
accuracy = [None] * 10
precision = [None] * 10
recall= [None] * 10
tp_rate = [None] * 10
tn_rate = [None] * 10
fp_rate = [None] * 10
fn_rate = [None] * 10
f1_measure = [None] * 10
error_rate = [None] * 10
balance_accuracy = [None] * 10
TSS = [None] * 10
HSS = [None] * 10

lst = Sequential()
lst.add(LSTM(units=9, input_shape=(X_train.shape[1],1)))
lst.add(Dense(16,activation='tanh'))
lst.add(Dense(1, activation='sigmoid'))
lst.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    
    X_train=X_train.to_numpy()
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
    

    lst.fit(X_train, y_train,epochs=5, batch_size=160)
    X_test=X_test.to_numpy()
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    ypred=lst.predict(X_test) 

    for i in range(len(ypred)):
        if ypred[i]>=0.5250:
            ypred[i]=1
        elif ypred[i]<0.5250:
            ypred[i]=0
    
    print("Fold = ",count+1)
    tp,tn,fp,fn=confusion_matrix(y_test,ypred)
    tp_rate[count],tn_rate[count],fp_rate[count],fn_rate[count],accuracy[count],precision[count],recall[count],f1_measure[count],error_rate[count],balance_accuracy[count],TSS[count],HSS[count] = evaluation_metrics(tp,tn,fp,fn)
    print("True Positive rate = ",tp_rate[count])
    print("True Negative rate = ",tn_rate[count])
    print("False Positive rate = ",fp_rate[count])
    print("False negative rate = ",fn_rate[count])
    print("Accuracy = ",accuracy[count])
    print("Precision = ",precision[count])
    print("Recall = ",recall[count])
    print("F1 Measure = ",f1_measure[count])
    print("Error rate = ",error_rate[count])
    print("Balanced Accuracy = ",balance_accuracy[count])
    print("True Skill statistics = ",TSS[count])
    print("Heidke Skill Score = ",HSS[count])
 
    print("-"*100)
    count+=1
    
print("*"*100)
print("Average True Positive rate = ",sum(tp_rate)/n)
print("Average True Negative rate = ",sum(tn_rate)/n)
print("Average False Positive rate = ",sum(fp_rate)/n)
print("Average False negative rate = ",sum(fn_rate)/n)
print("Average Accuracy = ", sum(accuracy)/n)
print("Average Precision = ",sum(precision)/n)
print("Average Recall = ",sum(recall)/n)
print("Average F1 Measure = ",sum(f1_measure)/n)
print("Average Error rate = ",sum(error_rate)/n)
print("Average Balanced Accuracy = ",sum(balance_accuracy)/n)
print("Average True Skill statistics = ",sum(TSS)/n)
print("Average Heidke Skill Score = ",sum(HSS)/n)
print("*"*100)


# In[ ]:




