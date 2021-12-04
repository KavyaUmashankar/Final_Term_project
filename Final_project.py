#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import warnings
warnings.filterwarnings("ignore") 


# In[2]:


#Loading the dataset
df = pd.read_csv("wine.csv")


# In[3]:


df.isnull().sum()


# In[4]:


df.head()


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[6]:


corr = df.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,linewidths=.5)


# In[7]:


label_encoder =LabelEncoder()
df['quality']= label_encoder.fit_transform(df['quality'])
print(label_encoder.inverse_transform((0,1)))
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[8]:


n=10
fold_value = (len(df)+1)//n


# In[9]:


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
    
    confusion_matrix_res = [[true_negative, false_negative],[false_positive,true_positive]]
    
    return true_positive,true_negative,false_positive,false_negative,confusion_matrix_res


# In[10]:


colnames=["fold","model","Confusion matrix","TruePositive rate","True Negative rate","False positive rate","False Negative rate","Accuracy","Precision","Recall","F1 measure","Error rate","Balance Accuracy","TSS","HSS"]
result = pd.DataFrame(columns=colnames)

copy_df=df

accuracy = 0.0
precision = 0.0
recall= 0.0
tp_rate = 0.0
tn_rate = 0.0
fp_rate = 0.0
fn_rate = 0.0
f1_measure = 0.0
error_rate = 0.0
balance_accuracy = 0.0
TSS = 0.0
HSS = 0.0
begin=0
end=0

for i in range( 1, n+1):
    end=begin+fold_value
    
    train =copy_df.drop(copy_df.index[begin:end])
    test = copy_df[begin:end]
    
    X_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    
    X_test =  test.iloc[:,:-1]
    y_test = test.iloc[:,-1]
    
     
    modelknn = KNeighborsClassifier(n_neighbors=2)
    modelknn.fit(X_train,y_train)
    predknn = modelknn.predict(X_test)
    tp,tn,fp,fn,matrix=confusion_matrix(y_test,predknn)
    tp_rate,tn_rate,fp_rate ,fn_rate ,accuracy ,precision ,recall ,f1_measure ,error_rate ,balance_accuracy ,TSS ,HSS  = evaluation_metrics(tp,tn,fp,fn)
    data = [i,"KNN",matrix,tp_rate,tn_rate,fp_rate ,fn_rate ,accuracy ,precision ,recall ,f1_measure ,error_rate ,balance_accuracy ,TSS ,HSS]
    result.loc[len(result)] = data
    
    modelrf = RandomForestClassifier()
    modelrf.fit(X_train,y_train)
    predrf = modelrf.predict(X_test)
    tp,tn,fp,fn,matrix=confusion_matrix(y_test,predrf)
    tp_rate,tn_rate,fp_rate ,fn_rate ,accuracy ,precision ,recall ,f1_measure ,error_rate ,balance_accuracy ,TSS ,HSS  = evaluation_metrics(tp,tn,fp,fn)
    data = [i,"RF",matrix,tp_rate,tn_rate,fp_rate ,fn_rate ,accuracy ,precision ,recall ,f1_measure ,error_rate ,balance_accuracy ,TSS ,HSS]
    result.loc[len(result)] = data

    
    lst = Sequential()
    lst.add(LSTM(units=11, input_shape=(11,1)))
    lst.add(Dense(16,activation='tanh'))
    lst.add(Dense(1, activation='softmax'))
    lst.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    X_train=X_train.to_numpy()
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
    lst.fit(X_train, y_train,epochs=5, batch_size=160)
    X_test=X_test.to_numpy()
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    predlstm =lst.predict(X_test) 
    tp,tn,fp,fn,matrix=confusion_matrix(y_test,predlstm)
    tp_rate,tn_rate,fp_rate ,fn_rate ,accuracy ,precision ,recall ,f1_measure ,error_rate ,balance_accuracy ,TSS ,HSS  = evaluation_metrics(tp,tn,fp,fn)
    data = [i,"LSTM",matrix,tp_rate,tn_rate,fp_rate ,fn_rate ,accuracy ,precision ,recall ,f1_measure ,error_rate ,balance_accuracy ,TSS ,HSS]
    result.loc[len(result)] = data
    result = result.append(pd.Series(), ignore_index=True)
    begin=end


# In[11]:


print(result)


# In[12]:


names = ["model","AVG TruePositive rate","AVG True Negative rate","AVG False positive rate","AVG False Negative rate","AVG Accuracy","AVG Precision","AVG Recall","AVG F1 measure","AVG Error rate","AVG Balance Accuracy","AVG TSS","AVG HSS"]
avg_df = pd.DataFrame(columns=names)
for i in ['KNN','RF','LSTM']:
    data = result.loc[result['model'] == i]
    avg=[i]
    for i in colnames[3:]:
        avg+=[data[i].mean()]
    avg_df.loc[len(avg_df)] = avg
print(avg_df)


# ## Random forest is the best algorithm, since it creates n number of trees by choosing random attributes, it chooses the class which is proivided max output from the trees

# In[ ]:





# In[ ]:




