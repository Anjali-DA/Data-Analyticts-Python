#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing sklearn libraries for training the dataset
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import precision_score,recall_score, accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


# In[2]:


df= pd.read_csv('heart.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


features= ['sex','fbs','restecg','exng','slp','thall','age', 'trtbps','chol', 'thalachh','oldpeak']
label= 'output'


# In[7]:


#separating the features(X) and lables(y)
X,y=df[features].values,df[label].values


# In[8]:


for n in range(0,4):
    print("Patient", str(n+1),"\n Features:", list(X[n]),"\n Label:", y[n])


# In[9]:


for col in features:
    df.boxplot(column=col,by='output',figsize=(12,6))
    plt.title(col)
plt.show()


# In[10]:


#splitting the data into 70-30% into training and test dataset
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.30, random_state=40)
print('Training set: %d rows\nTest set: %d rows' %(X_train.shape[0],X_test.shape[0]))


# Now we have splitted the data into four datasets:
# - **X_train**: The feature values- use to train the model
# - **y_train**: The corresponding labels- use to train the model
# - **X_test**: The feature values- used to validate the model
# - **y_test**: The corresponding labels- used to validate the model

# In[11]:


#Training and evaluating the binary classification the model
reg=0.01
model= LogisticRegression(C=1/reg,solver='liblinear').fit(X_train,y_train)
print(model,"\n")


# In[12]:


#evaluate the model using the test data
predictions= model.predict(X_test)
print('Predicted labels: ', predictions)
print('Actual label: ',y_test)


# - As the arrays of labels are too long so we will go with checking the accuracy of the predictions

# In[13]:


print('Accuracy score: ', accuracy_score(y_test,predictions))


# ### Performing classification report

# In[14]:


print(classification_report(y_test,predictions))


# In[15]:


#Evaluating the precision and recall score
print("Overall Precision: ", precision_score(y_test,predictions))
print("Overall Recall: ", recall_score(y_test,predictions))


# ### Performing the confusion matrix

# In[16]:


cm= confusion_matrix(y_test,predictions)
print(cm)


# **Note**: Statistical machine learning algorithms like logistic regression are based on probability. To see the probability pairs we will use **predict_proba**
# 
# **Setting the threshold values**
# A threshold value of 0.5 is used to decide whether the predicted label is a 1 (P(y) > 0.5) or a 0 (P(y) <= 0.5)

# In[17]:


y_scores= model.predict_proba(X_test)
print(y_scores)


# ### Plotting the ROC curve
# - The decision to score a predictions as a 1 or a 0 depends on the threshold to which the predicted probabilities are compared.
# - If we'll change the threshold, it would affect the predictions along with change in the confusion matrix.
# - To evaluate a classifier with true positive rate and false positive rate for a range of possible thresholds- we'll form a chart known as *received operator characteristic* **(ROC)**

# In[18]:


from sklearn.metrics import roc_curve, confusion_matrix
fpr, tpr, threshold= roc_curve(y_test,y_scores[:,1])


# In[19]:


#plotting the ROC curve
fig= plt.figure(figsize=(12,6))
plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()


# - For better performance of the model instead of using ROC curve, we'll use, **Area Under the Curve(AUC)**- is a value 0 and 1 that quantifies the overall performance of the model. The closer to 1 this value is, the better the model. 

# In[20]:


from sklearn.metrics import roc_auc_score

auc= roc_auc_score(y_test,y_scores[:,1])
print('Area under the curve: '+ str(auc))

