#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv('E:\\itsstudytym\\Python Project\\ML Notebook Sessions\\Breast Cancer Disease Prediction\\data.csv')
data.head()


# #### Check Null Values

# In[5]:


data.isnull().sum()


# #### Check Balanced or Imbalanced Dataset

# In[6]:


sns.countplot(x='diagnosis',data=data)


# In[7]:


data.shape


# In[8]:


M = len(data[data['diagnosis']=='M'])/len(data)
M


# #### Check Correlation

# In[9]:


data = data.drop('id',axis=1)
plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),annot=True)


# In[10]:


data = data.drop(['perimeter_mean','radius_mean','radius_worst','perimeter_worst','concave points_mean','radius_se','perimeter_se','texture_se','compactness_worst','concave points_worst','compactness_mean','texture_se'],axis=1)
data.head()data = data.drop(['texture_worst','area_worst','concavity_worst'],axis=1)


# In[11]:


data = data.drop(['texture_worst','area_worst','concavity_worst'],axis=1)


# In[12]:


plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),annot=True)


# In[13]:


data = data.drop(['Unnamed: 32'],axis=1)
data.head(10)


# ### Feature Scaling

# In[14]:


from sklearn.preprocessing import StandardScaler
stsc = StandardScaler()


# In[15]:


x = data.drop('diagnosis',axis=1)
y = data['diagnosis']


# In[16]:


x = pd.DataFrame(stsc.fit_transform(x))


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)


# ### Decision Tree Classification Model

# In[18]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(x_train,y_train)


# In[19]:


y_predict = dtree.predict(x_test)
y_predict


# In[20]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
dtree_acc = accuracy_score(y_test,y_predict)
dtree_acc


# In[21]:


dtree_cm = confusion_matrix(y_test,y_predict)
dtree_cm


# In[22]:


dtree_cls = classification_report(y_test,y_predict)
print(dtree_cls)


# ### Random Forest Classification Model

# In[23]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=5)
rfc.fit(x_train,y_train)


# In[24]:


y_pred = rfc.predict(x_test)
y_pred


# In[45]:


rfc_acc = accuracy_score(y_test,y_pred)
rfc_acc


# In[46]:


rfc_cm = confusion_matrix(y_test,y_pred)
rfc_cm


# In[47]:


rfc_clsr = classification_report(y_test,y_pred)
print(rfc_clsr)


# ### K-Nearest Neighbor Classification Model

# In[28]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn_nscore = []
for k in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn,x_train,y_train,cv=10)
    knn_nscore.append(score.mean())
plt.figure(figsize=(9,7))
plt.plot([k for k in range(1,50)],knn_nscore,color='red',marker='o')


# In[48]:


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)


# In[39]:


knny_predict = knn.predict(x_test)
knny_predict


# In[49]:


knn_acc = accuracy_score(y_test,knny_predict)
knn_acc


# In[52]:


print(confusion_matrix(y_test,knny_predict))


# In[51]:


print(classification_report(y_test,knny_predict))

