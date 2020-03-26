#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[101]:


df = pd.read_csv('E:\\itsstudytym\\Python Project\\ML Notebook Sessions\\Predict Heart Disease\\heart.csv')
df


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


sns.set_style('whitegrid')
sns.countplot(x='target',data=df)


# #### It is Balanced Dataset

# In[12]:


sns.countplot(x='sex',data=df)


# #### 1 for Male and 0 for Female, Looks like Men frequency is more than Women

# In[18]:


sns.countplot(x='sex',hue='target',data=df)


# #### sex(0) = Female and sex(1) = Male and target(0) = has not Heart Disease and target(1) = has Heart Disease 

# In[26]:


sns.countplot(x='fbs',hue='sex',data=df)


# In[27]:


sns.countplot(x='sex',hue='exang',data=df)


# In[41]:


sns.countplot(x='sex',hue='cp',data=df)


# In[28]:


plt.figure(figsize=(14,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# #### No Multicolinearity Exist

# In[34]:


sns.distplot(df['thalach'],kde=False,bins=30,color='blue')


# In[37]:


sns.distplot(df['chol'],kde=False,bins=30,color='green')


# In[43]:


sns.distplot(df['trestbps'],kde=False,bins=20,color='orange')


# In[45]:


plt.figure(figsize=(15,12))
sns.countplot(x='age',hue='target',data=df)


# #### Number of Peoples having heart disease according to age

# In[53]:


plt.figure(figsize=(7,5))
sns.scatterplot(x='age',y='chol',data=df,hue='target')


# In[55]:


plt.figure(figsize=(7,5))
sns.scatterplot(x='age',y='thalach',hue='target',data=df)


# In[56]:


plt.figure(figsize=(7,5))
sns.scatterplot(x='chol',y='thalach',hue='target',data=df)


# In[108]:


x = df.drop('target',axis=1)
y = df['target']


# In[103]:


cp = pd.get_dummies(df['cp'],drop_first=True,prefix='cp')
thal = pd.get_dummies(df['thal'],drop_first=True,prefix='thal')
slope = pd.get_dummies(df['slope'],drop_first=True,prefix='slope')


# In[109]:


x = pd.concat([x,cp,thal,slope],axis=1)
x


# In[110]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train.shape


# In[111]:


x_test.shape


# In[112]:


x_train.head()


# In[113]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)


# In[114]:


y_predict = model.predict(x_test)
y_predict


# In[123]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
dt_cm = confusion_matrix(y_test,y_predict)
sns.heatmap(dt_cm,annot=True)
plt.xlabel('Predict')
plt.ylabel('Actual')


# In[116]:


dt_acc = accuracy_score(y_test,y_predict)
dt_acc


# In[117]:


dt_cls = classification_report(y_test,y_predict)
print(dt_cls)


# In[118]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(x_train,y_train)


# In[119]:


log_model.predict(x_test)


# In[120]:


log_cm = confusion_matrix(y_test,y_predict)
log_cm


# In[121]:


log_acc = accuracy_score(y_test,y_predict)
log_acc


# In[122]:


log_cls = classification_report(y_test,y_predict)
print(log_cls)

