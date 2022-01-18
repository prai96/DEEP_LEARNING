#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification

from collections import Counter


# In[7]:



data = pd.read_csv('Imbalanced_data.csv')
data.head()


# In[8]:


data.shape


# In[5]:


data.value_counts()


# In[9]:


X, Y = make_classification(n_samples=10000, n_features=2, n_informative=2,
                            n_redundant=0, n_repeated=0, n_classes=2,
                            n_clusters_per_class=1,
                            weights=[0.95, 0.05],
                            class_sep=0.8, random_state=100)


# In[10]:


X1, X2 = list(), list()
for i, j in enumerate(X):
    X1.append(j[0])
    X2.append(j[1])


# In[11]:


df = pd.DataFrame({'X1':X1, 'X2':X2, 'Y':Y})


# In[12]:


pal = sns.color_palette('tab10')
print(pal.as_hex())


# In[13]:


#sns.set_palette('icefire')
sns.set_palette(['#55a3cd', '#9c2f45'])
sns.palplot(sns.color_palette())


# In[14]:


# Imblanced Data
plt.figure(figsize=(10,6),dpi=100)

sns.despine(left=True)
sns.scatterplot(x='X1', y='X2', hue = 'Y', data=df)
plt.show()


# In[15]:


df['Y'].value_counts()


# In[16]:


x = df.drop('Y', axis=1)
y = df['Y']


# # Resampling techinique of SMOTE,ADASYN,BODERLINE-SMOTE

# In[18]:


get_ipython().system('pip install imblearn')


# In[19]:


from imblearn.over_sampling import SMOTE

counter = Counter(y)
print('Before',counter)
# oversampling the train dataset using SMOTE
smt = SMOTE()
#X_train, y_train = smt.fit_resample(X_train, y_train)
X_train1, y_train1 = smt.fit_resample(x, y)

counter = Counter(y_train1)
print('After',counter)


# In[20]:


df_sm = X_train1.copy()
df_sm['Y'] = y_train1


# In[21]:


from imblearn.over_sampling import ADASYN

counter = Counter(y)
print('Before',counter)
# oversampling the train dataset using SMOTE
ada = ADASYN()
#X_train, y_train = smt.fit_resample(X_train, y_train)
X_train1, y_train1 = ada.fit_resample(x, y)

counter = Counter(y_train1)
print('After',counter)


# In[23]:


df_ada = X_train1.copy()
df_ada['Y'] = y_train1


# In[22]:


from imblearn.over_sampling import BorderlineSMOTE 

counter = Counter(y)
print('Before',counter)
#oversampling the train dataset using BorderlineSMOTE
sm = BorderlineSMOTE(random_state=42)
#X_train, y_train = smt.fit_resample(X_train, y_train)
X_train1, y_train1 = sm.fit_resample(x, y)

counter = Counter(y_train1)
print('After',counter)


# In[32]:


df_smg = X_train1.copy()
df_smg['Y'] = y_train1


# # VISUALIZATION
# 

# In[48]:


f, axes = plt.subplots(2,2,figsize=(15, 10), dpi=100)
sns.despine()
sns.scatterplot(x='X1', y='X2', hue = 'Y', data=df_smg, ax=axes[0,0])
axes[0,0].set_title('Resampling with SMOTEBORELINE', fontsize=14)
sns.scatterplot(x='X1', y='X2', hue = 'Y', data=df_sm, ax=axes[0,1])
axes[0,1].set_title('Resampling with SMOTE', fontsize=14)
sns.scatterplot(x='X1', y='X2', hue = 'Y', data=df_ada, ax=axes[1,0])
axes[1,0].set_title('Resampling with ADASYN', fontsize=14)  
plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()

