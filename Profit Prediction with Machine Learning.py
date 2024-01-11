#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("W:\SUMAN\BOA\Data science project\startups.csv")
df.head()


# In[2]:


df.describe()


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# In[5]:


import numpy as np


# In[6]:


x=df[["R&D Spend","Administration","Marketing Spend"]]
y=df["Profit"]


# In[7]:


x=x.to_numpy()
x


# In[8]:


y=y.to_numpy()
y


# In[9]:


y=y.reshape(-1,1)
y


# In[10]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)


# In[11]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain, ytrain)
ypred=model.predict(xtest)
df=pd.DataFrame(data={"Predicted Profit": ypred.flatten()})
print(df.head())

