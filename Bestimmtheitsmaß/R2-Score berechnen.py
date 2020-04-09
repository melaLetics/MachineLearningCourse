#!/usr/bin/env python
# coding: utf-8

# ### Praxisprojekt: R^2 - Wert berechnen

# In[3]:


import pandas as pd

df = pd.read_csv("./autos_prepared.csv")


# In[4]:


df.head()


# ### Teil 1: Train / Test

# In[8]:


# Train / Test

from sklearn.model_selection import train_test_split

X = df[["kilometer", "powerPS"]].values
Y = df[["price"]].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)


# ## Teil 2: Lineare Regression ausführen

# In[9]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


# In[10]:


print(model.score(X_test, y_test))


# 
