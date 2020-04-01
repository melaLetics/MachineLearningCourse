#!/usr/bin/env python
# coding: utf-8

# What is the predicted price of a car with 50000 kilometres?

# In[1]:


import pandas as pd

df = pd.read_csv("./autos_prepared.csv")


# In[2]:


df.head()


# In[3]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(df[["kilometer"]], df[["price"]])


# In[4]:


print("Intercept: " + str(model.intercept_))
print("Coef: " + str(model.coef_))


# In[5]:


print(model.predict([[50000]]))


# In[ ]:




