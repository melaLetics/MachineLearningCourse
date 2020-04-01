#!/usr/bin/env python
# coding: utf-8

# What is the predicted price of a car with 50000 kilometres?

# In[1]:


import pandas as pd

df = pd.read_csv("./autos_prepared.csv")
df.head()


# In[2]:


from sklearn.model_selection import train_test_split

X = df[["kilometer"]].values
Y = df[["price"]].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)


# In[3]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(df[["kilometer"]], df[["price"]])


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test, color = "red")
plt.show()


# In[5]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept: " + str(model.intercept_))
print("Coef: " + str(model.coef_))


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

predicted = model.predict(X_test)

plt.scatter(X_test, y_test, color = "red")
plt.plot(X_test, predicted)
plt.show()


# In[ ]:




