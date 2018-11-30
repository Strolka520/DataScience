#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv('USA_Housing.csv')


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.columns


# In[10]:


sns.pairplot(df)


# In[63]:


sns.distplot(df['Price'])


# In[13]:


sns.heatmap(df.corr(),annot=True)


# In[14]:


df.columns


# In[40]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[41]:


y = df['Price']


# In[42]:


from sklearn.cross_validation import train_test_split


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


lm = LinearRegression()


# In[46]:


lm.fit(X_train,y_train)


# In[47]:


print(lm.intercept_)


# In[48]:


lm.coef_


# In[49]:


X_train.columns


# In[51]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[52]:


cdf


# In[53]:


from sklearn.datasets import load_boston


# In[54]:


boston = load_boston()


# In[56]:


boston.keys()


# In[60]:


print(boston['target'])


# Predictions

# In[64]:


predictions = lm.predict(X_test)


# In[65]:


predictions


# In[66]:


plt.scatter(y_test,predictions)


# In[68]:


sns.distplot((y_test-predictions))


# In[69]:


from sklearn import metrics


# In[72]:


metrics.mean_absolute_error(y_test,predictions)


# In[73]:


metrics.mean_squared_error(y_test,predictions)


# In[74]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))


# In[ ]:




