#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


sns.set_style('darkgrid')


# In[111]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[112]:


df_train.shape


# In[5]:


df_train.head()


# In[6]:


df_train.info()


# In[8]:


df_train.describe()


# In[9]:


df_train.columns


# In[58]:


sns.distplot(df_train['SalePrice'])


# In[12]:


df_train.corr()


# In[59]:


f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(df_train.corr(), annot = True,linewidths=.2, fmt='.1f', ax=ax)


# In[ ]:





# In[ ]:





# In[101]:


df_train['SalePrice'].describe()


# In[ ]:





# In[100]:


sns.pairplot(df_train,x_vars=['TotalBsmtSF','GrLivArea','GarageArea','OverallQual','FullBath'],
             y_vars=['SalePrice'],kind='reg')


# In[109]:


df_train.columns


# In[113]:


X = df_train[['TotalBsmtSF','GrLivArea','GarageArea','OverallQual','FullBath']]


# In[114]:


y = df_train['SalePrice']


# In[123]:


#from sklearn.cross_validation import train_test_split 
from sklearn.model_selection import train_test_split


# In[124]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[125]:


from sklearn.linear_model import LinearRegression


# In[126]:


lm = LinearRegression()


# In[127]:


lm.fit(X_train,y_train)


# In[128]:


print(lm.intercept_)


# In[130]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[131]:


predictions = lm.predict(X_test)


# In[135]:


plt.scatter(y_test,predictions)


# In[143]:


sns.distplot((y_test-predictions),bins=35)


# In[139]:


from sklearn import metrics


# In[140]:


#Mean Absolute Error 
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

#Mean Squared Error
print('MSE:', metrics.mean_squared_error(y_test, predictions))

#Root Mean Squared Error
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




