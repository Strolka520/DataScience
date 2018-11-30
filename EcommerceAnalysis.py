#!/usr/bin/env python
# coding: utf-8

# <font size="5">Import packages and get an initial feel for the dataset</font>

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


customers = pd.read_csv('Ecommerce Customers.csv')


# In[10]:


customers.head()


# In[11]:


customers.describe()


# In[12]:


customers.info()


# <font size="5">Exploratory Data Analysis</font>

# In[17]:


sns.jointplot(data=customers,x='Time on Website',y='Yearly Amount Spent')


# In[20]:


sns.jointplot(data=customers,x='Time on App',y='Yearly Amount Spent')


# In[21]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)


# In[22]:


sns.pairplot(customers)


# Length of Membership appears to have a positively correlated relationship with Yearly Amount Spent. Let's explore <br></br>
# the relationship further.

# In[24]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


# The longer a customer is a member, the higher the yearly amount spent. Next we will start on the model.

# <font size="5">Training and Testing Data</font>

# In[26]:


customers.columns


# In[27]:


y=customers['Yearly Amount Spent']


# In[28]:


X=customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[29]:


from sklearn.cross_validation import train_test_split


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# <font size="5">Training the Model</font>

# In[32]:


from sklearn.linear_model import LinearRegression


# In[33]:


lm = LinearRegression()


# In[34]:


lm.fit(X_train,y_train)


# In[35]:


lm.coef_


# In[36]:


predictions = lm.predict(X_test)


# In[37]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')


# <font size="5">Evaluating the Model</font>

# In[40]:


from sklearn import metrics


# In[41]:


#Mean Absolute Error
print('MAE ', metrics.mean_absolute_error(y_test,predictions))

#Mean Squared Error
print('MSE ', metrics.mean_squared_error(y_test,predictions))

#Mean Absolute Error
print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[43]:


#R squared
metrics.explained_variance_score(y_test,predictions)


# In[46]:


#This model is fit very well especially with the test data


# <font size="5">Residuals</font>

# In[48]:


sns.distplot((y_test-predictions),bins=50)


# In[49]:


#The distribution is normal


# <font size="5">Conclusion</font>

# In[50]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
cdf

Holding all other features fixed, the coefficients can be interpreted as:<br></br>
    
    One unit increase of Avg. Session Length is associated with about 26 more dollars spent<br></br>
    One unit increase of Time on App is associated with about 39 more dollars spent<br></br>
    One unit increase of Time on Website is associated with about 0.19 more dollars spent<br></br> 
    One unit increase of Length of Membership is associated with about 61 more dollars spent<br></br> 
    
From here, the business could focus on keeping, maintaining, and cultivating long term members.<br></br>
Additionally, the business should weigh other factors on determining whether to focus on developing<br></br>
the app further since it is making good money or developing the website further in order to get<br></br>
it closer to the other categories.
# 
