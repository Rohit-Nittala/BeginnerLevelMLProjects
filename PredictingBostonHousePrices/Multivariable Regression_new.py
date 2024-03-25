#!/usr/bin/env python
# coding: utf-8

# # Notebook imports

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

get_ipython().run_line_magic('matplotlib', 'inline')


# # Gather Data

# In[2]:


load_boston = fetch_openml(data_id=531, as_frame=True, parser='pandas')


# In[3]:


boston_dataset = load_boston


# In[4]:


type(boston_dataset)


# In[5]:


boston_dataset


# In[6]:


dir(boston_dataset)


# In[7]:


print(boston_dataset.DESCR)


# In[8]:


boston_dataset.data


# In[9]:


boston_dataset.data.shape


# In[10]:


boston_dataset.feature_names


# In[11]:


#Actual prices in thousands (000s)
boston_dataset.target


# # Data exploration using pandas Dataframes

# In[12]:


#Creating a pandas Dataframe
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
#Adding an column for house prices ( target )
data['PRICE'] = boston_dataset.target
#converting to float
data = data.astype(float)


# # Cleaning data - checking for missing values

# In[13]:


pd.isnull(data)


# # Visualization - Histograms, Barcharts and Distribution

# In[14]:


plt.figure(figsize=(10,6))
plt.hist(data['PRICE'], bins=50, ec='black')
plt.xlabel("Price of Houses in 1000$")
plt.ylabel("No. of Houses")
plt.show()


# In[15]:


plt.figure(figsize=(10,6))
sns.distplot(data['PRICE'], bins=50)
plt.show()


# In[16]:


plt.figure(figsize=(10,6))
plt.hist(data['RM'], ec='black', color='#00796b')
plt.xlabel("Average number of rooms")
plt.ylabel("No. of Houses")
plt.show()


# In[17]:


plt.figure(figsize=(10,6))
plt.hist(data['RAD'], bins=24, ec='black', color='#7b1fa2')
plt.xlabel("Accessibility to Radial Highways")
plt.ylabel("No. of Houses")
plt.show()


# In[18]:


data['RAD']


# In[19]:


data['RAD'].value_counts()


# In[20]:


frequency = data['RAD'].value_counts()
plt.figure(figsize=(10,6))
plt.bar(frequency.index, height=frequency)
plt.xlabel("Accessibility to Radial Highways")
plt.ylabel("No. of Houses")
plt.show()


# # Correlation 
# ## $$ \rho _(XY) = corr(X,Y) $$
# ## $$ -1.0 \leq \rho _(XY) \leq + 1.0 $$

# In[21]:


data['PRICE'].corr(data["RM"])


# In[22]:


data['PRICE'].corr(data["PTRATIO"])


# In[23]:


data.corr()


# In[24]:


mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
mask


# In[25]:


plt.figure(figsize=(16,10))
sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={"size":14})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


# In[26]:


nox_dis_corr = round(data['NOX'].corr(data['DIS']), 3)

plt.figure(figsize=(9,6))
plt.title(f'DIS vs NOX (correlation) {nox_dis_corr}', fontsize=14)
plt.scatter(x=data['DIS'], y= data['NOX'], alpha=0.6, s=80, color='indigo')
plt.xlabel('DIS - Distance from employment', fontsize=14)
plt.ylabel('NOX - Nitric Oxide Pollution', fontsize=14)
plt.show()


# In[27]:


sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(kind="scatter", x=data['DIS'], y=data['NOX'], size=7, color='indigo', joint_kws={'alpha':0.5})
plt.show()


# In[28]:


sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=data['TAX'], y=data['RAD'], size=7, color='darkred', joint_kws={'alpha':0.5})
plt.show()


# In[29]:


type(data)


# In[30]:


sns.lmplot(x='TAX', y='RAD', data=data, height=7)
plt.show()


# In[31]:


sns.lmplot(x='RM', y='PRICE', data=data, height=7)
plt.show()


# In[32]:


rm_dis_price = round(data['RM'].corr(data['PRICE']), 3)

plt.figure(figsize=(9,6))
plt.title(f'RM vs PRICE (correlation) {rm_dis_price}', fontsize=14)
plt.scatter(x=data['RM'], y= data['PRICE'], alpha=0.6, s=80, color='indigo')
plt.xlabel('RM - No of rooms', fontsize=14)
plt.ylabel('PRICE - Cost of the house in boston', fontsize=14)
plt.show()


# In[33]:


get_ipython().run_cell_magic('time', '', "\nsns.pairplot(data, kind='reg', plot_kws={'line_kws':{'color':'cyan'}})\nplt.show()\n")


# ## Training and testing dataset

# In[34]:


prices = data['PRICE']
features = data.drop('PRICE', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

# %of training data set
len(X_train)/len(features)


# # Multivariable Regression

# In[35]:


regr = LinearRegression()
regr.fit(X_train, y_train)

# Printing out values of r^2 of both traning and test dataset
print('Traning data r-squated', regr.score(X_train, y_train))
print('Test data r-squared', regr.score(X_test, y_test))
print('Intercept Values:',regr.intercept_)
pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef'])


# # Data Transformations

# In[36]:


data['PRICE'].skew()


# In[37]:


y_log = np.log(data['PRICE'])
y_log.head()


# In[38]:


sns.distplot(y_log)
plt.title(f'Log price with skew {y_log.skew()}')
plt.show()


# In[39]:


sns.lmplot(x='LSTAT', y='PRICE', data = data, height=7, scatter_kws={'alpha':0.6}, line_kws={'color':'darkred'})
plt.show()


# ## Regression using log prices
# 

# In[40]:


prices = np.log(data['PRICE'])
features = data.drop('PRICE', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)
regr = LinearRegression()
regr.fit(X_train, y_train)

# Printing out values of r^2 of both traning and test dataset
print('Traning data r-squated', regr.score(X_train, y_train))
print('Test data r-squared', regr.score(X_test, y_test))
print('Intercept Values:',regr.intercept_)
pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef'])


# In[41]:


#Charles River Property Premium
np.e**0.080331


# ## p-values and evaluating coefficients

# In[42]:


x_incl_constant = sm.add_constant(X_train)
model = sm.OLS(y_train, x_incl_constant)

results = model.fit()

pd.DataFrame({'coef':results.params, 'p-values':round(results.pvalues, 3)})


# # Testing for multicollinearity
# 
# $$ TAX = \alpha _0 + \alpha _1 RM+ \alpha _2 NOX + .... + \alpha _{12}LSTAT $$
# 
# $$ VIF _{TAX} = \frac{1}{( 1 - r _{TAX} ^ 2)}$$

# In[43]:


variance_inflation_factor(exog = x_incl_constant.values, exog_idx = 1)


# In[44]:


len(x_incl_constant.columns)
x_incl_constant.shape[1]


# In[45]:


vif = []
for i in range(x_incl_constant.shape[1]):
    vif.append(variance_inflation_factor(exog=x_incl_constant.values,
                                   exog_idx = i))
print(vif)


# In[46]:


pd.DataFrame({'coef_name': x_incl_constant.columns,
             'vif':np.around(vif,2)})


# ## Model Simplification and BIF
# 

# In[47]:


#original model with log prices and all features

x_incl_constant = sm.add_constant(X_train)
model = sm.OLS(y_train, x_incl_constant)

results = model.fit()

org_coef = pd.DataFrame({'coef':results.params, 'p-values':round(results.pvalues, 3)})
print("the BIC value of this model with all the features is:", results.bic)
print("the r^2 value of this model with all the features is:", results.rsquared)


# In[48]:


#Reduced model #1 excluding INDUS

x_incl_constant = sm.add_constant(X_train)
x_incl_constant = x_incl_constant.drop(['INDUS'], axis=1)



model = sm.OLS(y_train, x_incl_constant)

results = model.fit()

coef_minus_indus = pd.DataFrame({'coef':results.params, 'p-values':round(results.pvalues, 3)})
print("the BIC value of this model without INDUS feature is:", results.bic)
print("the r^2 value of this model withour INDUS feature is:", results.rsquared)


# In[49]:


#Reduced model #2 excluding INDUS and AGE

x_incl_constant = sm.add_constant(X_train)
x_incl_constant = x_incl_constant.drop(['INDUS', 'AGE'], axis=1)



model = sm.OLS(y_train, x_incl_constant)

results = model.fit()

coef_minus_indus_age = pd.DataFrame({'coef':results.params, 'p-values':round(results.pvalues, 3)})
print("the BIC value of this model without INDUS and AGE features is:", results.bic)
print("the r^2 value of this model withour INDUS and AGE features is:", results.rsquared)


# In[50]:


frames = [org_coef, coef_minus_indus, coef_minus_indus_age]
pd.concat(frames, axis=1
         )


# ## Residual and Residual plots

# In[51]:


#Modified model: transformed (using log prices) and simplified( removing 2 features)

prices = np.log(data['PRICE'])
features = data.drop(['PRICE', 'INDUS', 'AGE'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

#Using Statsmodel
x_incl_constant = sm.add_constant(X_train)
model = sm.OLS(y_train, x_incl_constant)
results = model.fit()

#Residuals
# residuals = y_train - results.fittedvalues
# results.resid

#Graph of actual vs predicted Prices
corr = round(y_train.corr(results.fittedvalues), 2)
plt.scatter(x=y_train, y=results.fittedvalues, c='navy', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.xlabel('Actual Log Prices $ y _i$', fontsize=14)
plt.ylabel('Predicted log prices $ \hat y _i$', fontsize=14)
plt.title(f'Actual vs Predicted house prices: $ y _i$ vs $ \hat y _1$ (corr {corr})', fontsize = 17)
plt.show()

#Graph with actual house prices and not the log values of it by raising the to the power of e
plt.scatter(x=np.e**y_train, y=np.e**results.fittedvalues, c='red', alpha=0.6)
plt.plot(np.e**y_train, np.e**y_train, color='cyan')
plt.xlabel('Actual Prices in Thousnads $ y _i$', fontsize=14)
plt.ylabel('Predicted prices in Thousands $ \hat y _i$', fontsize=14)
plt.title(f'Actual vs Predicted house prices: $ y _i$ vs $ \hat y _1$ (corr {corr})', fontsize = 17)
plt.show()

#Residuals vs Predicte values
plt.scatter(x=results.fittedvalues, y=results.resid, c='green', alpha=0.6)

plt.xlabel('Predicted log prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title('Residuals vs Fitted values', fontsize = 17)
plt.show()

#Mean Squared Error & Rsquared
reduced_log_mse = round(results.mse_resid, 3)
reduced_log_rsquared = round(results.rsquared, 3)


# In[52]:


# Distribution of Residual (Cheking for Normality)
resid_mean = round(results.resid.mean(), 5)
resid_skew = round(results.resid.skew(), 3)


sns.distplot(results.resid, color='navy')
plt.title(f'Log price model:Residual, Skew = {resid_skew} and Mean = {resid_mean}')
plt.show()


# In[53]:


#Original model: normal prices and all features

prices = data['PRICE']
features = data.drop(['PRICE'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)
x_incl_constant = sm.add_constant(X_train)
model = sm.OLS(y_train, x_incl_constant)
results = model.fit()


#Graph of actual vs predicted Prices
corr = round(y_train.corr(results.fittedvalues), 2)
plt.scatter(x=y_train, y=results.fittedvalues, c='indigo', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.xlabel('Actual Prices 000s $ y _i$', fontsize=14)
plt.ylabel('Predicted prices 000s $ \hat y _i$', fontsize=14)
plt.title(f'Actual vs Predicted prices: $ y _i$ vs $ \hat y _1$ (corr {corr})', fontsize = 17)
plt.show()

 

#Residuals vs Predicte values
plt.scatter(x=results.fittedvalues, y=results.resid, c='indigo', alpha=0.6)

plt.xlabel('Predicted log prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title('Residuals vs Fitted values', fontsize = 17)
plt.show()


# Residual Distribution chart
resid_mean = round(results.resid.mean(), 5)
resid_skew = round(results.resid.skew(), 3)


sns.distplot(results.resid, color='indigo')
plt.title(f'Residual Skew  ({resid_skew}) and Mean ({resid_mean})')
plt.show()

#Mean Squared  Error & Rsquared
full_normal_mse = round(results.mse_resid, 3)
full_normal_rsquared = round(results.rsquared, 3)


# In[54]:


pd.DataFrame({'R-Squared': [reduced_log_rsquared, full_normal_rsquared],
             'MSE': [reduced_log_mse, full_normal_mse],
             'RMSE': np.sqrt([reduced_log_mse,full_normal_mse])},
             index=['Reduced Log model', 'Full Normal Price model'])


# In[58]:


#Calculating: our estimate is 30,000 for a house, find out the upper and lower bounds 
# for a 95% prediction interval using the reduced log model
print('The 1 S.D in log prices is:', np.sqrt(reduced_log_mse))
print('The 2 S.D in log prices is:', 2*np.sqrt(reduced_log_mse))

upper_bound = np.log(30) + 2*np.sqrt(reduced_log_mse)
print('the upper bound in log prices for 95% prediction interval is:', upper_bound)
print('the upper bound in normal price is:', np.e**upper_bound *1000)

lower_bound = np.log(30) - 2*np.sqrt(reduced_log_mse)
print('the lower bound in log prices for 95% prediction interval is:', lower_bound)
print('the lower bound in normal price is:', np.e**lower_bound *1000)


# In[ ]:




