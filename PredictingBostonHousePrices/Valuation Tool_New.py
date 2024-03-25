#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
 
from sklearn.datasets import fetch_openml
load_boston = fetch_openml(data_id=531, as_frame=True, parser='pandas')
 



# In[3]:


boston_dataset = load_boston
data = pd.DataFrame(data=boston_dataset.data, dtype=np.float64, columns=boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)
features.head()

data['PRICE'] = boston_dataset.target
log_prices = np.log1p(data['PRICE'])
target = pd.DataFrame(log_prices, columns=['PRICE'])
target.shape


# In[4]:


CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

property_stats = features.mean().values.reshape(1,11)
property_stats


# In[5]:


regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features) 

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)
RMSE


# In[6]:


def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river=False,
                    high_confidence=True):
    
    # Configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    
    # Make prediction
    log_estimate = regr.predict(property_stats)[0][0]
     
    # Calc Range
    if high_confidence:
        # Do X
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95       
    else:
        # Do Y
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    
    return log_estimate, upper_bound, lower_bound, interval


# In[7]:


get_log_estimate(3, 20)


# In[8]:


np.median(boston_dataset.target)


# In[9]:


ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)
log_est, upper, lower, conf = get_log_estimate(9, students_per_classroom=15, 
                                              next_to_river=False, high_confidence=False)

#Convert to today's dollars

dollar_est = np.e**log_est * SCALE_FACTOR * 1000
dollar_hi = np.e**upper * SCALE_FACTOR * 1000
dollar_low = np.e**lower * SCALE_FACTOR * 1000



#Round the dollar values to the nearest thousandth
rounded_est = np.around(dollar_est, -3)
rounded_hi = np.around(dollar_hi, -3)
rounded_low = np.around(dollar_low, -3)

print(f'The estimated property value is {rounded_est}.')
print(f'At {conf}% confidence the valuation range is ')
print(f'USD {rounded_low} at the lower end to USD {rounded_hi} at the high end. ')


# In[10]:


def get_dollar_estimate(rm, ptratio, chas=False, large_range=True ):
    """ Estimate the Price of a property in Boston
    
    Keyword Arguments:
    rm -- No of rooms in the property
    ptratio -- No of students per teacher in the classroom for a school in the area
    chas -- True if the property is next to the river , False otherwise
    large_range --  True for a 95% prediction interval, False for a 68% interval
    
    
    """
    
    
    
    
    log_est, upper, lower, conf = get_log_estimate(rm, 
                                                   students_per_classroom = ptratio, 
                                                  next_to_river=chas,
                                                   high_confidence=large_range)
    if rm < 1 or ptratio < 1:
        print('This is unrealistic. Try again.')
        return 

    #Convert to today's dollars

    dollar_est = np.e**log_est * SCALE_FACTOR * 1000
    dollar_hi = np.e**upper * SCALE_FACTOR * 1000
    dollar_low = np.e**lower * SCALE_FACTOR * 1000



    #Round the dollar values to the nearest thousandth
    rounded_est = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_hi, -3)
    rounded_low = np.around(dollar_low, -3)

    print(f'The estimated property value is {rounded_est}.')
    print(f'At {conf}% confidence the valuation range is ')
    print(f'USD {rounded_low} at the lower end to USD {rounded_hi} at the high end. ')


# In[11]:


get_dollar_estimate(rm=5, ptratio=10, chas=True)


# In[ ]:




