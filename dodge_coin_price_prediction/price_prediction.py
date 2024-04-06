# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:17:55 2024

@author: Reza
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor 

data = pd.read_csv("../datasets/dodge-usd.csv") 
data.head() 

data['Date'] = pd.to_datetime(data['Date'], 
							infer_datetime_format=True) 
data.set_index('Date', inplace=True) 

data.isnull().any() 

# Drop the null values from the dataframe
data = data.dropna()

data["gap"] = (data["High"] - data["Low"]) * data["Volume"] 
data["y"] = data["High"] / data["Volume"] 
data["z"] = data["Low"] / data["Volume"] 
data["a"] = data["High"] / data["Low"] 
data["b"] = (data["High"] / data["Low"]) * data["Volume"] 



df2 = data.tail(30) 
train = df2[:11] 
test = df2[-19:] 


from statsmodels.tsa.statespace.sarimax import SARIMAX 
model = SARIMAX(endog=train["Close"], exog=train.drop( 
	"Close", axis=1), order=(2, 1, 1)) 
results = model.fit() 
print(results.summary()) 
