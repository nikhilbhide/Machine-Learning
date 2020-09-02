#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:28:44 2020

@author: nik
"""


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from data_creation import generate_data_for_sum_operation, generate_data_for_pythagorus_operation

x_train, y_train, x_test, y_test = generate_data_for_pythagorus_operation()
regressor = LinearRegression()  
regressor.fit(x_train, y_train) #training the algorithm

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

y_pred = regressor.predict(x_test)

print(y_pred)
print(regressor.predict([[2000,3000],[4,5],[3,4]]))