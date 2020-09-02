#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:10:09 2020

@author: nik
"""


import numpy as np 
from math import sqrt

def generate_data_for_sum_operation():
    train_data = np.array([[1.0,1.0]])
    train_targets = np.array([2.0])
    print(train_data)
    for i in range(3,10000,2):
        train_data= np.append(train_data,[[i,i]],axis=0)
        train_targets= np.append(train_targets,[i+i])
    test_data = np.array([[2.0,2.0]])
    test_targets = np.array([4.0])
    for i in range(4,8000,4):
        test_data = np.append(test_data,[[i,i]],axis=0)
        test_targets = np.append(test_targets,[i+i])
        
    return train_data, train_targets,test_data, test_targets

def pythagorus(x,y):
    return sqrt(x**2 +y**2)
    
def generate_data_for_pythagorus_operation():
    train_data = np.array([[1.0,1.0]])
    train_targets = np.array([pythagorus(1.0,1.0)])
    
    for i in range(3,10000,2):
        train_data= np.append(train_data,[[i,i]],axis=0)
        train_targets= np.append(train_targets,[pythagorus(i,i)])
    test_data = np.array([[2.0,2.0]])
    test_targets = np.array([4.0])
    for i in range(4,8000,4):
        test_data = np.append(test_data,[[i,i]],axis=0)
        test_targets = np.append(test_targets,[pythagorus(i,i)])
    
    return train_data, train_targets,test_data, test_targets
