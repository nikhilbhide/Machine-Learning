#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:07:51 2020

@author: nik
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
from data_creation import generate_data_for_sum_operation

train_data, train_targets, test_data, test_targets = generate_data_for_sum_operation()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(20, activation=tf.nn.relu),
	keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae'])

model.fit(train_data, train_targets, epochs=50, batch_size=1)

test_loss, test_acc = model.evaluate(test_data, test_targets)
print('Test accuracy:', test_acc)
a= np.array([[2000,3000],[4,5]])
print(model.predict(a))

