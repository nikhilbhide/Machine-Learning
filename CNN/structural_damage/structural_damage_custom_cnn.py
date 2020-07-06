#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:57:04 2020

@author: nik
"""


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Activation
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.preprocessing import image

# dimensions of our images.
img_width, img_height = 224, 224
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/val'
nb_train_samples = 80
nb_validation_samples = 40
epochs = 50
batch_size = 5

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    class_mode='categorical')


history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/5 ,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples,
    workers=8)


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def predict(model, filepath):
    img = image.load_img(filepath, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    y_pred = model.predict(images, batch_size=10)
    print(y_pred)
    y_class = y_pred.argmax(axis=-1)
    
    print (y_class)
    
summarize_diagnostics(history)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
predict(model,'dampening.jpeg')