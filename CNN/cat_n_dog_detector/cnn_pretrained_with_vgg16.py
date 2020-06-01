#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 10:21:51 2020

@author: nik
"""

from keras.models import Model
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
from matplotlib.pyplot import imshow
import numpy as np
import json
import matplotlib.pyplot as plt

#create global variables
value_to_class = {}
img_width, img_height = 150, 150
num_channels = 3
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 25
batch_size = 16

#create model with decorated vgg 16 without final dense layer
#attach dense layer with 256 nodes and output layer with one node
def create_model_decorated_with_vgg16():
    #instantiate VGG16 model without final dense layer
    base_model = VGG16(weights = "imagenet", include_top=False, input_shape = (150, 150, 3))
    #make layer of the non trainable
    for layer in base_model.layers:
        layer.trainable = False    
    #summarize the base model
    base_model.summary()
    
    # Adding custom layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)
    #create final model    
    top_model = Model(inputs=base_model.input, outputs=output)
    model = Sequential(top_model.layers)
    model.summary()

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    return model

def generate_data():
    #this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    
    return (train_generator, validation_generator)

#evaluate model 
def evaluate_model(model, train_generator, validation_generator):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('model_cat_dog_detector_vgg16_pretrained.h5')
    model_json = model.to_json()

    with open("model_cat_dog_detector_vgg16_pretrained_in_json.json", "w") as json_file:
        json.dump(model_json, json_file)
    
    return history    

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

def predict_images(model):
    test_image = image.load_img('dataset/test1/49.jpg',target_size=(150,150))
    imshow(test_image)
    test_image=image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    pred = model.predict(test_image)
    print(pred)
    classes = model.predict_classes(test_image)
    get_label(classes)
    print(classes)
    pred_prob= model.predict_proba(test_image)
    print(pred_prob)
    
    test_image = image.load_img('dataset/test1/45.jpg',target_size=(150,150))
    imshow(test_image)
    test_image=image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    pred = model.predict(test_image)
    print(pred)
    classes = model.predict_classes(test_image)
    print(classes)
    pred_prob= model.predict_proba(test_image)
    print(pred_prob)
    
    test_image = image.load_img('dataset/test1/45.jpg',target_size=(150,150))
    imshow(test_image)
    test_image=image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    pred = model.predict(test_image)
    print(pred)
    classes = model.predict_classes(test_image)
    print(classes[0][0])
    pred_prob= model.predict_proba(test_image)
    print(pred_prob)

def create_value_to_label_map(indices):
    label_map = indices
    for key in label_map:
        value_to_class[label_map[key]] = key
    
def get_label(label_value):
    label = label_map[label_value]
    print(label)
    return label
    

training_generator, validation_generator = generate_data()
label_map = (validation_generator.class_indices)
model = create_model_decorated_with_vgg16()
history = evaluate_model(model, training_generator, validation_generator)
summarize_diagnostics(history)
create_value_to_label_map(training_generator.class_indices)
predict_images(model)
    

