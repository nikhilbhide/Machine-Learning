
"""
Created on Sun May 24 16:16:09 2020

@author: nik
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std

# =============================================================================
# prepare dataset for model evaluation
# download training and test datasets along with labels
# perform normalization of the images
# 
# =============================================================================
def prepare_dataset():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    return x_train, y_train, x_test, y_test


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

# =============================================================================
# create model for recognizing digit 
# select input shape as nh = 28 and nw = 28 with total pixes 784   
# =============================================================================
def create_model(input_shape, num_shape):
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
    )

    return model

def evaluate_model(model, x_train, y_train,x_test,y_test,num_classes,batch_size,epochs):
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.33)
    score = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(score)*100, std(score)*100, len(score)))

    # list all data in history
    print(history.history.keys())

    summarize_diagnostics(history)


def get_hyperparameters():
    # input shape
    input_shape = (28, 28, 1)
    # model / data parameters
    num_classes = 10
    # size of the batch to be used for gradient update
    batch_size = 128
    # num of epochs
    epochs = 15
    
    return input_shape, num_classes, batch_size, epochs

def predict_by_index(model, indicesToPredict, x_test):
    for index in indicesToPredict:
        plt.imshow(x_test[index].reshape(28, 28),cmap='Greys')
        plt.show()
        pred = model.predict(x_test[index].reshape(1, 28, 28, 1))
        print(pred.argmax())

    
input_shape, num_classes, batch_size, epochs = get_hyperparameters()
x_train,y_train,x_test,y_test = prepare_dataset()
model = create_model(input_shape,num_classes)
evaluate_model(model,x_train,y_train,x_test,y_test,num_classes,batch_size,epochs)
model.summary()
indices = [100,200,1040,5060,4502]
predict_by_index(model,indices,x_test)