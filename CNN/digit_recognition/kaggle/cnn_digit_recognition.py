
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
import pandas as pd 

# =============================================================================
# prepare dataset for model evaluation
# download training and test datasets along with labels
# perform normalization of the images
# 
# =============================================================================
def prepare_dataset():
    # the data, split between train and test sets
    df = pd.read_csv("train.csv")
    train=df.sample(frac=0.8,random_state=200) #random state is a seed value
    validation=df.drop(train.index)
    print(train.shape[0])
    
    y_train_df = train["label"]
    x_train_df = train.drop(labels = ["label"], axis = 1) 
    y_validation = validation["label"]
    x_validation = validation.drop(labels = ["label"], axis = 1) 

    print(type(y_train_df))
    
    x_train = x_train_df.to_numpy().reshape(x_train_df.shape[0],28,28,1)
    y_train = y_train_df.to_numpy()
    print(y_train.shape)
    x_validation = x_validation.to_numpy().reshape(x_validation.shape[0],28,28,1)
    y_validation = y_validation.to_numpy()
    
    
    return x_train,y_train,x_validation,y_validation
    
def prepare_test_dataset():
    # the data, split between train and test sets
    df = pd.read_csv("test.csv")
    print(df.shape)
    test = df.to_numpy().reshape(28000,28,28,1)
    
    return test

    
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
# This is two layer network with one full connection layer followed by output layer
# =============================================================================
def create_model(input_shape, num_shape, activation_function):
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation=activation_function),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation=activation_function),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation=activation_function),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128,activation=activation_function),
        layers.Dense(500,activation=activation_function),
        layers.Dense(num_classes, activation="softmax"),
    ]
    )

    return model

# =============================================================================
# create model for recognizing digit 
# select input shape as nh = 28 and nw = 28 with total pixes 784   
# This is two layer network with one full connection layer followed by output layer
# =============================================================================
def create_lenet5_model(input_shape, num_shape, activation_function):
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(12, kernel_size=(5, 5), activation=activation_function),
        layers.Dropout(0.5),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(5, 5), activation=activation_function),
        layers.Dropout(0.5),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(240,activation=activation_function),
        layers.Dropout(0.5),
        layers.Dense(168,activation=activation_function),
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
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.33,workers=-1,use_multiprocessing=True)
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
    epochs = 50
    
    return input_shape, num_classes, batch_size, epochs

def predict_by_index(model, indicesToPredict, x_test):
    for index in indicesToPredict:
        plt.imshow(x_test[index].reshape(28, 28),cmap='Greys')
        plt.show()
        pred = model.predict(x_test[index].reshape(1, 28, 28, 1))
        print(pred.argmax())

def evaluate_model_with_data_augmentation(model,x_train,y_train,x_test,y_test,num_classes,batch_size,epochs):
    from keras.preprocessing.image import ImageDataGenerator


    datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    fill_mode='nearest',
    validation_split = 0.2
    )

    datagen.fit(x_train)

    train_generator = datagen.flow(x_train, y_train, batch_size=60, subset='training')

    validation_generator = datagen.flow(x_train, y_train, batch_size=60, subset='validation')

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # fits the model on batches with real-time data augmentation:
    history = model.fit_generator(generator=train_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    steps_per_epoch = len(train_generator) / 60,
                    validation_steps = len(validation_generator) / 60,
                    epochs = 300,
                    verbose = 1,
                    workers=-1)
    
    score = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(score)*100, std(score)*100, len(score)))

    # list all data in history
    print(history.history.keys())

    summarize_diagnostics(history)

def plot_digits(X, Y):
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.tight_layout()
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title('Digit:{}'.format(Y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def predict_test_dataset():
    test = prepare_test_dataset()
    predictions = model.predict_classes(test, verbose=1)
    pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),
              "Label":predictions}).to_csv("submission.csv",
                                           index=False,
                                           header=True)

#get hyperparameters
input_shape, num_classes, batch_size, epochs = get_hyperparameters()
#prepare datasets
x_train,y_train,x_validation,y_validation = prepare_dataset()
#plot digits
plot_digits(x_train,y_train)                                           
#create model
model = create_model(input_shape,num_classes,'relu')
#evaluate model
evaluate_model(model,x_train,y_train,x_validation,y_validation,num_classes,batch_size,epochs)
#summarize model
model.summary()
#prediction
indices = [100,200,1040,5060,4502]
predict_by_index(model,indices,x_validation)


                                           
# import numpy as np
# train = np.genfromtxt('test.csv',delimiter=';',names=True,skip_header=1)
# y_train = train['label'] 
# print(train.shape)
# np.unique(train[:,1]
# y_train, x_train = np.split(train,[1])
# print(np.unique(train))
# x_test = np.genfromtxt('test.csv',delimiter=',',,skip_header=1)

