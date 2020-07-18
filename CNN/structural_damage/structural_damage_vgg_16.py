import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import Model, layers
from keras.models import load_model, model_from_json
import cv2
import os
import tensorflow as tf
import PIL
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.preprocessing import image

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/val'
image_size = 224
value_to_class = {0:"collapse",1:"crack",2:"dampening",3:"termite"}
test_dir = 'dataset/test/'
test_images_dir = 'dataset/test/'
nb_train_samples = 80
nb_validation_samples = 40
channels = 3

#create model with decorated resnetwithout final dense layer
#attach dense layer with 128 nodes and output layer with two nodes
def create_model_decorated_with_vgg16():
    conv_base = VGG16(weights = "imagenet", 
                       include_top=False,
                       input_shape = (image_size, image_size, channels))

    for layer in conv_base.layers:
        #set layers as non trainable
        layer.trainable = False
        
    x = conv_base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x) 
    x = layers.Dropout(0.5)(x)
    outputLayer = layers.Dense(4, activation='softmax')(x)
    model = Model(conv_base.input, outputLayer)

    #use adam as an optimizer
    optimizer = keras.optimizers.Adam()
    #use sparse_categorical_crossentropy as objective function
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy',ignore_accuracy_of_class(5)])
    
    return model


def generate_data():
    #this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        batch_size=32,
        class_mode='categorical',
        target_size=(image_size,image_size))

    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        shuffle=False,
        class_mode='categorical',
        target_size=(224,224))

    return (train_generator, validation_generator)



#evaluate model by running it on train and validation generator
#save model architecture and weights so that it could be used later on
def evaluate_model(model, train_generator, validation_generator):
    history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=nb_train_samples // 5,  # added in Kaggle
                              epochs=25,
                              validation_data=validation_generator,
                              validation_steps=nb_validation_samples,  # added in Kaggle
                              workers=12
                             )


    model.save('models/keras/vgg16_model_v1.h5')
    model.save_weights('models/keras/vgg16_weights_1.h5')
    with open('models/keras/vgg16_model_architecture_v1.json', 'w') as f:
        f.write(model.to_json())
        
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

def predict_labels(model, test_directory):
    BATCH_SIZE_TESTING = 1
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        directory = test_directory, 
        target_size = (image_size, image_size),
        batch_size = BATCH_SIZE_TESTING,
        class_mode = None,
        shuffle = False,
        seed = 123
    )  
    
    test_generator.reset()
    pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
    predicted_class_indices = np.argmax(pred, axis = 1)
    
    return (test_generator.classes, predicted_class_indices)

def plot_confusion_matrix(cm):
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.show()

def perform_validation(model):
    validation_img_paths = ["dataset/test/crack/created_images.jpg",
                            "dataset/test/crack/new_crack.jpg"]

    img_list = [Image.open(img_path) for img_path in validation_img_paths]

    validation_batch = np.stack([preprocess_input(np.array(img.resize((224,224))))
                                 for img in img_list])


    img_list = [Image.open(img_path) for img_path in validation_img_paths]

    pred_probs = model.predict(validation_batch)
    predicted_class_indices = np.argmax(pred_probs, axis = 1)    
    
    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        ax.set_title("{:.0f}% Crack, {:.0f}% Dampening".format(100*pred_probs[i,0],
                                                                100*pred_probs[i,1]))
        ax.imshow(img)
            
        plt.show()

def create_value_to_label_map(indices):
    label_map = indices
    for key in label_map:
        value_to_class[label_map[key]] = key
    
def get_label(label_value):
    label = value_to_class[label_value]
    return label
            
def visualize_filters(model):
    # summarize filter shapes
    for layer in model.layers:
        
	# check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # get filter weights

def load_pretrained_model():
    # load json and create model
    json_file = open('models/keras/vgg16_model_architecture_v1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/keras/vgg16_model_v1.h5")
    print("Loaded model from disk")
    return loaded_model

def evaluate_predictions(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm) 
    print("Accuracy Score :",accuracy_score(true_labels, predicted_labels))
    print("Report : ")
    print(classification_report(true_labels, predicted_labels))
    plot_confusion_matrix(cm)

def save_model(model,filePath):
        plot_model(model, to_file="vgg16.png", show_shapes=True, show_layer_names=True)

def predict(model):
    img = image.load_img('3.jpeg', target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    y_pred = model.predict(images, batch_size=10)
    print(y_pred)
    y_class = y_pred.argmax(axis=-1)
    print(y_pred[0][2])
    print (get_label(y_class[0]))

def ignore_accuracy_of_class(class_to_ignore=0):
    def ignore_acc(y_true, y_pred):
        y_true_class = keras.backend.argmax(y_true, axis=-1)
        y_pred_class = keras.backend.argmax(y_pred, axis=-1)

        ignore_mask = keras.backend.cast(keras.backend.not_equal(y_pred_class, class_to_ignore), 'int32')
        matches = keras.backend.cast(keras.backend.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = keras.backend.sum(matches) / keras.backend.maximum(keras.backend.sum(ignore_mask), 1)
        return accuracy

    return ignore_acc
 
# =============================================================================
# training_generator, validation_generator = generate_data()
# label_map = (validation_generator.class_indices)
# model = create_model_decorated_with_vgg16()
# history = evaluate_model(model, training_generator, validation_generator)
# summarize_diagnostics(history)
# create_value_to_label_map(training_generator.class_indices)
# =============================================================================
#perform_validation(model)
#visualize_filters(model)
loaded_model = load_pretrained_model()
true_labels, predicted_labels = predict_labels(loaded_model,test_dir)
evaluate_predictions(true_labels, predicted_labels)
#save_model(loaded_model,"vgg16.jpg")
predict(loaded_model)
