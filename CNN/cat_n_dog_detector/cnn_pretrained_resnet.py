import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json
import cv2
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import PIL

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'
image_size = 224

train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    batch_size=32,
    class_mode='binary',
    target_size=(224,224))

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    shuffle=False,
    class_mode='binary',
    target_size=(224,224))

conv_base = ResNet50(
    include_top=False,
    weights='imagenet')

for layer in conv_base.layers:
    layer.trainable = False
    
x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x) 
predictions = layers.Dense(2, activation='softmax')(x)
model = Model(conv_base.input, predictions)


optimizer = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=347 // 32,  # added in Kaggle
                              epochs=10,
                              validation_data=validation_generator,
                              validation_steps=10,  # added in Kaggle
                              workers=8
                             )


model.save('models/keras/resnet_model.h5')
model.save_weights('models/keras/resnet_weights.h5')
with open('models/keras/resnet_architecture.json', 'w') as f:
        f.write(model.to_json())
        
new_model = load_model('models/keras/resnet_model.h5')
        
# load
with open('models/keras/resnet_architecture.json') as f:
    new_model = model_from_json(f.read())
new_model.load_weights('models/keras/resnet_weights.h5')


validation_img_paths = ["dataset/test/test/cat.4001.jpg",
                        "dataset/test/test/dog.26.jpg",
                        "dataset/test/test/dog.95.jpg",
                        "dataset/test/test/cat.4057.jpg"]

img_list = [Image.open(img_path) for img_path in validation_img_paths]


validation_batch = np.stack([preprocess_input(np.array(img.resize((224,224))))
                             for img in img_list])


img_list = [Image.open(img_path) for img_path in validation_img_paths]

pred_probs = model.predict(validation_batch)
pred_probs


fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title("{:.0f}% Alien, {:.0f}% Predator".format(100*pred_probs[i,0],
                                                            100*pred_probs[i,1]))
    ax.imshow(img)
    
# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1
    
test_generator = validation_datagen.flow_from_directory(
    directory = 'dataset/test_images/classes',
    target_size = (image_size, image_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
)  

test_generator.reset()

pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

predicted_class_indices = np.argmax(pred, axis = 1)

cm = confusion_matrix(test_generator.classes, predicted_class_indices)
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

print(cm) 
print("Accuracy Score :",accuracy_score(test_generator.classes, predicted_class_indices))
print("Report : ")
print(classification_report(test_generator.classes, predicted_class_indices))

# or
#cm = np.array([[1401,    0],[1112, 0]])

plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

TEST_DIR = 'dataset/test/test'
f, ax = plt.subplots(5, 5, figsize = (15, 15))

for i in range(0,25):
    imgBGR = cv2.imread(TEST_DIR + test_generator.filenames[i])
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    
    # a if condition else b
    predicted_class = "Dog" if predicted_class_indices[i] else "Cat"

    ax[i//5, i%5].imshow(imgRGB)
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title("Predicted:{}".format(predicted_class))    

plt.show()

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


def generate_output_csv_label_by_filename():
    for counter in range (0,test_generator.filenames.len):
        print(test_generator.filenames[counter])
        
summarize_diagnostics(history)        