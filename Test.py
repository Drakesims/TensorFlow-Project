import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import cv2
import os
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


Model_Path = "dog_breed_classifier.h5"

dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
data_dir = tf.keras.utils.get_file('stanford_dogs.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')
dogs_dir = data_dir / "Images"
image_count = len(list(data_dir.glob('*/**/*.jpg')))
print(image_count)

img_height, img_width = 120, 120
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    dogs_dir,
    validation_split = 0.3,
    subset="training",
    seed=123,
    label_mode = 'categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dogs_dir,
  validation_split=0.3,
  subset="validation",
  seed=123,
  label_mode = 'categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)

num_classes = len(class_names)
'''
if os.path.exists(Model_Path):
    print("Loading presaved model")
    resnet_model = load_model(Model_Path)
    resnet_model.compile(optimizer=Adam(learning_rate=0.001), 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])
                        
    loss, acc = resnet_model.evaluate(val_ds, verbose=0)
    print(f" Model evaluation after loading - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    print("Model has been loaded and compiled")
else:'''
print("Training model")

resnet_model = Sequential()

pretrained_model = tf.keras.applications.ResNet50(include_top = False,
                                                input_shape = (img_height,img_width,3),
                                                pooling = 'avg',
                                                classes = num_classes,
                                                weights = 'imagenet')
for layer in pretrained_model.layers:
    layer.trainable = False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(256, activation = 'relu'))
resnet_model.add(Dense(num_classes, activation = 'softmax'))

resnet_model.summary()

resnet_model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

epochs=15
history = resnet_model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs
)

resnet_model.save(Model_Path)
print("Model Saved As", Model_Path)

plot1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4, ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train','validation'])
plt.show()

plot2 = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train','validation'])
plt.show()

def predict_dog_breed(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (img_height, img_width))
    image = np.expand_dims(image_resized, axis=0)

    pred = resnet_model.predict(image)
    output_class = class_names[np.argmax(pred)]
    
    print(f"I predict this is a {output_class}")

user_image = input("Enter the path of the image you want to predict: ")
if os.path.exists(user_image):
    predict_dog_breed(user_image)
else:
    print("Invalid file path.")