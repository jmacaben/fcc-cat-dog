# Each section is split according to the respective cells in the notebook provided by freeCodeCamp
# Some of the given code may have been changed for use outside of a notebook environment

# Cell 1 (given)
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import zipfile
import requests

# Cell 2 (given)
# Get project files
url = 'https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip'
response = requests.get(url)

with open('cats_and_dogs.zip', 'wb') as file:
    file.write(response.content)

with zipfile.ZipFile('cats_and_dogs.zip', 'r') as zip_ref:
    zip_ref.extractall()

PATH = 'cats_and_dogs'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Get the number of files in each directory
# Each have the subdirectories "dogs" and "cats".
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# Variables for pre-processing and training
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Cell 3
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(train_dir, batch_size=batch_size, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(validation_dir, batch_size=batch_size, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

# flow_from_directory expects subdirectories for each class
# From looking at the directory structure, we see that the test data is not in subdirectories
# So, we will create a subdirectory called "unknown" (if it doesn't already exist) and move all the test images into it
test_subdir = os.path.join(test_dir, 'unknown')
if not os.path.exists(test_subdir):
    os.makedirs(test_subdir)
    for fname in os.listdir(test_dir):
        fpath = os.path.join(test_dir, fname)
        if os.path.isfile(fpath):
            os.rename(fpath, os.path.join(test_subdir, fname))

test_data_gen = test_image_generator.flow_from_directory(test_dir, batch_size=1, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode=None, shuffle=False)

# Cell 4 (given)
def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "%% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "%% cat")
    plt.show()

sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])

# Cell 5
# Add 4-6 random transformations as arguments to ImageDataGenerator
# Include rescaling the same as before
train_image_generator = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Cell 6 (given)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)

# Cell 7
# Create a model for the neural network that outputs class probabilities
# Involve a stack of Conv2D and MaxPooling2D layers and then a fully connected layer on top that is activated by a ReLU activation function
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') 
])

# Compile the model passing the arguments to set the optimizer and loss
# Pass in metrics=['accuracy'] to view training and validation accuracy for each training epoch
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Cell 8
# Use the fit method on your model to train the network
# Make sure to pass in arguments for x, steps_per_epoch, epochs, validation_data, and validation_steps
history = model.fit(
    train_data_gen,  
    steps_per_epoch=total_train // batch_size,  
    epochs=epochs, 
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

# Cell 9 (given)
# Visualize the accuracy and loss of the model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()