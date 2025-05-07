# Each section is split according to the respective cells in the notebook provided by freeCodeCamp

# Cell 1
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import os
import zipfile
import requests

# Cell 2 
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


