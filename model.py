
# coding: utf-8

# Model.py
# JW / 12.8.17(mod.19.9.17)

# In[1]:


## Model.py
## Behavioral Cloning Project
## JW / 12.8.17(mod.19.9.17)

from tqdm import tqdm_notebook
import os
import pandas as pd
import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, MaxPooling2D, Convolution2D, Lambda, ELU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Activation
import tensorflow as tf
tf.python.control_flow_ops = tf

epochs = 20
batch_size = 128
dataset_dir = "/Users/jussi/Desktop/CarND/data_nd"
image_columns = 32
image_rows = 16
image_channels = 1
side_shift = 0.3


def preproccess_image(image):
    image = (cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[:, :, 1]
    image = image.reshape(160, 320, 1)
    image = cv2.resize(image, (image_columns, image_rows))
    return image


def prepare(data):
    x, y = [], []

    for i in range(len(data)):
        line_data = data.iloc[i]
        y_steer = line_data['steering']
        path_center = line_data['center'].strip()
        path_left = line_data['left'].strip()
        path_right = line_data['right'].strip()

        for path, shift in [(path_center, 0), (path_left, side_shift), (path_right, -side_shift)]:
            # read image
            image_path = os.path.join(dataset_dir, path)
            image = cv2.imread(image_path)

            # preprocess image
            image = preproccess_image(image)

            # add image
            x.append(image)
            y.append(y_steer + shift)

            # add flipped image
            image = image[:, ::-1]
            x.append(image)
            y.append(-(y_steer + shift))

    return np.array(x).astype('float32'), np.array(y).astype('float32')


# comma.ai

def model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(image_rows, image_columns, image_channels)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu', name='Conv1'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv2'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv3'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512, activation='elu', name='FC1'))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1, name='output'))
    model.summary()
    
    return model
    

if __name__ == '__main__':
    print("Loading images...")

    data = pd.read_csv(os.path.join(dataset_dir, "/Users/jussi/Desktop/CarND/data_nd/driving_log.csv"))

    X_train, y_train = prepare(data)
    X_train, y_train = shuffle(X_train, y_train)
    X_train = np.expand_dims(X_train, axis=3)

    model = model()
    model.compile('adam', 'mean_squared_error', ['mean_squared_error'])
    checkpoint = ModelCheckpoint("model.h5", monitor='val_mean_squared_error', verbose=1,
                                  save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4,
                                verbose=1, mode='min')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1,
                      callbacks=[checkpoint, early_stop], validation_split=0.15, shuffle=True)


model.save('model.h5')
del model  # deletes the existing model

model = load_model('model.h5')

print("Saved model to disk")

