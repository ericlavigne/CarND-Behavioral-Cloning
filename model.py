# create and train the model

import argparse
import base64
import json

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import cv2

import time
from PIL import Image
from PIL import ImageOps

from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D

import tensorflow as tf
tf.python.control_flow_ops = tf # mysterious fix to keras/tensorflow issue

# Image processing snippets from drive.py
# image = Image.open(BytesIO(base64.b64decode(imgString)))
# image_array = np.asarray(image)
# transformed_image_array = image_array[None, :, :, :]

# Model prediction snippet from drive.py
# steering_angle = float(model.predict(transformed_image_array, batch_size=1))

# Later I'll probably add speed as an input variable and
# acceleration as an output variable.  This will involve
# a small change to drive.py around line 44.
# steering_angle = ...

# Does the simulator support negative throttle? This is needed both
# for braking at barricades in map 2, as well as for 3-point turns
# that I'd like to try later. Will start by changing throttle = 0.2
# to throttle = -0.2 to check if that works. If not, will need to
# ask Udacity staff if simulator can be upgraded.

default_data_dir = '../CarND-Simulator'

steering_bins = [-1, -0.4, -0.15, 0.0, 0.15, 0.4, 1]
throttle_bins = [-1, 0, 1]

def convert_steer_angle_to_bin(angle):
  best_bin = None
  best_diff = 10
  for i,bin_angle in enumerate(steering_bins):
    diff = abs(angle - bin_angle)
    if diff < best_diff:
      best_diff = diff
      best_bin = i
  return best_bin

# import model as m
# m.load_image('/Users/ericlavigne/workspace/CarND-Simulator/IMG/center_2017_01_21_19_10_57_316.jpg')

def load_image(file_name):
  img = mpimg.imread(file_name)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  return img

# import model as m; m.load_summary_data('../CarND-Simulator').head(5)
# Result is DataFrame: http://pandas.pydata.org/pandas-docs/stable/api.html#dataframe

def load_summary_data(data_dir):
  file_name = data_dir + '/driving_log.csv'
  col_names = ['img_center','img_left','img_right','steer','throttle','brake','speed']
  df = pd.read_csv(file_name, names=col_names)
  df['steer_bin'] = df['steer'].apply(lambda angle: convert_steer_angle_to_bin(angle))
  df.drop('img_left', 1, inplace=True)
  df.drop('img_right', 1, inplace=True)
  return df

# sample = m.load_sample(m.default_data_dir)
# 4 seconds to load sample of 1000 - not bad :-)

def load_sample(data_dir, sample_size=10):
  df = load_summary_data(data_dir)
  df = df.sample(sample_size)
  df['img'] = df['img_center'].apply(lambda file_name: load_image(file_name))
  return df

def create_model():
  model = Sequential()
  # Convolution2D(output_depth, convolution height, convolution_width, ...)
  model.add(Convolution2D(10, 5, 5, border_mode='valid', input_shape=(160,320,3)))
  # model.add(Dropout(0.5)
  model.add(Flatten())
  model.add(Dense(32))
  model.add(Dense(len(steering_bins), activation='softmax'))
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

def train_model(model, data_dir):
  pass

# Saving and loading keras models
# https://keras.io/models/about-keras-models/

# model.summary()
# model.get_config()
# model = Model.from_config(config)
  
def save_model(model,path='model'):
  with open(path + '.json', 'w') as arch_file:
    arch_file.write(model.to_json())
  model.save_weights(path + '.h5')

def load_model(path):
  with open(path + '.json', 'r') as arch_file:
    model = model_from_json(arch_file.read())
    model.load_weights(path + '.h5')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimDrive Training')
    parser.add_argument('load_model', type=str,
                        help='Path to model definition for loading (without json/h5 extension)')
    parser.add_argument('save_model', type=str,
                        help='Path to model definition for saving (without json/h5 extension)')
    parser.add_argument('training_data', type=str,
                        help='Path to folder with driving_log.csv and IMG subfolder')
    args = parser.parse_args()

    print(args.model)    
