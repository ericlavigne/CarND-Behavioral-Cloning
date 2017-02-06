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

from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2

from math import ceil
from random import random

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

steering_bins = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
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

def convert_bin_to_steer_angle(bin):
  return steering_bins[bin]

def bin_probabilities_to_angle(bins):
  print("bins: " + str(bins))
  weighted_bins = [x for x in bins]
  weighted_total = sum(weighted_bins)
  threshold = random() * weighted_total
  #print("random: " + str(r) + " / " + str(total_prob))
  weight_so_far = 0
  for i in range(len(bins)):
    weight_so_far += weighted_bins[i]
    if weight_so_far >= threshold:
      angle = convert_bin_to_steer_angle(i)
      #print("selected " + str(i) + " for angle " + str(angle))
      print("selected " + str(angle) + " (" + str(i) + ") from " + str(bins))
      return angle

# import model as m; m.load_image('/Users/ericlavigne/workspace/CarND-Simulator/IMG/center_2017_01_21_19_10_57_316.jpg')

def load_image(file_name):
  img = mpimg.imread(file_name)
  return convert_image_to_input_format(img)

def convert_image_to_input_format(original):
  img = original
  #print("Image has shape " + str(img.shape))
  #img = cv2.resize(img, (320, 160), interpolation=cv2.INTER_AREA)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  img = (img / 255) - 0.5
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

# import model as m; sample = m.load_sample(m.default_data_dir)
# import model as m; sample = m.load_sample(m.default_data_dir, sample_filter='validation')
# import model as m; sample = m.load_sample(m.default_data_dir, sample_filter='training')
# 4 seconds to load sample of 1000 - not bad :-)

def load_sample(data_dir, sample_size=10, sample_filter='all'):
  df = load_summary_data(data_dir)
  # Chunks of 100 frames (~10 seconds) set aside for validation
  df['index'] = df.index
  if sample_filter == 'validation':
    df = df[((df['index'] // 100) % 10) == 0]
  if sample_filter == 'training':
    df = df[((df['index'] // 100) % 10) != 0]
  # Minority oversampling - all steer bins are equally represented
  num_bins = len(steering_bins)
  per_bin = ceil(sample_size * 2 / num_bins + 1)
  df = pd.concat([df[df['steer_bin'] == i].sample(per_bin, replace=True) for i in range(num_bins)])
  df = df.sample(sample_size)
  # Add a column to represent pixel values
  df['img'] = df['img_center'].apply(lambda file_name: load_image(file_name))
  return df

# output_array = m.sample_to_output_array(sample)

def sample_to_output_array(sample):
  num_rows = len(sample)
  num_categories = len(steering_bins)
  result = np.zeros((num_rows, num_categories))
  for i,steer_bin in enumerate(sample['steer_bin']):
    result[i][steer_bin] = 1
  return result

# generates random subsets of the training data for use in keras's fit_generator
# sample_generator(data_dir=default_data_dir, batch_size=20, sample_filter='training')

class sample_generator(object):
  def __init__(self, data_dir=default_data_dir, batch_size=100, sample_filter='all'):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.sample_filter = sample_filter

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    sample = load_sample(self.data_dir,
                         sample_size=self.batch_size,
                         sample_filter=self.sample_filter)
    input_array = sample_to_input_array(sample)
    output_array = sample_to_output_array(sample)
    return (input_array, output_array)

# input_array = m.sample_to_input_array(sample)

def sample_to_input_array(sample):
  return np.stack(sample['img'].values)

def create_model():
  model = Sequential()
  # Convolution2D(output_depth, convolution height, convolution_width, ...)
  model.add(Convolution2D(6, 5, 5, border_mode='valid', activation='tanh', input_shape=(160,320,3))) # -> (156,316,3)
  model.add(Dropout(0.5))
  model.add(Convolution2D(12, 5, 5, border_mode='valid', activation='tanh', subsample=(2,2))) # -> (76,156,12)
  model.add(Dropout(0.5))
  model.add(Convolution2D(18, 5, 5, border_mode='valid', activation='tanh', subsample=(2,2))) # -> (36,76,18)
  model.add(Dropout(0.5))
  model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='tanh', subsample=(2,2))) # -> (16,36,24)
  model.add(Dropout(0.5))
  model.add(Convolution2D(30, 3, 5, border_mode='valid', activation='tanh', subsample=(2,2))) # -> (7,16,30)
  model.add(Dropout(0.5))
  model.add(Flatten()) # 7x16x30 -> 3360
  model.add(Dense(180, activation='tanh', W_regularizer=l2(0.01)))
  model.add(Dropout(0.4))
  model.add(Dense(90, activation='tanh', W_regularizer=l2(0.01)))
  model.add(Dropout(0.3))
  model.add(Dense(30, activation='tanh', W_regularizer=l2(0.01)))
  model.add(Dropout(0.2))
  model.add(Dense(len(steering_bins), activation='softmax', W_regularizer=l2(0.01)))
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

# import model as m; mod = m.create_model(); hist = m.train_model(mod, m.default_data_dir)

def train_model(model, data_dir):
  return model.fit_generator(sample_generator(data_dir=data_dir,
                                              batch_size=100,
                                              sample_filter='training'),
                             samples_per_epoch=500,
                             nb_epoch=20,
                             validation_data=sample_generator(data_dir=data_dir,
                                                              batch_size=100,
                                                              sample_filter='validation'),
                             nb_val_samples=100)

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
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.load_weights(path + '.h5')
    return model

# python model.py --training_data=/Users/ericlavigne/workspace/CarND-Simulator --save_model=model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimDrive Training')
    parser.add_argument('--load_model', type=str, required=False, default=None,
                        help='Path to model definition for loading (without json/h5 extension)')
    parser.add_argument('--save_model', type=str, required=False, default=None,
                        help='Path to model definition for saving (without json/h5 extension)')
    parser.add_argument('--training_data', type=str, required=True,
                        help='Path to folder with driving_log.csv and IMG subfolder')
    args = parser.parse_args()

    model = None
    if args.load_model:
      print("Loading model from " + args.load_model)
      model = load_model(args.load_model)
    else:
      print("Creating new model")
      model = create_model()

    data_dir = None
    if args.training_data:
      print("Training data in " + args.training_data)
      data_dir = args.training_data
    else:
      print("Need to specify training_data directory")
      exit

    train_model(model, data_dir)

    if args.save_model:
      print("Saving to " + args.save_model)
      save_model(model, args.save_model)
    else:
      print("Not saving because save_model not specified")
    