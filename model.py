"""
   model.py: create, train, and validate the model

   Usage: python model.py --training_data=/Users/ericlavigne/workspace/CarND-Simulator --save_model=model
"""

import argparse
import cv2
import gc
from io import BytesIO
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import re

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2

import tensorflow as tf
tf.python.control_flow_ops = tf # mysterious fix to keras/tensorflow issue

default_data_dir = './data'

def convert_image_to_input_format(original):
  """Image preprocessing that must be applied before feeding as input to the model.
     Called from both model.py and drive.py."""
  img = original
  # Crop bottom to hide car (y=130, hint of left/right/center camera)
  # Crop top to hide non-road scenery (y=60, trees/skies/mountains not relevant)
  img = img[60:130,0:320] # crop 320x160 -> 320x70, removing bottom (car hood) and top (scenery)
  # Convert color representation from blue/green/red to hue/saturation/value
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # Normalize all channels to [-0.5,0.5] range
  img = (img / 255) - 0.5
  return img

# import model as m; m.load_image('/Users/ericlavigne/workspace/CarND-Simulator/IMG/center_2017_01_21_19_10_57_316.jpg')

def load_image(file_name):
  """Convert file name to image, including preprocessing for use as model input."""
  img = mpimg.imread(file_name)
  return convert_image_to_input_format(img)

# import model as m; m.load_summary_data('../CarND-Simulator').head(5)

def load_summary_data(data_dir):
  """Result is Pandas DataFrame"""
  file_name = data_dir + '/driving_log.csv'
  col_names = ['img_center','img_left','img_right','steer','throttle','brake','speed','notes']
  df = pd.read_csv(file_name, names=col_names)
  return df

# import model as m; sample = m.load_sample(m.default_data_dir)

def load_sample(data_dir):
  """Similar to load_summary_data except image paths replaced with actual images"""
  df = load_summary_data(data_dir)
  for camera in ['left','right','center']:
    column = 'img_' + camera 
    df[column] = df[column].apply(lambda file_name: load_image(re.sub(r".*/IMG/", data_dir + "/IMG/", file_name)))
  return df

# input_array = m.sample_to_input_array(sample)

def sample_to_input_array(sample):
  """Convert Pandas DataFrame to model input array.
     Result is three times longer than input due to left/right/center
     becoming separate examples in model input."""
  return np.concatenate((np.stack(sample['img_left'].values),
                         np.stack(sample['img_center'].values),
                         np.stack(sample['img_right'].values)))

# output_array = m.sample_to_output_array(sample)

def sample_to_output_array(sample):
  """Convert Pandas DataFrame to model output array.
     Result is three times longer than original DataFrame due to
     left/right/center becoming separate rows in model output."""
  num_rows = len(sample)
  result = np.zeros((num_rows * 3, 1))
  for camera_index, camera in enumerate(['left','center','right']):
    angle_offset = [0.05, 0.00, -0.05][camera_index]
    for sample_index,steer_angle in enumerate(sample['steer']):
      result[(camera_index * num_rows) + sample_index][0] = steer_angle + angle_offset
  return result

def compile_model(model):
  """Would be part of create_model, except that same settings
     also need to be applied when loading model from file."""
  model.compile(optimizer='adam',
                loss='mean_absolute_error',
                metrics=['mean_absolute_error','mean_squared_error'])

def create_model():
  """Create neural network model, defining layer architecture."""
  model = Sequential()
  # Convolution2D(output_depth, convolution height, convolution_width, ...)
  model.add(Convolution2D(6, 5, 5, border_mode='valid', activation='tanh', input_shape=(70,320,3))) # -> (66,316,6)
  model.add(Dropout(0.5))
  model.add(Convolution2D(12, 5, 5, border_mode='valid', activation='tanh', subsample=(2,2))) # -> (31,156,12)
  model.add(Dropout(0.5))
  model.add(Convolution2D(18, 5, 5, border_mode='valid', activation='tanh', subsample=(2,2))) # -> (14,76,18)
  model.add(Dropout(0.5))
  model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='tanh', subsample=(1,2))) # -> (10,36,24)
  model.add(Dropout(0.5))
  model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='tanh', subsample=(1,2))) # -> (6,16,24)
  model.add(Dropout(0.5))
  model.add(Flatten()) # 6x16x24 -> 2304
  model.add(Dense(30, activation='tanh', W_regularizer=l2(0.01)))
  model.add(Dropout(0.4))
  model.add(Dense(25, activation='tanh', W_regularizer=l2(0.01)))
  model.add(Dropout(0.3))
  model.add(Dense(20, activation='tanh', W_regularizer=l2(0.01)))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='tanh', W_regularizer=l2(0.01)))
  compile_model(model)
  return model

# import model as m; mod = m.create_model(); hist = m.train_model(mod, m.default_data_dir)

def train_model(model, data_dir, validation_percentage=None, epochs=100):
  """Train the model. With so few examples, I usually prefer
     to use all examples for training. Setting aside some
     examples for validation is supported but not recommended."""
  sample = load_sample(data_dir)
  input_array = sample_to_input_array(sample)
  output_array = sample_to_output_array(sample)
  if validation_percentage:
    return model.fit(input_array, output_array, nb_epoch=epochs, validation_split = validation_percentage / 100.0)
  else:
    return model.fit(input_array, output_array, nb_epoch=epochs)

def save_model(model,path='model'):
  """Save model as .h5 and .json files. Specify path without these extensions."""
  with open(path + '.json', 'w') as arch_file:
    arch_file.write(model.to_json())
  model.save_weights(path + '.h5')

def load_model(path):
  """Load model from .h5 and .json files. Specify path without these extensions."""
  with open(path + '.json', 'r') as arch_file:
    model = model_from_json(arch_file.read())
    compile_model(model)
    model.load_weights(path + '.h5')
    return model

# python model.py --training_data=data --save_model=model

# python model.py --training_data=data --validation_percentage=10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimDrive Training')
    parser.add_argument('--load_model', type=str, required=False, default=None,
                        help='Path to model definition for loading (without json/h5 extension)')
    parser.add_argument('--save_model', type=str, required=False, default=None,
                        help='Path to model definition for saving (without json/h5 extension)')
    parser.add_argument('--validation_percentage', type=float, default=None,
                        help='Percentage of training data to set aside for validation')
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

    validation_percentage = None
    if args.validation_percentage:
      print("Validation percentage set to " + str(args.validation_percentage))
      validation_percentage = args.validation_percentage
    else:
      print("No validation - using all examples for training")

    train_model(model, data_dir, validation_percentage=validation_percentage)

    if args.save_model:
      print("Saving to " + args.save_model)
      save_model(model, args.save_model)
    else:
      print("Not saving because save_model not specified")

    # Workaround for TensorFlow bug. Force early garbage collection of model.
    model = None
    gc.collect()
    