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

def convert_image_and_speed_to_input_format(original,speed):
  """Image preprocessing that must be applied before feeding as input to the model.
  
     This includes adding speed (normalized [-0.5,0.5]) as a fourth channel as
     workaround for not knowing how to introduce a non-image input after the
     convolutional layers in keras.
     
     Called from both model.py and drive.py."""
     
  img = original
  # Crop bottom to hide car (y=130, hint of left/right/center camera)
  # Crop top to hide non-road scenery (y=60, trees/skies/mountains not relevant)
  img = img[60:130,0:320] # crop 320x160 -> 320x70, removing bottom (car hood) and top (scenery)
  # Convert color representation from blue/green/red to hue/saturation/value
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # Normalize all channels to [-0.5,0.5] range
  img = (img / 255) - 0.5
  # Add normalized speed 
  hue_channel, saturation_channel, value_channel = cv2.split(img)
  speed_channel = np.ones(hue_channel.shape) * (speed / 60.0)
  img = cv2.merge((hue_channel, saturation_channel, value_channel, speed_channel))
  
  return img

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
  convert_image_path = lambda file_name: re.sub(r".*/IMG/", data_dir + "/IMG/", file_name)
  read_image = lambda file_name: mpimg.imread(convert_image_path(file_name))
  for camera in ['left','right','center']:
    column = 'img_' + camera 
    df_img_and_speed = df[[column,'speed']].apply(tuple, axis=1)
    df[column] = df_img_and_speed.apply(lambda img_and_speed: convert_image_and_speed_to_input_format(read_image(img_and_speed[0]),
                                                                                                      img_and_speed[1]))
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
     left/right/center becoming separate rows in model output.
     Output includes both steering angle and throttle."""
  num_rows = len(sample)
  result = np.zeros((num_rows * 3, 2))
  for camera_index, camera in enumerate(['left','center','right']):
    angle_offset = [0.05, 0.00, -0.05][camera_index]
    for sample_index,steer_and_throttle in enumerate(sample[['steer','throttle']].apply(tuple, axis=1)):
      steer_angle = steer_and_throttle[0]
      throttle = steer_and_throttle[1]
      result_index = (camera_index * num_rows) + sample_index
      result[result_index][0] = steer_angle + angle_offset
      result[result_index][1] = throttle * 0.2
  return result

driving_loss_weights = tf.constant([0.8, 0.2])
  
def driving_loss(y_true, y_pred):
  """Custom loss function for driving agent. Steering angle is more difficult
     and more important than throttle, so loss should give steering angle more
     weight."""
  individual_losses = tf.abs(y_pred - y_true)
  weighted_individual_losses = tf.multiply(individual_losses, driving_loss_weights)
  return tf.reduce_mean(weighted_individual_losses, axis=-1)

def compile_model(model):
  """Would be part of create_model, except that same settings
     also need to be applied when loading model from file."""
  model.compile(optimizer='adam',
                loss=driving_loss,
                metrics=['mean_absolute_error','mean_squared_error'])

def create_model():
  """Create neural network model, defining layer architecture."""
  model = Sequential()
  # Convolution2D(output_depth, convolution height, convolution_width, ...)
  model.add(Convolution2D(8, 5, 5, border_mode='valid', activation='tanh', input_shape=(70,320,4))) # -> (66,316,8)
  model.add(Dropout(0.5))
  model.add(Convolution2D(16, 5, 5, border_mode='valid', activation='tanh', subsample=(2,2))) # -> (31,156,16)
  model.add(Dropout(0.5))
  model.add(Convolution2D(20, 5, 5, border_mode='valid', activation='tanh', subsample=(2,2))) # -> (14,76,20)
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
  model.add(Dense(2, activation='tanh', W_regularizer=l2(0.01)))
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
    