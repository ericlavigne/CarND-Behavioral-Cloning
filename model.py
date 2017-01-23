# create and train the model

import argparse
import base64
import json

import numpy as np

import time
from PIL import Image
from PIL import ImageOps

from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import tensorflow as tf
tf.python.control_flow_ops = tf # mysterious fix to keras/tensorflow issue

image = Image.open(BytesIO(base64.b64decode(imgString)))
image_array = np.asarray(image)
transformed_image_array = image_array[None, :, :, :]

steering_angle = float(model.predict(transformed_image_array, batch_size=1))

# Later I'll probably add speed as an input variable and
# acceleration as an output variable.  This will involve
# a small change to drive.py around line 44.
# steering_angle = ...

# Does the simulator support negative throttle? This is needed both
# for braking at barricades in map 2, as well as for 3-point turns
# that I'd like to try later. Will start by changing throttle = 0.2
# to throttle = -0.2 to check if that works. If not, will need to
# ask Udacity staff if simulator can be upgraded.



# Saving and loading keras models
# https://keras.io/models/about-keras-models/

model.summary()
model.get_config()
model = Model.from_config(config)

from models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)

with open(args.model, 'r') as jfile:
  model = model_from_json(jfile.read())

weights_file = args.model.replace('json', 'h5')
model.load_weights(weights_file)

# Main

# Args could include training data folder and output model file.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()

    
