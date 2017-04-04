## Behavioral Cloning - Training a Self-Driving Car by Example
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project satisfies the requirements for Udacity's Behavioral Cloning project.

The goal of this project was to train a neural network to mimic human driving
behavior. Students typically collect thousands of frames of examples, combined
with corresponding driving angles, and train a convolutional neural network to
predict those angles.

I took a different approach. My neural network used just 25 frames of training
data for the Lake track. My neural network used 32 additional frames (57 total)
for the Mountain track. I carefully selected frames, and associated steering
angles, to explain to the neural network how it should drive around the track.
Quality beats quantity.

[Lake Track Video](https://youtu.be/-UHQ7DkMcRA)

[Mountain Track Video](https://youtu.be/o4A9901Yoa4)

Thank you to Feng Gao for inspiring me to attempt training on a small, carefully
curated selection of frames.

*Note: Find the latest version of this project on
[Github](https://github.com/ericlavigne/CarND-Behavioral-Cloning).*

Selecting Frames for Training
---

Each frame was selected for a specific purpose. I started by selecting just a
few frames from key points in the video, such as the starting point and the
first turn. After adding a new frame I would retrain, run the simulator, and
watch for the car's first time running off the road. I would add one more frame
to show the car what it should have done at that point, retrain, and repeat.

See the full list of training data in
[data/driving_log.csv](https://github.com/ericlavigne/CarND-Behavioral-Cloning/blob/master/data/driving_log.csv).
In particular, note the last column where I describe the purpose of each frame.

You can also view the images for each frame in
[data/IMG](https://github.com/ericlavigne/CarND-Behavioral-Cloning/tree/master/data/IMG).

Pre-Processing
---

I cropped all input images to remove the sky and the hood of the car, both
of which are irrelevant to driving and may have distracted the model.

I converted the color representations from RGB to HSV, after hearing that this
worked well for another student. I have not experimented to determine whether
this conversion was beneficial.

I transformed all image channels from (0,255) range to (-0.5,0.5) because
small ranges centered around 0 are more suitable for neural network training.

Pre-processing code is in the convert_image_and_speed_to_input_format function.

Data Augmentation
---

Each frame includes three images: center, left, and right. In an earlier version
of this project, I used the left and right images as a form of augmentation,
applying an inward correction of 0.1 to the steering angle of the left and right
cameras. This augmentation made the car's movements much smoother, especially
in straight parts of the track, but reduced the car's ability to handle sharp
turns.

I later removed the augmentation to improve the car's maneuverability, but 
you can find commented-out augmentation code in the sample_to_input_array and
sample_to_output_array functions.

Model Architecture
---

The neural network model includes five 5x5 convolutional layers with depths of
8, 16, 20, 24, and 24. Each of these layers has 50% dropout to prevent
overfitting. Due to sub-sampling, these layers condense the original 70x320 input
into 6x16. These convolutional layers are followed by three fully-connected
hidden layers with 30, 25, and 20 neurons. These fully connected hidden layers
have dropouts of 40%, 30%, and 20%. A single output neuron predicts the steering
angle as a range from -1 to 1 (which translates to -25 degrees to 25 degrees).

All neural network layers use tanh activation, which has worked much better in
my experience than the relu activations that are commonly recommended. I use the
Adam optimizer to avoid parameter tuning for the learning rate. A generator is
not needed because the small training set easily fits in memory.

See details about the model architecture in create_model, compile_model, and
train_model.

Validation
---

The usual training/validation split is not really appropriate for a small
training set like this. Every image is important for training.

In order to meet the official requirement for validation, I added this feature
as an optional commandline parameter. I found that validation loss was consistently
lower than training loss, due to heavy use of dropouts for regularization.

More importantly, the car drives around the track without touching the edges.
The car's behavior is far more important than validation scores.

Discussion
---

While the car performs well on sharp turns in the Lake and Mountain tracks,
its movements could be a lot smoother on straight sections. I've seen that
augmentation with the left and right cameras makes the car drive more smoothly,
but at the expense of maneuverability. The solution may be more sophisticated
calculation of the offsets, so that for sharp turns the offset is smaller on the
inner side and larger on the outer side.

The car also makes it through only the first few turns on the more advanced
Jungle track. It is not yet clear whether adding more training frames will be
sufficient to complete the jungle track or whether a deeper neural network
will also be needed.

## Instructions

Training
---

Download and run Udacity's car simulator (only available to Udacity students).
Run Udacity's simulator in manual mode to record examples of good driving.

Run the following command to train a model based on that data.

```
python model.py --training_data=/directory/where/you/saved/examples --save_model=model
```

Running the Simulator
---

Download and run Udacity's car simulator (only available to Udacity students).
Run Udacity's simulator in autonomous mode.

Run the following command to connect the trained model to that simulator.

```
python drive.py model.json
```

The drive.py file will accept connections on port 4567. Udacity's simulator
will check for a service on port 4567. It doesn't matter which order these
programs are started in because both will retry the connection.

Installation (using VirtualEnv)
---

First time:

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
deactivate
```

When starting to work:

```
source env/bin/activate
```

After installing new libraries:

```
pip freeze > requirements.txt
```

When finished:

```
deactivate
```
