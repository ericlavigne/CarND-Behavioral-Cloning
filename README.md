# Behavioral Cloning

Explains the structure of your network and training approach,
including complete descriptions of the problems and the strategies.

## Training

Download and run Udacity's car simulator (only available to Udacity students).
Run Udacity's simulator in manual mode to record examples of good driving.

Run the following command to train a model based on that data.

```
python model.py --training_data=/directory/where/you/saved/examples --save_model=model
```

## Running the Simulator

Download and run Udacity's car simulator (only available to Udacity students).
Run Udacity's simulator in autonomous mode.

Run the following command to connect the trained model to that simulator.

```
python drive.py model.json
```

The drive.py file will accept connections on port 4567. Udacity's simulator
will check for a service on port 4567. It doesn't matter which order these
programs are started in because both will retry the connection.

## Installation (using VirtualEnv)

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
