# https://cntk.ai/pythondocs/CNTK_105_Basic_Autoencoder_for_Dimensionality_Reduction.html
# author:wuyi
# date:2018-03-12--14
# Part A - Time series prediction with LSTM (Basics)
#  
# 
"""
Goal

We use simulated data set of a continuous function (in our case a sine wave).
From N previous values of the y=sin(t)y=sin(t) function 
where yy is the observed amplitude signal at time tt, 
we will predict M values of yy for the corresponding future time points.
"""

# Import the relevant modules
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

import matplotlib.pyplot as plt # plt show images
import matplotlib.image as mpimg # mpimg read images
import numpy as np

from PIL import Image

# Import CNTK
import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

fname = "E:\\ProgramLib\\Python\\CNTK\\testdata\\sinewave.jpg"
img = Image.open(fname)

plt.figure("sinewave") # 
plt.imshow(img)
plt.axis('on') # 
plt.title('sinewave') # 
plt.show()

import math
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import time

import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for out build system)

# %matplotlib inline

# fast or slow mode
isFast = True

#------ 1. Data generation ------
# Split the data into training, validation and test sets.
def split_data(data, val_size=0.1, test_size=0.1):
	"""
	splits np.array into training, validation and test
	"""
	pos_test = int(len(data) * (1 - test_size))    # 1000 in 10000 for testing, so 9000 for training (10000 * 0.9)
	pos_val = int(len(data[:pos_test]) * (1 - val_size))    # 9000 * 0.9 = 8100

	train, val, test = data[:pos_val], data[pos_val:pos_test], data[pos_test:]  # 0-8100, 8101-9000, 9001-10000

	return {"train": train, "val": val, "test": test}

def generate_data(fct, x, time_steps, time_shift):
	"""
	generate sequences to feed to rnn for fct(x)
	"""
	data = fct(x)
	if not isinstance(data, pd.DataFrame):
		data = pd.DataFrame(dict(a = data[0:len(data) - time_shift],
		                         b = data[time_shift:]))  
	rnn_x = []
	for i in range(len(data) - time_steps + 1):
		rnn_x.append(data['a'].iloc[i: i + time_steps].as_matrix())
	rnn_x = np.array(rnn_x)

	# Reshape or rearrange the data from row to columns
	# to be compatible with the input needed by the LSTM model
    # which expects 1 float per time point in a given batch
	rnn_x = rnn_x.reshape(rnn_x.shape + (1,))

	rnn_y = data['b'].values
	rnn_y = rnn_y[time_steps - 1 :]
	rnn_y = rnn_y.reshape(rnn_y.shape + (1,))  # rectify the errors of data

	return split_data(rnn_x), split_data(rnn_y)

N = 5 # input: N subsequent values
M = 5 # output: predict 1 value M steps ahead
X, Y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), N, M)

# f, a = plt.subplots(3, 1, figsize=(12, 8))
# for j, ds in enumerate(["train", "val", "test"]):
#	a[j].plot(Y[ds], label=ds + 'raw');
# [i.legend() for i in a];
# plt.show()

# ------ 2.Create Model ------
# input: N subsequent values
# output: predict 1 value M steps ahead
def create_model(x):
	"""Create the model for time series prediction"""
	with C.layers.default_options(initail_state = 0.1):
		m = C.layers.Recurrence(C.layers.LSTM(N))(x)
		m = C.sequence.last(m)
		m = C.layers.Dropout(0.2, seed=1)(m)
		m = C.layers.Dense(1)(m)
		return m

#------ 3.Train Model ------
def next_batch(x, y, ds):
    """get the next batch to process
	   ds: name of dataset, train val test
	"""

    def as_batch(data, start, count):
        part = []
        for i in range(start, start + count):
            part.append(data[i])
        return np.array(part)

    for i in range(0, len(x[ds])-BATCH_SIZE, BATCH_SIZE):
        yield as_batch(x[ds], i, BATCH_SIZE), as_batch(y[ds], i, BATCH_SIZE)

# Training parameters
TRAINING_STEPS = 10000
BATCH_SIZE = 100
EPOCHS = 10 if isFast else 100

# x_axes = [C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()]
# C.input_variable(1, dynamic_axes=x_axes)

# input sequences
x = C.sequence.input_variable(1, name='x')

# create the model
z = create_model(x)

# expected output (label), also the dynamic axes of the model output
# is specified as the model of the label input
l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name='y')

# the learning rate
learning_rate = 0.02
lr_schedule = C.learning_parameter_schedule(learning_rate)

# loss function
loss = C.squared_error(z, l)

# use squared error to determine error for now
error = C.squared_error(z, l)

# use fsadagrad optimizer
momentum_schedule = C.momentum_schedule(0.9, minibatch_size=BATCH_SIZE)
learner = C.fsadagrad(z.parameters,
                      lr = lr_schedule,
					  momentum = momentum_schedule,
					  unit_gain = True)

trainer = C.Trainer(z, (loss, error), [learner])

# start to train
loss_summary = []
start = time.time()
for epoch in range(0, EPOCHS):
	for x1, y1 in next_batch(X, Y, "train"):
		# print (len(x1),x1[:3], len(y1),y1[:3])
		trainer.train_minibatch({x: x1, l: y1})
	if epoch % (EPOCHS / 10) == 0:
		training_loss = trainer.previous_minibatch_loss_average
		loss_summary.append(training_loss)
		print("epoch: {}, loss: {:.5f}".format(epoch, training_loss))
print("training took {0:.1f} sec".format(time.time() - start))

# A look how the loss function shows how well the model is converging
plt.figure('training loss')
plt.title('CNTK-106-LSTM-PartA-sin')
plt.plot(loss_summary, label='training loss')
plt.show()

#------4.evaluation ------
def get_mse(X,Y,labeltxt):
	result = 0.0
	for x1, y1 in next_batch(X, Y, labeltxt):
		eval_error = trainer.test_minibatch({x : x1, l : y1})
		result += eval_error
	return result/len(X[labeltxt])

# Print the train and validation errors
for labeltxt in ["train", "val"]:
	print("mse for {}: {:.6f}".format(labeltxt, get_mse(X, Y, labeltxt)))

# Print validate and test error
labeltxt = "test"
print("mse for {}: {:.6f}".format(labeltxt, get_mse(X, Y, labeltxt)))


#------5.Predict ------
# predict
f, a = plt.subplots(3, 1, figsize = (12, 8))
for j, ds in enumerate(["train", "val", "test"]):
    results = []
    for x1, y1 in next_batch(X, Y, ds):
        pred = z.eval({x: x1})
        results.extend(pred[:, 0])
    a[j].plot(Y[ds], label = ds + ' raw')
    a[j].plot(results, label = ds + ' predicted')
[i.legend() for i in a]
plt.title('Predict') # 
plt.show()