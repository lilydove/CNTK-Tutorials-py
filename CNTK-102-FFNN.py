#coding:utf-8 #
# https://cntk.ai/pythondocs/CNTK_102_FeedForward.html
# Feed Forward Network with Simulated Data
# 0:red bad;  1:blue benign 

from IPython.display import Image

# img2 = plt.open(r"https://upload.wikimedia.org/wikipedia/en/5/54/Feed_forward_neural_net.gif")
#plt.imshow(img2)
#plt.show()
# Figure 2
# Import the relevant components
# from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.pyplot as plt
#%matplotlib inline

import numpy as np
import sys
import os

import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

#------1-Data Generation------
# Define the network
input_dim = 2
num_output_classes = 2

# Ensure that we always get the same results
np.random.seed(0)

# 1------Data------
# Helper function to generate a random data sample
def generate_random_data_sample(sample_size, feature_dim, num_classes):
	# Create synthetic data using Numpy.
	Y = np.random.randint(size=(sample_size,1), low=0, high=num_classes)
	
	# Make sure that the data is separable
	X = (np.random.randn(sample_size, feature_dim)+3) * (Y+1)

	# Specify the data type to match the input variable used later in the tutorial
	# (default type is double)
	X = X.astype(np.float32)

	# convert class 0 into the vector "1 0 0"
	# class 1 into the vector "0 1 0", ...
	class_ind = [Y==class_number for class_number in range(num_classes)]
	Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
	return X,Y

mysamplesize = 64
features, labels = generate_random_data_sample(mysamplesize, input_dim, num_output_classes)
print (features, labels)

# Plot the data
import matplotlib.pyplot as plt
#%matplotlib inline #errors:SyntaxError: invalid syntax

# Let 0 represent malignant/red and 1 represent benign/blue
colors = ['r' if label == 0 else 'b' for label in labels[:,0]] # all lows for 0th comlumn

plt.scatter(features[:,0], features[:,1], c=colors)
plt.xlabel("Age (scaled)")
plt.ylabel("Tumor size (in cm)")
plt.show()

#-------2 Model Creation------ 
num_hidden_layers = 2
hidden_layers_dim = 50

# The input variable (representing 1 obeservation, in out example of age and size) x, which 
# in this case has dimension of 2.
#
# The label variable has a dimensionality equal to the number of output classes in our case 2.
input = C.input_variable(input_dim)
label = C.input_variable(num_output_classes)

def linear_layer(input_var, output_dim):
	input_dim = input_var.shape[0]
	
	weight = C.parameter(shape=(input_dim, output_dim)) # m*n
	bias = C.parameter(shape=(output_dim)) # n

	return bias + C.times(input_var, weight)

def dense_layer(input_var, output_dim, nonlinearity):
	l = linear_layer(input_var, output_dim)

	return nonlinearity(l)

"""
h1 = dense_layer(features, hidden_layers_dim, C.sigmoid)
h2 = dense_layer(h1, hidden_layers_dim, C.sigmoid)
"""
"""2018-02-14
h = dense_layer(input_var, hidden_layer_dim, sigmoid)
for i in range(1, num_hidden_layers):
	h = dense_layer(h, hidden_layer_dim, sigmoid)
"""

# Define a multilayer feedforward classification model
def fully_connected_classifier_net(input_var, num_output_classes, hidden_layer_dim, num_hidden_layers, nonlinearity):
	h = dense_layer(input_var, hidden_layer_dim, nonlinearity)
	for i in range(1, num_hidden_layers):    # 嵌套
		h = dense_layer(h, hidden_layer_dim, nonlinearity)
    
	return linear_layer(h, num_output_classes)

# Create the fully conneted classifier
# 2018-02-16

z = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim, num_hidden_layers, C.sigmoid)
"""
def create_model(features):
	with C.layers.default_options(init=C.layers.glorot_uniform(), activation=C.sigmoid):
		h = features
		for _ in range(num_hidden_layers):
			h = C.layers.Dense(hidden_layers_dim)(h)
		last_layer = C.layers.Dense(num_output_classes, activation = None)

		return last_layer(h)

z1 = create_model(input)
"""
# Combine nets

loss = C.cross_entropy_with_softmax(z, label)
eval_error = C.classification_error(z, label)

learning_rate = 0.5	# if it is 0.1 or less, the error rate is about 0.5.
lr_schedule = C.learning_parameter_schedule(learning_rate)
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, eval_error), [learner])	# input, output, optimizer

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {}, Train Loss: {}, Train Error: {}".format(mb, training_loss, eval_error))

    return mb, training_loss, eval_error

#------Run train------
# Initialize the parameters for the trainer
minibatch_size = 25
num_samples = 10000
num_minibatches_to_train = num_samples / minibatch_size

# Run the trainer and perform model training
training_progress_output_freq = 20

plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, int(num_minibatches_to_train)):
    features, labels = generate_random_data_sample(minibatch_size, input_dim, num_output_classes)

    # Specify the input variables mapping in the model to actual minibatch data for training
    trainer.train_minibatch({input : features, label : labels})
    batchsize, loss, error = print_training_progress(trainer, i,
                                                     training_progress_output_freq, verbose=0)

    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)

# Compute the moving average loss to smooth out the noise in SGD
plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])

# Plot the training loss and the training error
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')

plt.show()

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()

# Generate new data
test_minibatch_size = 25
features, labels = generate_random_data_sample(test_minibatch_size, input_dim, num_output_classes)

trainer.test_minibatch({input : features, label : labels})

#------Prediction------
out = C.softmax(z)
predicted_label_probs = out.eval({input : features})
print("Label    :", [np.argmax(label) for label in labels])
print("Predicted:", [np.argmax(row) for row in predicted_label_probs])
