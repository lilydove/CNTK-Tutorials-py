# https://cntk.ai/pythondocs/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.html

'''
Instruction:
1.Data
    C.io.StreamDef
	C.io.CTFDeserializer
	C.io.MinibatchSource
2.Model
    C.input_variable(input_dim)
	C.layers.Convolution2D.(filter_shape=(5,5),
		                           num_filters=8,
								   strides=(2,2),
								   pad=True, name='first_conv')(h)
	C.layers.Dense(num_output_classes, activation = None)(features)
	C.cross_entropy_with_softmax(z, label)
	C.classification_error(z, label)
3.Train
    C.learning_parameter_schedule(learning_rate)
	C.sgd(z.parameters, lr_schedule)
	C.Trainer(z, (loss, label_error), [learner])
'''
# Import the relevant components
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix the random seed so that LR examples are repeatable

#%matplotlib inline

# Initialization
# Define the data dimensions
input_dim_model = (1, 28, 28)    # images are 28 x 28 with 1 channel of color (gray)
input_dim = 28*28                # used by readers to treat input data as a vector
num_output_classes = 10

#------1. Data reading-------
# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):

	labelStream = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False)
	featureStream = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)

	deserailizer = C.io.CTFDeserializer(path, C.io.StreamDefs(labels = labelStream, features = featureStream))

	return C.io.MinibatchSource(deserailizer, randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

# Ensure the training and test data is generated and available for this tutorial.
# We search in two locations in the toolkit for the cached MNIST data set.
data_found = False
filepath = 'E:\\ProgramLib\\Python\\CNTK\\testdata'

for data_dir in [os.path.join("..", "Examples", "Image", "DataSets", "MNIST"), os.path.join("data", "MNIST"),filepath]:
    train_file = os.path.join(data_dir, "Train-2828_cntk_text.txt")
    print (train_file)
    test_file = os.path.join(data_dir, "Test-2828_cntk_text.txt")
    print (test_file)
    if os.path.isfile(train_file) and os.path.isfile(test_file):
		data_found = True
		break

if not data_found:
	raise ValueError("Please generate the data by completing CNTK 103 Part A")

print ("Data directory is {0}".format(data_dir))

'''
from IPython.display import Image
# Plot images with strides of 2 and 1 with padding turned on
images = [("https://www.cntk.ai/jup/cntk103d_padding_strides.gif" , 'With stride = 2'),
          ("https://www.cntk.ai/jup/cntk103d_same_padding_no_strides.gif", 'With stride = 1')]


Image(url="https://www.cntk.ai/jup/cntk103d_conv2d_final.gif", width= 300)
for im in images:
    print(im[1])
    display(Image(url=im[0], width=200, height=200))
'''

#------2 Create Model------
input = C.input_variable(input_dim)
label = C.input_variable(num_output_classes)

# function to build model
def create_model(features):
	with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
		h = features
		h = C.layers.Convolution2D(filter_shape=(5,5),
		                           num_filters=8,
								   strides=(2,2),
								   pad=True, name='first_conv')(h)
		h = C.layers.Convolution2D(filter_shape=(5,5),
		                           num_filters=16,
								   strides=(2,2),
								   pad=True, name='second_conv')(h)
		r = C.layers.Dense(num_output_classes, activation=None, name='classify')(h)
		return r

# Create the model
z = create_model(x)


# Print the output shapes / parameters of different components
print("Output Shape of the first convolution layer:", z.first_conv.shape)
print("Bias value of the last dense layer:", z.classify.b.value)

C.logging.log_number_of_parameters(z)

def create_criterion_function(model, labels):
    loss = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error(model, labels)
    return loss, errs # (model, labels) -> (loss, error metric)

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
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
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))

    return mb, training_loss, eval_error