# https://cntk.ai/pythondocs/CNTK_105_Basic_Autoencoder_for_Dimensionality_Reduction.html
# author:wuyi
# date:2018-03-10
# CNTK 105: Basic autoencoder (AE) with MNIST data
# mnist_autoendcoder_simple
# 
"""
Goal:
Our goal is to train an autoencoder that 
compresses MNIST digits image to a vector of smaller dimension 
and then restores the image.
"""
# Import the relevant modules
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

from IPython.display import Image
from PIL import Image

import matplotlib.pyplot as plt # plt show images
import matplotlib.image as mpimg # mpimg read images
import numpy as np

import os
import sys

# Import CNTK
import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

#------Download images ------
#try:
#	from urllib.request import urlretrieve
#except ImportError:
#	from urllib import urlretrieve

# Figure 1
# src = r'http://cntk.ai/jup/MNIST-image.jpg'
# print ('Downloading ' + src)
# fname, h = urlretrieve(src, 'E:\\ProgramLib\\Python\\CNTK\\testdata\\testdata20180310.jpg')
# print ("File name : " , fname)
# print ("File head info :", h)

# Show images
# F 1
img = Image.open("E:\\ProgramLib\\Python\\CNTK\\testdata\\testdata20180310.jpg")
plt.figure("MNIST-image") # 
plt.imshow(img)
plt.axis('on') # 
plt.title('MNIST-image') # 
plt.show()
#F 2 # Figure 2
# Image(url="http://cntk.ai/jup/SimpleAEfig.jpg", width=200, height=200)
img = Image.open("E:\\ProgramLib\\Python\\CNTK\\testdata\\SimpleAEfig.jpg")
plt.figure("SimpleAEfig") # 
plt.imshow(img)
plt.axis('on') # 
plt.title('SimpleAEfig') # 
plt.show()


# fast or slow mode
isFast = True

# ------ 1.Data reading ------
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

# ------ 2. Create Model ------
input_dim = 784
encoding_dim = 32
output_dim = input_dim

def create_model(features):
    with C.layers.default_options(init = C.glorot_uniform()):
        # We scale the input pixels to 0-1 range
        encode = C.layers.Dense(encoding_dim, activation = C.relu)(features/255.0)
        decode = C.layers.Dense(output_dim, activation = C.sigmoid)(encode)

    return decode


# ------ 3.Train and Test ------
def train_and_test(reader_train, reader_test, model_func):

    ###############################################
    # Training the model
    ###############################################

    # Instantiate the input and the label variables
    input = C.input_variable(input_dim)
    label = C.input_variable(input_dim)

    # Create the model function
    model = model_func(input)

    # The labels for this network is same as the input MNIST image.
    # Note: Inside the model we are scaling the input to 0-1 range
    # Hence we rescale the label to the same range
    # We show how one can use their custom loss function
    # loss = -(y* log(p)+ (1-y) * log(1-p)) where p = model output and y = target
    # We have normalized the input between 0-1. Hence we scale the target to same range

    target = label/255.0
    loss = -(target * C.log(model) + (1 - target) * C.log(1 - model))
    label_error  = C.classification_error(model, target)

    # training config
    epoch_size = 30000        # 30000 samples is half the dataset size
    minibatch_size = 64
    num_sweeps_to_train_with = 5 if isFast else 100
    num_samples_per_sweep = 60000
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) // minibatch_size


    # Instantiate the trainer object to drive the model training
    lr_per_sample = [0.00003]
    lr_schedule = C.learning_parameter_schedule_per_sample(lr_per_sample, epoch_size)

    # Momentum which is applied on every minibatch_size = 64 samples, relations
    momentum_schedule = C.momentum_schedule(0.9126265014311797, minibatch_size)

    # We use a variant of the Adam optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = C.fsadagrad(model.parameters,
                         lr=lr_schedule, momentum=momentum_schedule)

    # Instantiate the trainer
    progress_printer = C.logging.ProgressPrinter(0)
	# Inside, Outside, Force, Light
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # Map the data streams to the input and labels.
    # Note: for autoencoders input == label
    input_map = {
        input  : reader_train.streams.features,
        label  : reader_train.streams.features
    }

    aggregate_metric = 0
    for i in range(num_minibatches_to_train):
        # Read a mini batch from the training data file
        data = reader_train.next_minibatch(minibatch_size, input_map = input_map)

        # Run the trainer on and perform model training
        trainer.train_minibatch(data)
        samples = trainer.previous_minibatch_sample_count
        aggregate_metric += trainer.previous_minibatch_evaluation_average * samples

    train_error = (aggregate_metric*100.0) / (trainer.total_number_of_samples_seen)
    print("Average training error: {0:0.2f}%".format(train_error))

    #############################################################################
    # Testing the model
    # Note: we use a test file reader to read data different from a training data
    #############################################################################

    # Test data for trained model
    test_minibatch_size = 32
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    test_result = 0.0

    # Test error metric calculation
    metric_numer    = 0
    metric_denom    = 0

    test_input_map = {
        input  : reader_test.streams.features,
        label  : reader_test.streams.features
    }

    for i in range(0, int(num_minibatches_to_test)):

        # We are loading test data in batches specified by test_minibatch_size
        # Each data point in the minibatch is a MNIST digit image of 784 dimensions
        # with one pixel per dimension that we will encode / decode with the
        # trained model.
        data = reader_test.next_minibatch(test_minibatch_size,
                                       input_map = test_input_map)

        # Specify the mapping of input variables in the model to actual
        # minibatch data to be tested with
        eval_error = trainer.test_minibatch(data)

        # minibatch data to be trained with
        metric_numer += np.abs(eval_error * test_minibatch_size)
        metric_denom += test_minibatch_size

    # Average of evaluation errors of all test minibatches
    test_error = (metric_numer*100.0) / (metric_denom)
    print("Average test error: {0:0.2f}%".format(test_error))

    return model, train_error, test_error

isTrain = False  # not to train and evaluate directly with model trained
modelFile = r'E:\ProgramLib\Python\CNTK\model\mnist_autoendcoder_simple.cmf'
if not isTrain :
    # Load model
    model = C.Function.load(modelFile)
    print ("Loading model")
else:
    # Feed data to train
    num_label_classes = 10
    reader_train = create_reader(train_file, True, input_dim, num_label_classes)
    reader_test = create_reader(test_file, False, input_dim, num_label_classes)
    model, simple_ae_train_error, simple_ae_test_error = train_and_test(reader_train,
                                                                    reader_test,
                                                                    model_func = create_model )
    # Save model
    model.save(modelFile)

#------ 4. Evaluation ------
# Read some data to run the eval
num_label_classes = 10
reader_eval = create_reader(test_file, False, input_dim, num_label_classes)

eval_minibatch_size = 50
eval_input_map = {input : reader_eval.streams.features}

eval_data = reader_eval.next_minibatch(eval_minibatch_size,
                                       input_map = eval_input_map)
img_data = eval_data[input].asarray()

# Select a random image
# np.random.seed(0) # be sure to get the same result
idx = np.random.choice(eval_minibatch_size)

orig_image = img_data[idx,:,:]
decoded_image = model.eval(orig_image)[0]*255

# Print image statistics
def print_image_stats(img, text):
	print(text)
	print("Max: {0:.2f}, Median: {1:.2f}, Mean: {2:.2f}, Min: {3:.2f}".format(np.max(img),
	                                                                          np.median(img),
																			  np.mean(img),
																			  np.min(img)))
# Print original image
print_image_stats(orig_image, "Original image statistics:")

# Print decoded image
print_image_stats(decoded_image, "Decoded image statistics:")

# Define a helper function to plot a pair of images
def plot_image_pair(img1, text1, img2, text2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))

    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title(text1)
    axes[0].axis("off")

    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title(text2)
    axes[1].axis("off")
    plt.show() # CNTK's tutorial has not the row

# Plot the original and the decoded image
img1 = orig_image.reshape(28,28)
text1 = 'Original image'

img2 = decoded_image.reshape(28,28)
text2 = 'Decoded image'

plot_image_pair(img1, text1, img2, text2)