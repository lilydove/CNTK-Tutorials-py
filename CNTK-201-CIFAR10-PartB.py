# https://cntk.ai/pythondocs/CNTK_201B_CIFAR-10_ImageHandsOn.html
# CNTK 201: Part B - Image Understanding
# Author: wuyi
# Date: 2018-03-24--04.01
"""
This tutorial shows how to implement image recognition task using convolution network with CNTK v2 Python API.
You will start with a basic feedforward CNN architecture to classify CIFAR dataset,
then you will keep adding advanced features to your network. Finally,
you will implement a VGG net and residual net like the one that won ImageNet competition but smaller in size.
"""

from IPython.display import Image
from PIL import Image

import matplotlib.pyplot as plt # plt show images
import matplotlib.image as mpimg # mpimg read images
import numpy as np

import os
import PIL
import sys

import cntk as C

if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
        print ('CPU')
    else:
        C.device.try_set_default_device(C.device.gpu(0))
        print ('GPU')

# The parameters to control
isTrain = True

modelFile = r'E:\ProgramLib\Python\CNTK\model\cifar10-cnn.cmf'

def show_pics():
    # Figure 1
    fname = 'E:\\ProgramLib\\Python\\CNTK\\testdata\\cifar-10.png'
    img = Image.open(fname)

    plt.figure("cifar-10") #
    plt.imshow(img)
    plt.axis('on') #
    plt.title('cifar-10') #
    plt.show()

    # Figure 2
    fname = 'E:\\ProgramLib\\Python\\CNTK\\testdata\\Conv2D.png'
    img = Image.open(fname)

    plt.figure("Conv2D") #
    plt.imshow(img)
    plt.axis('on') #
    plt.title('Conv2D') #
    plt.show()

    # Figure 3
    fname = 'E:\\ProgramLib\\Python\\CNTK\\testdata\\Conv2DFeatures.png'
    img = Image.open(fname)

    plt.figure("Conv2DFeatures") #
    plt.imshow(img)
    plt.axis('on') #
    plt.title('Conv2DFeatures') #
    plt.show()

    # Figure 4
    fname = 'E:\\ProgramLib\\Python\\CNTK\\testdata\\MaxPooling.png'
    img = Image.open(fname)

    plt.figure("MaxPooling") #
    plt.imshow(img)
    plt.axis('on') #
    plt.title('MaxPooling') #
    plt.show()

	# Figure 5
    fname = 'E:\\ProgramLib\\Python\\CNTK\\testdata\\CNN.png'
    img = Image.open(fname)

    plt.figure("CNN") #
    plt.imshow(img)
    plt.axis('on') #
    plt.title('CNN') #
    plt.show()

#show_pics()
#------1. Data ------
# Determine the data path for testing
# Check for an environment variable defined in CNTK's test infrastructure
envvar = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'
def is_test(): return envvar in os.environ
isTest = is_test()
print (isTest)
filepath = 'E:/ProgramLib/Python/CNTK/testdata/'
if isTest:
    data_path = os.path.join(os.environ[envvar],'Image','CIFAR','v0','tutorial201')
    data_path = os.path.normpath(data_path)
else:
    data_path = filepath + 'CIFAR-10/'
print (data_path)

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3
num_classes  = 10

import cntk.io.transforms as xforms

#
# Define the reader for both training and evaluation action.
#
def create_reader(map_file, mean_file, train):
    print("Reading map file:", map_file)
    print("Reading mean file:", mean_file)

    if not os.path.exists(map_file) or not os.path.exists(mean_file):
       raise RuntimeError("This tutorials depends 201A tutorials, please run 201A first.")

    # transformation pipline for the features has jitter/crop only when training
    transforms = []
	# train uses data augmentation (translation only)
    if train:
	    transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8)
        ]

    transforms += [
        xforms.scale(width=image_width, height=image_height,channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]
    # deserializer
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file,C.io.StreamDefs(
         features = C.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
         labels   = C.io.StreamDef(field='label', shape=num_classes)
    )))

# Create the train and test readers
reader_train = create_reader(data_path  + 'train_map.txt',
                             data_path + 'CIFAR-10_mean.xml', True)
reader_test  = create_reader(data_path + 'test_map.txt',
                             data_path +'CIFAR-10_mean.xml', False)
print (reader_train)
print("Read data successfully.")

#------2. Model------
# Basic CNN
def create_basic_model(input, out_dims):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        net = C.layers.Convolution((5,5), 32, pad=True)(input)
        net = C.layers.MaxPooling((3,3), strides=(2,2))(net)

        net = C.layers.Convolution((5,5), 32, pad=True)(net)
        net = C.layers.MaxPooling((3,3), strides=(2,2))(net)

        net = C.layers.Convolution((5,5), 32,pad=True)(net)
        net = C.layers.MaxPooling((3,3), strides=(2,2))(net)

        net = C.layers.Dense(64)(net)
        net = C.layers.Dense(out_dims, activation=None)(net)

    return net


#------3. Training and Evaluation------
#
# Train and evaluate the net work
#
def train_and_evaluate(reader_train, reader_test, max_epochs, model_func):
    # Input variables denoting the features and label data
    input_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))

    # Normalize the input
    feature_scale = 1.0 / 256.0
    input_var_norm = C.element_times(feature_scale, input_var)

	# apply model to input
    z = model_func(input_var_norm, out_dims = 10)  # input, output

	#
	# Training action
    #

    # loss and metric
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)   # inner and outer

    # training config
    epoch_size     = 10
    minibatch_size = 64

    # set training parameters
    lr_per_minibatch     = C.learning_parameter_schedule([0.01]*10 + [0.003]*10 + [0.001],epoch_size = epoch_size)
    momentums            = C.momentum_schedule(0.9, minibatch_size = minibatch_size)
    l2_reg_weight        = 0.001

    # trainer object
    learner = C.momentum_sgd(z.parameters,
	                         lr = lr_per_minibatch,
							 momentum = momentums,
							 l2_regularization_weight=l2_reg_weight)
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    trainer = C.Trainer(z, (ce, pe), [learner], [progress_printer]) # in,out,force,running

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    C.logging.log_number_of_parameters(z)
    print ()

    # perform model training
    batch_index = 0
    plot_data = {'batchindex': [], 'loss':[], 'error':[]}
    for epoch in range(max_epochs):    # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:   #loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count),
                                               input_map=input_map)  # fetch minibatch
            trainer.train_minibatch(data)

            sample_count += data[label_var].num_samples

            # for visualization...
            plot_data['batchindex'].append(batch_index)
            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
            print(plot_data['loss'])
            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)

            batch_index += 1
        trainer.summarize_training_progress()

    #
    # Evaluation action
    #
    epoch_size = 10
    minibatch_size = 16

    # process minibatches and evaluate the model
    metric_number    = 0
    metric_denom    = 0
    sample_count     = 0
    minibatch_index  = 0

    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)
        
        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)

        # minibatch data to be trained with
        metric_number += trainer.test_minibatch(data) * current_minibatch
        metric_denom  += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print('')
    print('Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}'.format(minibatch_index+1, (metric_denom*100.0)/metric_denom, metric_denom))
    print('')

    # Visualize training result:
    window_width    = 1
    loss_cumsum     = np.cumsum(np.insert(plot_data['loss'], 0, 0))
    error_cumsum    = np.cumsum(np.insert(plot_data['error'], 0, 0))

    print('loss_cumsum = ', loss_cumsum, len(loss_cumsum))
    
    # Moving average.
    plot_data['bacthindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
    plot_data['avg_loss']   = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
    plot_data['avg_error']  = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width

    print ('batchindex = ', plot_data['batchindex'], loss_cumsum[window_width:],loss_cumsum[:-window_width])
    print ('avg_loss = ', plot_data['avg_loss'])
    print ('avg_error = ', plot_data['avg_error'])

    plt.figure(1)
    plt.subplot(211)
    plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss ')

    plt.show()

    plt.subplot(212)
    plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Minibatch run vs. Label Prediction Error ')
    plt.show()

    return C.softmax(z)
print (reader_test.stream_infos())
pred = train_and_evaluate(reader_train,
                           reader_test,
                           max_epochs=5,
                           model_func=create_basic_model)



