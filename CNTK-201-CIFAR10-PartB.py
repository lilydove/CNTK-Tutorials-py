# https://cntk.ai/pythondocs/CNTK_201B_CIFAR-10_ImageHandsOn.html
# CNTK 201: Part B - Image Understanding
# Author: wuyi
# Date: 2018-03-24--26
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
print("Read data successfully.")

