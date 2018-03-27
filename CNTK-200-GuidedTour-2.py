# https://cntk.ai/pythondocs/CNTK_200_GuidedTour.html
# Putting it all Together: Advanced Training Example

from __future__ import print_function
import cntk
import numpy as np
import scipy.sparse
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
cntk.cntk_py.set_fixed_random_seed(1) # fix the random seed so that LR examples are repeatable
from IPython.display import Image
import matplotlib.pyplot

#------1. Data ------
input_shape_mn = (28, 28)  # MNIST digits are 28 x 28
num_classes_mn = 10        # classify as one of 10 digits
# Fetch the MNIST data. Best done with scikit-learn
try:
	from sklearn import datasets, utils
	mnist = datasets.fetch_mldata("MNIST original")
	X, Y = mnist.data / 225.0, mnist.target
	X_train_mn, X_test_mn = X[:60000].reshape((-1,28,28)),X[60000:].reshape((-1,28,28))
	Y_train_mn, Y_test_mn = Y[:60000].astype(int), Y[60000:].astype(int)
except: # workaround if scikit-learn is not present
	import requests, io, gzip
	#X_train_mn, X_test_mn = (np.fromstring(gzip.GzipFile(fileobj=io.BytesIO
	#                        (requests.get('http://yann.lecun.com/exdb/mnist/' + name + '-images-idx3-ubyte.gz').content))
	#						.read()[16:], dtype=np.uint8).reshape((-1,28,28)).astype(np.float32) / 255.0 for name in ('train', 't10k'))
	#Y_train_mn, Y_test_mn = (np.fromstring(gzip.GzipFile(fileobj=io.BytesIO
	#                        (requests.get('http://yann.lecun.com/exdb/mnist/' + name + '-labels-idx1-ubyte.gz').content))
	#						.read()[8:], dtype=np.uint8).astype(int) for name in ('train', 't10k'))
	print("loading finished")
	X_train_mn, X_test_mn = (np.fromstring(gzip.open('E:/ProgramLib/Python/CNTK/testdata/' + name + '-images-idx3-ubyte.gz')
							.read()[16:], dtype=np.uint8).reshape((-1,28,28)).astype(np.float32) / 255.0 for name in ('rain', 't10k'))
	Y_train_mn, Y_test_mn = (np.fromstring(gzip.open('E:/ProgramLib/Python/CNTK/testdata/' + name + '-labels-idx1-ubyte.gz')
							.read()[8:], dtype=np.uint8).astype(int) for name in ('rain', 't10k'))
	print("loading local files finised")

# Shuffle the training data.
np.random.seed(0)  # always use the same reordering, for reproducability
idx = np.random.permutation(len(X_train_mn))
X_train_mn, Y_train_mn = X_train_mn[idx], Y_train_mn[idx]

# Further split off a cross-validation set
X_train_mn, X_cv_mn = X_train_mn[:54000], X_train_mn[54000:]
Y_train_mn, Y_cv_mn = Y_train_mn[:54000], Y_train_mn[54000:]

# Our model expects float32 features, and cross-entropy expects one-hot encoded labels.
Y_train_mn, Y_cv_mn, Y_test_mn = (scipy.sparse.csr_matrix((np.ones(len(Y),np.float32), (range(len(Y)), Y)), shape=(len(Y), 10)) for Y in (Y_train_mn, Y_cv_mn, Y_test_mn))
X_train_mn, X_cv_mn, X_test_mn = (X.astype(np.float32) for X in (X_train_mn, X_cv_mn, X_test_mn))

# Have a peek.
import pylab
matplotlib.pyplot.rcParams['figure.figsize'] = (5, 0.5)
matplotlib.pyplot.axis('off')
graph = matplotlib.pyplot.imshow(np.concatenate(X_train_mn[0:10], axis=1), cmap="gray_r")
pylab.show()


#------2.Model------
# Create model and criterion function.
def create_model_mn_factory():
    with cntk.layers.default_options(activation=cntk.ops.relu, pad=False):
        return cntk.layers.Sequential([
            cntk.layers.Convolution2D((5,5), num_filters=32, reduction_rank=0, pad=True), # reduction_rank=0 for B&W images
            cntk.layers.MaxPooling((3,3), strides=(2,2)),
            cntk.layers.Convolution2D((3,3), num_filters=48),
            cntk.layers.MaxPooling((3,3), strides=(2,2)),
            cntk.layers.Convolution2D((3,3), num_filters=64),
            cntk.layers.Dense(96),
            cntk.layers.Dropout(dropout_rate=0.5),
            cntk.layers.Dense(num_classes_mn, activation=None) # no activation in final layer (softmax is done in criterion)
        ])
model_mn = create_model_mn_factory()

#------3.Training------
x = cntk.input_variable(input_shape_mn)
y = cntk.input_variable(num_classes_mn, is_sparse=True)

@cntk.Function
def criterion_mn_factory(data, label_one_hot):
    z = model_mn(data)
    loss = cntk.cross_entropy_with_softmax(z, label_one_hot)
    metric = cntk.classification_error(z, label_one_hot)
    return loss, metric

criterion_mn = criterion_mn_factory(x, y)

# Create the learner.
N = len(X_train_mn)
lrs = cntk.learning_parameter_schedule_per_sample([0.001]*12 + [0.0005]*6 + [0.00025]*6 + [0.000125]*3 + [0.0000625]*3 + [0.00003125], epoch_size=N)
momentums = cntk.momentum_schedule_per_sample([0]*5 + [0.9990239141819757], epoch_size=N)
minibatch_sizes = cntk.minibatch_size_schedule([256]*6 + [512]*9 + [1024]*7 + [2048]*8 + [4096], epoch_size=N)

learner = cntk.learners.momentum_sgd(model_mn.parameters, lrs, momentums)

# Create progress callbacks for logging to file and TensorBoard event log.
# Prints statistics for the first 10 minibatches, then for every 50th, to a log file.
progress_writer = cntk.logging.ProgressPrinter(50, first=10, log_to_file='E:/ProgramLib/Python/CNTK/model/CNTK-200-my.log')
tensorboard_writer = cntk.logging.TensorBoardProgressWriter(50, log_dir='E:/ProgramLib/Python/CNTK/model/CNTK-200-my_tensorboard_logdir',
                                                            model=criterion_mn)

# Create a checkpoint callback.
# Set restore=True to restart from available checkpoints.
epoch_size = len(X_train_mn)
checkpoint_callback_config = cntk.CheckpointConfig('E:/ProgramLib/Python/CNTK/model/CNTK-200-model_mn.cmf', epoch_size, preserve_all=True, restore=False)

# Create a cross-validation based training control.
# This callback function halves the learning rate each time the cross-validation metric
# improved less than 5% relative, and stops after 6 adjustments.
prev_metric = 1 # metric from previous call to the callback. Error=100% at start.
def adjust_lr_callback(index, average_error, cv_num_samples, cv_num_minibatches):
    global prev_metric
    if (prev_metric - average_error) / prev_metric < 0.05: # did metric improve by at least 5% rel?
        learner.reset_learning_rate(cntk.learning_parameter_schedule(learner.learning_rate() / 2, minibatch_size=1))
        if learner.learning_rate() < lrs[0] / (2**7-0.1): # we are done after the 6-th LR cut
            print("Learning rate {} too small. Training complete.".format(learner.learning_rate()))
            return False # means we are done
        print("Improvement of metric from {:.3f} to {:.3f} insufficient. Halving learning rate to {}.".format(prev_metric, average_error, learner.learning_rate()))
    prev_metric = average_error
    return True # means continue

cv_callback_config = cntk.CrossValidationConfig((X_cv_mn, Y_cv_mn), 3*epoch_size, minibatch_size=256,
                                                callback=adjust_lr_callback, criterion=criterion_mn)

# Callback for testing the final model.
test_callback_config = cntk.TestConfig((X_test_mn, Y_test_mn), criterion=criterion_mn)

# Train!
callbacks = [progress_writer, tensorboard_writer, checkpoint_callback_config, cv_callback_config, test_callback_config]
progress = criterion_mn.train((X_train_mn, Y_train_mn), minibatch_size=minibatch_sizes,
                              max_epochs=10, parameter_learners=[learner], callbacks=callbacks)

# Progress is available from return value
losses = [summ.loss for summ in progress.epoch_summaries]
print('loss progression =', ", ".join(["{:.3f}".format(loss) for loss in losses]))
