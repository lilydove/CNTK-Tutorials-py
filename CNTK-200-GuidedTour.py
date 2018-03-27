# https://cntk.ai/pythondocs/CNTK_200_GuidedTour.html
# CNTK 200: A Guided Tour

from __future__ import print_function
import cntk
import numpy as np
import scipy.sparse
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
cntk.cntk_py.set_fixed_random_seed(1) # fix the random seed so that LR examples are repeatable
from IPython.display import Image
import matplotlib.pyplot
# %matplotlib inline
matplotlib.pyplot.rcParams['figure.figsize'] = (40,40)

# *Conceptual* numpy implemetation of CNTK's Dense layer (simplified,e.g. no back-prop)
def Dense(out_dim, activation):
	# create the learnable parameters
	b = np.zeros(out_dim)
	W = np.ndarray((0,out_dim)) # input dimension is unknown
	# define the function itself
	def dense(x):
		if len(W) == 0: # first call :reshape and initialize W
			W.resize((x.shape[-1], W.shape[-1]), refcheck=False)
			print (x.shape[-1])
			W[:] = np.random.randn(*W.shape) * 0.05
		return activation(x.dot(W) + b)
	# return as function object: can be called & holds parameters as members
	dense.W = W
	dense.b = b
	return dense

d = Dense(5, np.tanh)       # create the function object
y = d(np.array([1,2,3]))    # apply it like a function
W = d.W                     # access member like an object , dim(x) = 3, dim(y) = 5
print('W = ', d.W)
print('y = ', y)


# Your First CNTK Network: Simple Logistic Regression
print ("Simple Logistic Regression:")
input_dim_lr = 2    # classify 2-dimensional data
num_classes_lr = 2  # into one of two classes
# ------1. Data ------
# This example uses synthetic data from normal distributions,
# which we generate in the following.
#  X_lr[corpus_size,input_dim] - input data
#  Y_lr[corpus_size]           - labels (0 or 1), one-hot-encoded
np.random.seed(0)
def generate_synthetic_data(N):
    Y = np.random.randint(size=N, low=0, high=num_classes_lr)  # labels
    X = (np.random.randn(N, input_dim_lr)+3) * (Y[:, None]+1)  # data
    # Our model expects float32 features, and cross-entropy
    # expects one-hot encoded labels.
    Y = scipy.sparse.csr_matrix((np.ones(N,np.float32), (range(N), Y)), shape=(N, num_classes_lr))
    X = X.astype(np.float32)
    return X, Y
X_train_lr, Y_train_lr = generate_synthetic_data(20000)
X_test_lr, Y_test_lr = generate_synthetic_data(1024)
print('data = \n',X_train_lr[:4])
print('labels = \n', Y_train_lr[:4].todense())
#------2. Model ------
model_lr_factory = cntk.layers.Dense(num_classes_lr, activation=None)
x = cntk.input_variable(input_dim_lr)
y = cntk.input_variable(num_classes_lr, is_sparse=True)
model_lr = model_lr_factory(x)

#------3. Training ------
"""
To avoid evaluating the model twice, 
we use a Python function definition with decorator syntax. 
This is also a good time to tell CNTK about the data types of our inputs,
which is done via the decorator @Function:
"""
@cntk.Function
def criterion_lr_factory(data, label_one_hot):
	z = model_lr_factory(data)  # apply model. Computes a non-normalize log probability for every output class.
	loss = cntk.cross_entropy_with_softmax(z, label_one_hot) # applies softmax to z under the hood
	metric = cntk.classification_error(z, label_one_hot)
	return loss, metric
criterion_lr = criterion_lr_factory(x, y)
print('criterion_lr:', criterion_lr)

learner = cntk.sgd(model_lr.parameters,
                   cntk.learning_parameter_schedule(0.1))
progress_writer = cntk.logging.ProgressPrinter(0)

criterion_lr.train((X_train_lr, Y_train_lr), parameter_learners=[learner],
                   callbacks=[progress_writer])

print(model_lr.W.value)  # peek at updated W

test_metric_lr = criterion_lr.test((X_test_lr, Y_test_lr),
                                    callbacks=[progress_writer]).metric
model_lr = model_lr_factory(x)
print('model_lr:', model_lr)

z = model_lr(X_test_lr[:20])
print("Label    :", [label.todense().argmax() for label in Y_test_lr[:20]])
print("Predicted:", [z[i,:].argmax() for i in range(len(z))])

input_shape_mn = (28, 28)  # MNIST digits are 28*28
num_classes_mn = 10        # classify as one of 10 digits


#------Your Second CNTK Network: MNIST Digit Recognition------
"""
------1. Data------
1.data

------2. Model------
2.model:dense,Convolution2D,MaxPooling,Dropout,Dense
3.loss:cntk.cross_entropy_with_softmax(z, label_one_hot)
4.metric:cntk.classification_error(z, label_one_hot)

------3. Parameter------
5.feed data structure:cntk.input_variable
6.lr:cntk.learning_parameter_schedule_per_sample
7.monemtum:cntk.momentum_schedule_per_sample
8.minibatch:cntk.minibatch_size_schedule

------4. Solver------
9.learner: cntk.learners.momentum_sgd(model_mn.parameters, lrs, momentums)

------5. Monitor------
10.logging:cntk.logging.ProgressPrinte()

------6. Running------
11.Function.train((X_train_mn, Y_train_mn), minibatch_size=minibatch_sizes,
                   max_epochs=40, parameter_learners=[learner], 
				   callbacks=[progress_writer])
"""
#------1. Data ------
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

#------2. Model ------
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

#------3. Training ------
"""
The decorator will 'compile' the Python function into CNTK's internal graph representation.
Thus, the resulting criterion not a Python function but a CNTK Function object.
"""
@cntk.Function
def criterion_mn_factory(data, label_one_hot):
    z = model_mn(data)
    loss = cntk.cross_entropy_with_softmax(z, label_one_hot)
    metric = cntk.classification_error(z, label_one_hot)
    return loss, metric
x = cntk.input_variable(input_shape_mn)
y = cntk.input_variable(num_classes_mn, is_sparse=True)
criterion_mn = criterion_mn_factory(x,y)

"""
the learning rate is specified as a list
([0.001]*12 + [0.0005]*6 +...). 
Together with the epoch_size parameter, 
this tells CNTK to use 0.001 for 12 epochs, 
and then continue with 0.005 for another 6, etc.
"""
N = len(X_train_mn)
lrs = cntk.learning_parameter_schedule_per_sample([0.001]*12 + [0.0005]*6 + [0.00025]*6 + [0.000125]*3 + [0.0000625]*3 + [0.00003125], epoch_size=N)
momentums = cntk.momentum_schedule_per_sample([0]*5 + [0.9990239141819757], epoch_size=N)
minibatch_sizes = cntk.minibatch_size_schedule([256]*6 + [512]*9 + [1024]*7 + [2048]*8 + [4096], epoch_size=N)

learner = cntk.learners.momentum_sgd(model_mn.parameters, lrs, momentums)
progress_writer = cntk.logging.ProgressPrinter()
# Start to train
criterion_mn.train((X_train_mn, Y_train_mn), minibatch_size=minibatch_sizes,
                   max_epochs=10, parameter_learners=[learner], callbacks=[progress_writer])
test_metric_mn = criterion_mn.test((X_test_mn, Y_test_mn), callbacks=[progress_writer]).metric