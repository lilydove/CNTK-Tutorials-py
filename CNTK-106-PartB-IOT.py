# https://cntk.ai/pythondocs/CNTK_106B_LSTM_Timeseries_with_IOT_Data.html# author:wuyi
# date:2018-03-15--16
# Part B - Time series prediction with LSTM (IOT)
#  
# 
"""
Goal

Using historic daily production of a solar panel, 
we want to predict the total power production of the solar panel array for a day. 
We will be using the LSTM based time series prediction model developed in part A 
to predict the daily output of a solar panel based on the initial readings of the day.
"""
"""
Mechine learning
1.Object:  The goal you want
2.Data:    The things you have
3.Model:   The tool you use to get your goal
4.Learner: The way you build the tool

Working contents
1.Gathering / Preprocessing data
2.Selecting /Creating model
3.Tuning parameters
4.Training 
5.Testing
6.Application
"""

from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import time

import cntk as C

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import cntk.tests.test_utils

# cntk.tests.test_utils.set_device_from_pytest_env() #(only needed for our build system)

# to make things reproduceable, seed random
np.random.seed(0)

isFast = True
isTrain = False
isTest = False
modelFile = r'E:\ProgramLib\Python\CNTK\model\IOT_LSTM.cmf'

# we need around 2000 epochs to see good accuracy. For testing 100 epochs will do.
EPOCHS = 100 if isFast else 2000


# ------1. Data generation ------
# Pre-processing
def generate_solar_data(input_url, time_steps, normalize=1, val_size=0.1, test_size=0.1):
    """
    generate sequences to feed to rnn based on data frame with solar panel data
    the csv has the format: time ,solar.current, solar.total
     (solar.current is the current output in Watt, solar.total is the total production
      for the day so far in Watt hours)

	CVS:
		time,               solar.current,    solar.total
        2013-12-02 7am,     6.3,              1.7
        2013-12-02 7:30am,  44.3,             11.4
...
    """
    # try to find the data file local. If it doesn't exist, download it.
    data_dir = r"E:\ProgramLib\Python\CNTK\testdata"
    cache_path = os.path.join(data_dir, "Solar")
    cache_file = os.path.join(cache_path, "solar.csv")
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(cache_file):
        urlretrieve(input_url, cache_file)
        print("dowloaded data successfully from ", input_url)
    else:
        print("using cache for", input_url)

    df = pd.read_csv(cache_file, index_col="time", parse_dates=['time'], dtype=np.float32)

    df["date"] = df.index.date  # extract the date from the index of "time" to a new colmun of 'date'

    # normalize data
    df['solar.current'] /= normalize
    df['solar.total'] /= normalize

    # group by day, find the max for a day and a new column. max
    grouped = df.groupby(df.index.date).max()
    grouped.columns = ["solar.current.max", "solar.total.max", "date"]

    # merge continous readings and daily max values into a single frame
    df_merged = pd.merge(df, grouped, right_index=True, on="date")
    df_merged = df_merged[["solar.current", "solar.total",
                           "solar.current.max", "solar.total.max"]]

    # we group by day so we can process a day at a time
    grouped = df_merged.groupby(df_merged.index.date)
    per_day = []
    for _, group in grouped:
        per_day.append(group)
	# print(per_day[0])
	# print("per_day: ", len(per_day))

    # split the dataset into train, validatation and test sets on day boundaries
    val_size = int(len(per_day) * val_size)
    test_size = int(len(per_day) * test_size)
    next_val = 0
    next_test = 0

    result_x = {"train": [], "val": [], "test": []}
    result_y = {"train": [], "val": [], "test": []}

    # generate sequences a day at a time
	# pick in sequence, 8 values for training, 1 for validation and 1 for test until there is no more data
    for i, day in enumerate(per_day):
        # if we have less than 8 datapoints for a day we skip over the
        # day assuming something is missing in the raw data
        total = day["solar.total"].values
        if len(total) < 8:
            continue
        if i >= next_val:
            current_set = "val"
            next_val = i + int(len(per_day) / val_size)
        elif i >= next_test:
            current_set = "test"
            next_test = i + int(len(per_day) / test_size)
        else:
            current_set = "train"
        max_total_for_day = np.array(day["solar.total.max"].values[0])
        for j in range(2, len(total)):
            result_x[current_set].append(total[0:j])
            result_y[current_set].append([max_total_for_day])
            if j >= time_steps:
                break
    # make result_y a numpy array
    for ds in ["train", "val", "test"]:
        result_y[ds] = np.array(result_y[ds])
    return result_x, result_y

# Data caching
# We keep upto 14 inputs from a day
TIMESTEPS = 14
# 20000 is the maximum total output in our dataset. We normalize all values with
# this so our inputs are between 0.0 and 1.0 range.
NORMALIZE = 20000

X, Y = generate_solar_data("https://www.cntk.ai/jup/dat/solar.csv",
                           TIMESTEPS, normalize=NORMALIZE)
						 
print(X['train'][0:3])
print(Y['train'][0:3])

# process batches of 10 days
BATCH_SIZE = TIMESTEPS * 10

def next_batch(x, y, ds):
    """get the next batch for training"""

    def as_batch(data, start, count):
        return data[start:start + count]

    for i in range(0, len(x[ds]), BATCH_SIZE):
        yield as_batch(X[ds], i, BATCH_SIZE), as_batch(Y[ds], i, BATCH_SIZE)

# ------2. Model Creation (LSTM network) ------
"""
1.7,11.4 -> 10300
1.7,11.4,67.5 -> 10300
1.7,11.4,67.5,250.5 ... -> 10300
1.7,11.4,67.5,250.5,573.5 -> 10300
"""
# Specify the internal-state dimensions of the LSTM cell
H_DIMS = 15

def create_model(x):
    """Create the model for time series prediction"""
    with C.layers.default_options(initial_state=0.1):
        m = C.layers.Recurrence(C.layers.LSTM(H_DIMS))(x)  # parametres + inputs
        m = C.sequence.last(m)
        m = C.layers.Dropout(0.2)(m)
        m = C.layers.Dense(1)(m)
        return m

#------3. Training ------
#1:input 2:model 3:output 4:superparameters 5:loss 6:error 7:optimizer
# input sequences
x = C.sequence.input_variable(1)  # 1 dimention

# create the model
z = create_model(x)

# expected output (label), also the dynamic axes of the model output
# is specified as the model of the label input
l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name='y')

# the learning rate
learning_rate = 0.005
lr_schedule = C.learning_parameter_schedule(learning_rate)

# loss function
loss = C.squared_error(z, l)

# use squared error to determine error for now
error = C.squared_error(z, l)

# use adam optimizer
momentum_schedule = C.momentum_schedule(0.9, minibatch_size=BATCH_SIZE)
learner = C.fsadagrad(z.parameters,
                      lr = lr_schedule,
					  momentum = momentum_schedule)
trainer = C.Trainer(z, (loss, error), [learner]) # data model losser learner

# Time to start training
# training

loss_summary = []
if isTrain :
    start = time.time()
    for epoch in range(0, EPOCHS):
        for x_batch, l_batch in next_batch(X, Y, "train"):
            trainer.train_minibatch({x: x_batch, l: l_batch})

        if epoch % (EPOCHS / 10) == 0:
            training_loss = trainer.previous_minibatch_loss_average
            loss_summary.append(training_loss)
            print("epoch: {}, loss:{:.4f}".format(epoch, training_loss))

    print("Training book {:.1f} sec".format(time.time() - start))

    z.save(modelFile)
    print ("Saving model")
    plt.plot(loss_summary, label='training loss')
    plt.show()

#------4. Testing ------
# validate
if isTest:
    def get_mse(X,Y,labeltxt):
        result = 0.0
        for x1, y1 in next_batch(X, Y, labeltxt):
            eval_error = trainer.test_minibatch({x : x1, l : y1})
            result += eval_error
        return result/len(X[labeltxt])

# Print the train and validation errors
    for labeltxt in ["train", "val"]:
        print("mse for {}: {:.6f}".format(labeltxt, get_mse(X, Y, labeltxt)))

# Print the test error
    labeltxt = "test"
    print("mse for {}: {:.6f}".format(labeltxt, get_mse(X, Y, labeltxt)))

#------5. Prediction ------
# predict
if not isTrain:
    z = C.Function.load(modelFile)
    
f, a = plt.subplots(2, 1, figsize=(12, 8))
for j, ds in enumerate(["val", "test"]):
    results = []
    for x_batch, _ in next_batch(X, Y, ds):
        pred = z.eval({x: x_batch})
        results.extend(pred[:, 0])
    # because we normalized the input data we need to multiply the prediction
    # with SCALER to get the real values.
    a[j].plot((Y[ds] * NORMALIZE).flatten(), label=ds + ' raw')
    a[j].plot(np.array(results) * NORMALIZE, label=ds + ' pred')
    a[j].legend()
    plt.show()