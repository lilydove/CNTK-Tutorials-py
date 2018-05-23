# https://cntk.ai/pythondocs/CNTK_203_Reinforcement_Learning_Basics.html
# CNTK 203: Reinforcement Learning Basics

from IPython.display import Image
from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd 
import seaborn as sns
import cntk as C
import os

try:
    import gym
except:
    #pip install gym
    import gym

style.use('ggplot')
Image(url="https://cntk.ai/jup/polecart.git", width=300, height=300)

isFast = True
# Select the right target device when this notebook is being tested:
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(c.device.gup(0))

env = gym.make('CartPole-v0')

STATE_COUNT = env.observation_space.shape[0]
ACTION_COUNT = env.action_space.n

# Targetted reward
REWARD_TARGET = 30 if isFast else 200
# Averaged over these these many episodes
BATCH_SIZE_BASELINE = 20 if isFast else 50

H = 64 # hidden layer size
# ------1.Data------


