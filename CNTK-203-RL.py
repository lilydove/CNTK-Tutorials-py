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

try:
    import gym
except:
    #pip install gym
    import gym

style.use('ggplot')
Image(url="https://cntk.ai/jup/polecart.git", width=300, height=300)

isFast = True

# ------1.Data------


