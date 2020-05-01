import numpy as np
import random
import os
import csv
from collections import deque
import functools as ft

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K

from viz import *
from reward import *
from gridworld import *
from BeliefUpdate import *
from PreparePolicy import *
from InitialPosition import *
import Transition
import Attention

class Remember:
    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001

    def __call__(self, state, action, reward, next_state, done, memory):
        memory.append((state, action, reward, next_state, done))
        return memory


class Replay:
    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001

    def __call__(self, minibatch, model, targetModel):
        # minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = targetModel.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)

                states.append(state[0])
                targets.append(target[0])

        states_mb = np.array(states)
        targets_mb = np.array(targets)
        return states_mb, targets_mb