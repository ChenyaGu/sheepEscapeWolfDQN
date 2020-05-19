import tensorflow as tf
import numpy as np
from collections import deque
import random
import os
import gym

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from buildModel import *

bufferSize = 2000
batchSize = 1
updateFrequency = 5
epsilon = 0.9
learningRate = 0.1
layerWidths = [10, 10, 10]

buildModel = BuildModel(4, 2, 0.8)
episodeRange = 100
scoreList = []
model = buildModel(layerWidths)
train = TrainDQN(batchSize, updateFrequency, learningRate, 1)
replayBuffer = deque(maxlen=bufferSize)
env = gym.make('CartPole-v0')
for episode in range(episodeRange):
    states = env.reset()
    score = 0
    while True:
        action = sampleAction(model, states)
        nextStates, reward, done, _ = env.step(action)
        replayBuffer = memorize(replayBuffer, states, action, nextStates, reward)
        miniBatch = random.sample(replayBuffer, 1)
        train(model, miniBatch)
        score += reward
        states = nextStates
        if done:
            scoreList.append(score)
            print('episode:', episode, 'score:', score, 'max:', max(scoreList))
            break
env.close()

import matplotlib.pyplot as plt

plt.plot(scoreList, color='green')
plt.show()


