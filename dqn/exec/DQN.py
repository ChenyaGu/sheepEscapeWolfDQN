import tensorflow as tf
import numpy as np
import sys
from collections import deque
import os
import gym
import random
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))


from src.buildModelRL import *
from env.discreteMountainCarEnv import *
from env.discreteCartPole import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# envName = 'MountainCar-v0'
envName = 'CartPole-v0'
env = gym.make(envName)
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.n
seed = 1
bufferSize = 10000
maxReplaySize = 64
batchSize = 32
gamma = 0.9
epsilon = 0.6
learningRate = 0.001
layerWidths = [30]
scoreList = []
episodeRange = 1000
actionDelay = 2
updateFrequency = 1

# transit = TransitMountCarDiscrete()
# reset = ResetMountCarDiscrete(seed)
# getReward = rewardMountCarDiscrete
# isTerminal = IsTerminalMountCarDiscrete()

transit = TransitCartPole()
reset = ResetCartPole(seed)
getReward = RewardCartPole()
isTerminal = IsTerminalCartPole()

sampleAction = SampleAction(actionDim)
initializeReplayBuffer = InitializeReplayBuffer(reset, transit, getReward, isTerminal, actionDim)
buildModel = BuildModel(stateDim, actionDim, gamma)
trainOneStep = TrainOneStep(batchSize, updateFrequency, learningRate, gamma)
model = buildModel(layerWidths)
replayBuffer = deque(maxlen=bufferSize)
replayBuffer = initializeReplayBuffer(replayBuffer, maxReplaySize)
miniBatch = sampleData(replayBuffer, batchSize)
forwardOneStep = ForwardOneStep(transit, getReward, isTerminal)
runTimeStep = RunTimeStep(forwardOneStep, sampleAction, trainOneStep, batchSize, epsilon, actionDelay, actionDim)
runEpisode = RunEpisode(reset, runTimeStep)
runAlgorithm = RunAlgorithm(episodeRange, runEpisode)
model, scoreList, trajectory = runAlgorithm(model, replayBuffer)


env.close()

import matplotlib.pyplot as plt

plt.plot(scoreList, color='green')
plt.show()

showDemo = False
if showDemo:
    visualize = VisualizeCartPole()
    visualize(trajectory)