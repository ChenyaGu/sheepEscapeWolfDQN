import pandas as pd
import numpy as np
import random
from src.initialPosition import *


def normalizeVector(rawVector, targetLength):
    rawLength = np.linalg.norm(rawVector)
    changeRate = np.divide(targetLength, rawLength)
    return np.multiply(rawVector, changeRate)


class SelectAction:
    def __init__(self, stateSize, actionSize, model, epsilon):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.model = model
        self.epsilon = epsilon

    def __call__(self, currentState):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.actionSize)
            return action
        else:
            actionValues = self.model.predict(currentState)
            action = np.argmax(actionValues[0])
            return action


class Transition:
    def __init__(self, movingRange, speedList, actionSize):
        self.movingRange = movingRange
        self.speedList = speedList
        self.actionSize = actionSize

    def __call__(self, currentState, currentAction, currentVelocity, identity):
        minX = 0
        minY = 1
        maxX = 2
        maxY = 3
        angleList = [i * (2 * np.pi / self.actionSize) for i in range(0, self.actionSize)]
        actionList = [np.array((round(np.cos(actionAngles), 10), round(np.sin(actionAngles), 10)))
                      for actionAngles in angleList]
        currentPosition = np.array(currentState)
        newVelocity = normalizeVector(np.add(currentVelocity, actionList[currentAction])
                                      , self.speedList[identity])
        newPosition = currentPosition + newVelocity
        if newPosition[0] > self.movingRange[maxX]:
            newPosition[0] = 2 * self.movingRange[maxX] - newPosition[0]
        if newPosition[0] < self.movingRange[minX]:
            newPosition[0] = 2 * self.movingRange[minX] - newPosition[0]
        if newPosition[1] > self.movingRange[maxY]:
            newPosition[1] = 2 * self.movingRange[maxY] - newPosition[1]
        if newPosition[1] < self.movingRange[minY]:
            newPosition[1] = 2 * self.movingRange[minY] - newPosition[1]
        newVelocity = newPosition - currentPosition
        print(newVelocity)
        newState = newPosition
        return newState


class Reward:
    def __init__(self, sheepID, wolfID):
        self.sheepID = sheepID
        self.wolfID = wolfID

    def __call__(self, state, belief, minDis):
        endPunishment = -100
        isEnd = IsEnd(state, self.sheepID, self.wolfID)
        if isEnd(minDis):
            return endPunishment
        else:
            sheepPos = state[self.sheepID]
            targetPosList = state[self.wolfID:]
            distanceList = [computeDistance(sheepPos, target) for target in targetPosList]
            subtletyList = [belief[i][1] for i in range(len(belief))]

            def disReward(distance, subtlety, const=1):
                reward = const * distance * subtlety
                return reward

            distanceReward = np.sum(
                [disReward(distance, subtlety) for (distance, subtlety) in zip(distanceList, subtletyList)])

            # def sigmoid(x, m=1, s=1):
            #     ePower = np.exp(-s * x)
            #     sigmoidValue = m / (1.0 + ePower)
            #     return sigmoidValue
            #
            # def barrierPunish():
            #     return
            #
            # barrierPunishment = barrierPunish()
            # totalReward = distanceReward - barrierPunishment
            return distanceReward


class IsEnd:
    def __init__(self, state, sheepID, wolfID):
        self.state = state
        self.sheepID = sheepID
        self.wolfID = wolfID

    def __call__(self, minDis):
        sheepCoordinates = self.state[self.sheepID]
        wolfCoordinates = self.state[self.wolfID]
        if computeDistance(sheepCoordinates, wolfCoordinates) <= minDis:
            return True
        else:
            return False

# minibatch = random.sample(memory, batch_size)
