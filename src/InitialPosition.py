import numpy as np
import itertools as it


def samplePosition(positionRange):
    minX = 0
    minY = 1
    maxX = 2
    maxY = 3
    positionX = np.random.uniform(positionRange[minX], positionRange[maxX])
    positionY = np.random.uniform(positionRange[minY], positionRange[maxY])
    position = [positionX, positionY]
    return position


def computeDistance(position1, position2):
    diff = np.asarray(position1) - np.asarray(position2)
    distance = np.linalg.norm(diff)
    return distance


class InitialPosition:
    def __init__(self, movingRange, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep):
        self.movingRange = movingRange
        self.minDistanceEachOther = minDistanceEachOther
        self.maxDistanceEachOther = maxDistanceEachOther
        self.minDistanceWolfSheep = minDistanceWolfSheep
        self.totalNum = 100000
        self.sheepId = 0
        self.wolfId = 1

    def __call__(self, objectNum):
        positionList = [samplePosition(self.movingRange) for i in range(objectNum)]
        pairList = list(it.combinations(range(objectNum), 2))
        sampleCount = 1
        while sampleCount < self.totalNum:
            distanceEachOtherArray = np.array(
                [computeDistance(positionList[index[0]], positionList[index[1]]) for index in pairList])
            distanceWolfSheep = computeDistance(positionList[self.sheepId], positionList[self.wolfId])
            if (distanceWolfSheep > self.minDistanceWolfSheep) \
                    & np.all(distanceEachOtherArray > self.minDistanceEachOther) \
                    & np.all(distanceEachOtherArray < self.maxDistanceEachOther):
                break
            else:
                positionList = [samplePosition(self.movingRange) for i in range(objectNum)]
                sampleCount = sampleCount + 1
        if sampleCount == self.totalNum:
            print("unable to initial correct positionList")
            return False
        else:
            return positionList
