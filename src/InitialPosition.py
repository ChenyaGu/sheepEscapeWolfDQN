import numpy as np
import itertools as it


def samplePosition(positionRange):
    positionX = np.random.uniform(positionRange[0], positionRange[2])
    positionY = np.random.uniform(positionRange[1], positionRange[3])
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

    def __call__(self, numberObjects):
        positionList = [samplePosition(self.movingRange) for i in range(numberObjects)]
        pairList = list(it.combinations(range(numberObjects), 2))
        sampleCount = 1
        while sampleCount < self.totalNum:
            distanceEachOtherArray = np.array(
                [computeDistance(positionList[index[0]], positionList[index[1]]) for index in pairList])
            distanceWolfSheep = computeDistance(positionList[0], positionList[1])
            if (distanceWolfSheep > self.minDistanceWolfSheep) \
                    & np.all(distanceEachOtherArray > self.minDistanceEachOther) \
                    & np.all(distanceEachOtherArray < self.maxDistanceEachOther):
                break
            else:
                positionList = [samplePosition(self.movingRange) for i in range(numberObjects)]
                sampleCount = sampleCount + 1
        if sampleCount == self.totalNum:
            print("unable to initial correct positionList")
            return False
        else:
            print(sampleCount)
            return positionList

# movingRange = [0, 0, 364, 364]
# minDistanceEachOther = 50
# maxDistanceEachOther = 180
# minDistanceWolfSheep = 120
# numberObjects = 6
#
# initialPosition = InitialPosition(movingRange, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep)
# positionList = initialPosition(numberObjects)
