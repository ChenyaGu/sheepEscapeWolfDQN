import unittest
from ddt import ddt, data, unpack
from src.initialPosition import *


@ddt
class TestInitPos(unittest.TestCase):
    def setUp(self):
        self.movingRange = [0, 0, 364, 364]
        self.minDistanceEachOther = 50
        self.maxDistanceEachOther = 180
        self.minDistanceWolfSheep = 120
        self.numberObjects = 6

    @data(([0, 0], [1, 1], 2 ** 0.5),
          ([1, -1], [-1, 1], 8 ** 0.5))
    @unpack
    def testComputeDistance(self, p1, p2, groundTruthDis):
        # pass
        distance = computeDistance(p1, p2)
        truthValue = np.array_equal(distance, groundTruthDis)
        self.assertTrue(truthValue)

    @data((50, 180, 120))
    @unpack
    def testInitialPosition(self, compareDis1, compareDis2, compareDis3):
        # pass
        initialPosition = InitialPosition(self.movingRange, self.minDistanceEachOther, self.maxDistanceEachOther,
                                          self.minDistanceWolfSheep)
        positionList = initialPosition(self.numberObjects)
        distance = computeDistance(positionList[0], positionList[1])
        truthValue = (distance > compareDis1) & (distance < compareDis2) & (distance > compareDis3)
        self.assertTrue(truthValue)


if __name__ == '__main__':
    unittest.main()
