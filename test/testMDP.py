import unittest
from ddt import ddt, data, unpack
from src.MDP import *


@ddt
class TestMDP(unittest.TestCase):
    def setUp(self):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.model =
        self.epsilon = 0.05

    @data(([0, 0], [1, 1], 2 ** 0.5),
          ([1, -1], [-1, 1], 8 ** 0.5))
    @unpack
    def testSelectAction(self, p1, p2, groundTruthDis):
        # pass?
        initialPosition = InitialPosition(self.movingRange, self.minDistanceEachOther, self.maxDistanceEachOther,
                                          self.minDistanceWolfSheep)
        positionList = initialPosition(self.numberObjects)


    @data(([0, 0], [1, 1], 2 ** 0.5),
          ([1, -1], [-1, 1], 8 ** 0.5))
    @unpack
    def testTransition(self, p1, p2, groundTruthDis):


    @data(([0, 0], [1, 1], 2 ** 0.5),
          ([1, -1], [-1, 1], 8 ** 0.5))
    @unpack
    def testReward(self, p1, p2, groundTruthDis):


    @data(([0, 0], [1, 1], 2 ** 0.5),
      ([1, -1], [-1, 1], 8 ** 0.5))
    @unpack
    def testIsEnd(self, p1, p2, groundTruthDis):
