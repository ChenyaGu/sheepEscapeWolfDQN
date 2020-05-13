import unittest
from ddt import ddt, data, unpack
from src.MDP import *


@ddt
class TestMDP(unittest.TestCase):
    def setUp(self):
        self.sheepId = 0
        self.wolfId = 1
        self.movingRange = [0, 0, 15, 15]
        self.speedList = [8, 9, 10]
        self.currentVelocity = [0, 0]
        # self.stateSize = stateSize
        self.actionSize = 16
        # self.model = model
        self.epsilon = 0.05

    # @data(([0, 0], [1, 1], 2 ** 0.5),
    #       ([1, -1], [-1, 1], 8 ** 0.5))
    # @unpack
    # def testSelectAction(self, p1, p2, groundTruthDis):
    #     # pass?
    #    transition = Transition(self.movingRange, self.speedList, self.actionSize)
    #    newState = transition(currentState, currentAction, currentVelocity, identity)
    #    truthValue = np.array_equal(newState, groundTruthState)
    #    self.assertTrue(truthValue)

    @data(([10, 10], 0, [12, 10]),
          ([0, 0], 8, [8, 0]))
    @unpack
    def testTransition(self, currentState, currentAction, groundTruthState):
        # pass
        currentVelocity = self.currentVelocity
        identity = self.sheepId
        transition = Transition(self.movingRange, self.speedList, self.actionSize)
        newState = transition(currentState, currentAction, currentVelocity, identity)
        truthValue = np.array_equal(newState, groundTruthState)
        self.assertTrue(truthValue)

    @data((([0, 0], [0, 1], [1, 1]), ([1, 50], [0, 11]), 1, -100),
          (([0, 0], [0, 2], [0, 1]), ([1, 50], [0, 11]), 1, 111))
    @unpack
    def testReward(self, state, belief, minDis, groundTruthValue):
        # pass
        reward = Reward(self.sheepId, self.wolfId)
        rewardValue = reward(state, belief, minDis)
        truthValue = np.array_equal(rewardValue, groundTruthValue)
        self.assertTrue(truthValue)

    @data((([0, 0], [1, 1]), 1, 0),
          (([0, 0], [0, 1]), 1, 1))
    @unpack
    def testIsEnd(self, state, minDis, groundTruthValue):
        # pass
        isEnd = IsEnd(state, self.sheepId, self.wolfId)
        value = isEnd(minDis)
        truthValue = np.array_equal(value, groundTruthValue)
        self.assertTrue(truthValue)


if __name__ == '__main__':
    unittest.main()
