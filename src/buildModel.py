import tensorflow as tf
import numpy as np
from collections import deque
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BuildModel:
    def __init__(self, numStateSpace, numActionSpace, gamma, seed=128):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.gamma = gamma
        self.seed = seed

    def __call__(self, layersWidths, actionLayerWidths = [1], summaryPath="./tbdata"):
        print("Generating DQN Model with layers: {}".format(layersWidths))
        graph = tf.Graph()
        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            with tf.name_scope('inputs'):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="states")
                nextStates_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="nextStates")
                reward_ = tf.placeholder(tf.float32, [None, ], name="reward")
                act_ = tf.placeholder(tf.int32, [None, ], name="act")
                tf.add_to_collection("states", states_)
                tf.add_to_collection("nextStates", nextStates_)
                tf.add_to_collection("reward_", reward_)
                tf.add_to_collection("act_", act_)

            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)

            with tf.variable_scope("evalNet"):
                with tf.variable_scope("trainEvalHiddenLayers"):
                    activation_ = states_
                    for i in range(len(layersWidths)):
                        fcLayer = tf.layers.Dense(units=layersWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="fc{}".format(i+1), trainable = True)
                        activation_ = fcLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                        evalHiddenOutput_ = tf.identity(activation_, name="output")
                with tf.variable_scope("trainEvalOutputLayers"):
                    activation_ = evalHiddenOutput_
                    for i in range(len(actionLayerWidths)):
                        fcLayer = tf.layers.Dense(units=actionLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="fc{}".format(i+1))
                        activation_ = fcLayer(activation_)
                        tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    outputEvalFCLayer = tf.layers.Dense(units=self.numActionSpace, activation=None, kernel_initializer=initWeight,
                                                    bias_initializer=initBias, name="fc{}".format(len(actionLayerWidths) + 1))
                    evalNetOutput_ = outputEvalFCLayer(activation_)
                    tf.add_to_collections(["weights", f"weight/{outputEvalFCLayer.kernel.name}"], outputEvalFCLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{outputEvalFCLayer.bias.name}"], outputEvalFCLayer.bias)
                    tf.add_to_collections("evalNetOutput", evalNetOutput_)

            with tf.variable_scope("targetNet"):
                with tf.variable_scope("trainTargetHiddenLayers"):
                    activation_ = nextStates_
                    for i in range(len(layersWidths)):
                        fcLayer = tf.layers.Dense(units=layersWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="fc{}".format(i+1), trainable = True)
                        activation_ = fcLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                        targetHiddenOutput_ = tf.identity(activation_, name="output")
                with tf.variable_scope("trainTargetOutputLayers"):
                    activation_ = targetHiddenOutput_
                    for i in range(len(actionLayerWidths)):
                        fcLayer = tf.layers.Dense(units=actionLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="fc{}".format(i+1))
                        activation_ = fcLayer(activation_)
                        tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    outputTargetFCLayer = tf.layers.Dense(units=self.numActionSpace, activation=None, kernel_initializer=initWeight,
                                                    bias_initializer=initBias, name="fc{}".format(len(actionLayerWidths) + 1))
                    targetNetOutput_ = outputTargetFCLayer(activation_)
                    tf.add_to_collections(["weights", f"weight/{outputTargetFCLayer.kernel.name}"], outputTargetFCLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{outputTargetFCLayer.bias.name}"], outputTargetFCLayer.bias)
                    tf.add_to_collections("tarNetOutput", targetNetOutput_)

            with tf.variable_scope("loss"):
                with tf.variable_scope("QTarget"):
                    QTarget_ = reward_ + self.gamma * tf.reduce_max(targetNetOutput_, axis=1, name='QTarget')
                    gradQTarget_ = tf.stop_gradient(QTarget_)
                    tf.add_to_collections("QTarget", QTarget_)
                    tf.add_to_collections("gradQTarget", gradQTarget_)
                with tf.variable_scope("QEval"):
                    actIndices = tf.stack([tf.range(tf.shape(act_)[0], dtype=tf.int32), act_], axis=1)
                    QEvalwrtAct_ = tf.gather_nd(params=evalNetOutput_, indices=actIndices)
                    tf.add_to_collections("QEvalwrtAct", QEvalwrtAct_)

                loss_ = tf.reduce_mean(tf.squared_difference(gradQTarget_, QEvalwrtAct_, name='TD_error'))
                tf.add_to_collection("loss", loss_)

            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                evalParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evalNet')
                targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetNet')
                tf.add_to_collection("trainParams_", evalParams_)
                tf.add_to_collection("targetParams_", targetParams_)
                tf.add_to_collection("learningRate_", learningRate_)

            with tf.name_scope("train"):
                trainOpt_ = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer').minimize(loss_, var_list=evalParams_)
                tf.add_to_collection("trainOp", trainOpt_)

            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)
            if summaryPath is not None:
                trainWriter = tf.summary.FileWriter(summaryPath + "/train", graph=tf.get_default_graph())
                testWriter = tf.summary.FileWriter(summaryPath + "/test", graph=tf.get_default_graph())
                tf.add_to_collection("writers", trainWriter)
                tf.add_to_collection("writers", testWriter)
            saver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", saver)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        return model


class TrainDQN:

    def __init__(self, batchSize, updateFrequency, learningRate, reporter):
        self.batchSize = batchSize
        self.updateFrequency = updateFrequency
        self.learningRate = learningRate
        self.reporter = reporter
        self.step = 0

    def __call__(self, model, miniBatch):

        # if len(miniBatch) < self.batchSize:
        #     return
        print("ENTER TRAIN")
        self.step += 1
        graph = model.graph
        states_ = graph.get_collection_ref("states")[0]
        nextStates_ = graph.get_collection_ref("nextStates")[0]
        reward_ = graph.get_collection_ref("reward")
        act_ = graph.get_collection_ref("act")
        learningRate_ = graph.get_collection_ref("learningRate")
        loss_ = graph.get_collection_ref("loss")[0]
        trainOp = graph.get_collection_ref("trainOp")[0]
        fullSummaryOp = graph.get_collection_ref('summaryOps')[0]
        trainWriter = graph.get_collection_ref('writers')[0]
        fetches = [loss_, trainOp, fullSummaryOp]

        # evalParams_ = tf.get_collection_ref("evalParams")[0]
        # targetParams_ = tf.get_collection_ref("targetParams")[0]
        evalParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evalNet')
        targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetNet')
        paramsHolder_ = [tf.assign(t, e) for t, e in zip(targetParams_, evalParams_)]

        if self.step % self.updateFrequency == 0:
            model.run(paramsHolder_)

        statesBatch, actBatch, nextStatesBatch, rewardBatch = miniBatch[0]
        # feedDict = {states_: statesBatch, act_: actBatch, nextStates_: nextStatesBatch, reward_: rewardBatch}
        feedDict = {states_: tuple(statesBatch), act_: tuple(actBatch), nextStates_: tuple(nextStatesBatch), reward_: tuple(rewardBatch)}
        lossDict, _, summary = model.run(fetches, feed_dict=feedDict)

        self.reporter(evalDict, trainWriter, summary)

        return model


def sampleAction(model, states):
    graph = model.graph
    QEval_ = graph.get_collection_ref('evalNetOutput')[0]
    states_ = graph.get_collection_ref("states")[0]
    QEval = model.run(QEval_, feed_dict={states_: [states]})
    # action = [0 for _ in range(len(QEval[0]))]
    # action[np.argmax(QEval)] = 1
    return np.argmax(QEval)

def memorize(replayBuffer, states, act, nextStates, reward):
    replayBuffer.append((states, act, nextStates, reward))
    return replayBuffer


if __name__ == '__main__':
    #test
    buildModel = BuildModel(2,2,0.8)
    trainDQN = TrainDQN(32, 1, 0.1, 0)
    minibatch = [[1,2], [0,1], [1,2], [1]]
    statesBatch, actBatch, nextStatesBatch, rewardBatch = minibatch
    model = buildModel([3, 1])
    trainDQN(model, minibatch)
    graph = model.graph
    output_ = graph.get_collection_ref('evalNetOutput')[0]
    states_ = graph.get_collection_ref("states")[0]
    output = model.run(output_, feed_dict={states_: [[1,2]]})
    print(output)
    print(np.argmax(output))
    print(statesBatch)
    print(sampleAction(model, statesBatch))














