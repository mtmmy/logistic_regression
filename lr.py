import load_dataset
import time
import numpy as np

startTime = time.time()
path = "../assignment2_data/"
trainingData = load_dataset.read("training", path)
testData = load_dataset.read("testing", path)

inputSize = 784
numSlasses = 10
numTrains = 5
size = 100
learningRate = 0.001

trainLbls = trainingData[0][:size * 6]
trainImgs = [image.flatten() for image in trainingData[1][:size * 6]]
testLbls = testData[0][:size]
testImgs = [image.flatten() for image in testData[1][:size]]

weightedVectors = [[1] * 784 for _ in range(10)]

def lrFunction(z):
    return 1 / (1 + np.exp(-z))