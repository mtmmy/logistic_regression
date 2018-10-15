import load_dataset
import time
import numpy as np

startTime = time.time()

path = "../assignment2_data/"
trainings = load_dataset.read("training", path)
tests = load_dataset.read("testing", path)

class ImageData:
    def __init__(self, label, image):
        self.label = label
        self.image = image

learningRate = 0.0001
testSize = 10000
trainSize = 60000

trainingData = [ImageData(label, image.flatten().astype(float)) for label, image in zip(trainings[0][:trainSize], trainings[1][:trainSize])]
testData = [ImageData(label, image.flatten().astype(float)) for label, image in zip(tests[0][:trainSize], tests[1][:trainSize])]

weightedVectors = np.zeros((10, 784), dtype=float)

preprocessingTime = time.time()
print("--- Preprocessing spent {} seconds ---".format(preprocessingTime - startTime))

def getPYoverX(x, label):
    inner = [np.inner(w, x) for w in weightedVectors]
    inner = np.array(inner)
    normalizedE = np.exp(inner - max(inner))
    p = normalizedE / sum(normalizedE)
    return p[label]

def getPrediction(d):
    p = [getPYoverX(d.image, i) for i in range(10)]
    return np.argmax(p)

def trainIteration():
    global weightedVectors 
    sumW = np.zeros((10, 784), dtype=float)
    for td in trainingData:
        diffP = 1 - getPYoverX(td.image, td.label)
        sumW[td.label] += td.image * diffP
    weightedVectors += learningRate * sumW

def test(n):
    correct = 0
    for td in testData:
        if td.label == getPrediction(td):
            correct += 1
    accuracy = correct / testSize
    print("Iterate {} times: {}".format(n + 1, accuracy))
    return accuracy

iterations = 100
accuracyRate = []
accuracyRate10 = []
for i in range(iterations):
    trainIteration()
    accuracy = test(i)
    accuracyRate.append(accuracy)
    if i == 0:
        accuracyRate10.append(accuracy)
    if i % 10 == 9:
        accuracyRate10.append(accuracy)

print(accuracyRate)
print(accuracyRate10)
print("--- Totally, {} iterations spent {} seconds ---".format(iterations, time.time() - startTime))
