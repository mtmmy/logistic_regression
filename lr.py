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

learningRate = 0.00001
testSize = 10000
trainSize = 60000

trainingData = [ImageData(label, image.flatten().astype(float)) for label, image in zip(trainings[0][:trainSize], trainings[1][:trainSize])]
testData = [ImageData(label, image.flatten().astype(float)) for label, image in zip(tests[0][:testSize], tests[1][:testSize])]

weightedVectors = np.zeros((9, 784), dtype=float)

preprocessingTime = time.time()
print("--- Preprocessing spent {} seconds ---".format(preprocessingTime - startTime))

def getPs(x):
    inner = [np.inner(w, x) for w in weightedVectors]
    inner.append(0)
    inner = np.array(inner)
    es = np.exp(inner)
    p = es / sum(es)
    return p

def getPrediction(d):
    return np.argmax(getPs(d.image))

def trainIteration2():
    global weightedVectors 
    sumW = np.zeros((9, 784), dtype=float)
    for td in trainingData:
        guessP = getPs(td.image)
        for l in range(9):
            if l == td.label:
                sumW[l] += td.image * (1 - guessP[l]) / len(trainingData)
            else:
                sumW[l] -= td.image * guessP[l] / len(trainingData)
        
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
for i in range(iterations):
    trainIteration2()
    accuracy = test(i)
    accuracyRate.append(accuracy)

print(accuracyRate)
print("--- Totally, {} iterations spent {} seconds ---".format(iterations, time.time() - startTime))
