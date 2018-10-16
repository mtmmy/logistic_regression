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
testSize = 100
trainSize = 600

trainingData = [ImageData(label, image.flatten().astype(float)) for label, image in zip(trainings[0][:trainSize], trainings[1][:trainSize])]
testData = [ImageData(label, image.flatten().astype(float)) for label, image in zip(tests[0][:testSize], tests[1][:testSize])]

# weightedVectors = np.zeros((10, 784), dtype=float)
weightedVectors = np.zeros((9, 784), dtype=float)

preprocessingTime = time.time()
print("--- Preprocessing spent {} seconds ---".format(preprocessingTime - startTime))

def getPfromSoftMax(x, label):
    if label == 9:
        return 1 / (1 + getSumSoftMax(x))
    return np.exp(np.inner(weightedVectors[label], x)) / (1 + getSumSoftMax(x))

def getSumSoftMax(x):
    exps = [np.exp(np.inner(w, x)) for w in weightedVectors]
    return sum(exps)

def getPredictuonFromSoftMax(d):
    p = [getPfromSoftMax(d.image, l) for l in range(10)]
    return np.argmax(p)

def getPs(x):
    inner = [np.inner(w, x) for w in weightedVectors]
    inner.append(0)
    inner = np.array(inner)
    es = np.exp(inner)
    p = es / sum(es)
    return p

# def getPYoverX(x, label):
#     inner = [np.inner(w, x) for w in weightedVectors]
#     inner = np.array(inner)
#     normalizedE = np.exp(inner - max(inner))
#     p = normalizedE / sum(normalizedE)
#     return p[label]

def getPrediction(d):
    # p = [getPYoverX(d.image, i) for i in range(10)]
    return np.argmax(getPs(d.image))

# def trainIteration():
#     global weightedVectors 
#     sumW = np.zeros((10, 784), dtype=float)
#     for td in trainingData:
#         diffP = 1 - getPYoverX(td.image, td.label)
#         sumW[td.label] += td.image * diffP
#     weightedVectors += learningRate * sumW

def trainIteration2():
    global weightedVectors 
    sumW = np.zeros((9, 784), dtype=float)
    for td in trainingData:        
        # guessP = [getPfromSoftMax(td.image, l) for l in range(10)]
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
        # if td.label == getPredictuonFromSoftMax(td):
            correct += 1
    accuracy = correct / testSize
    print("Iterate {} times: {}".format(n + 1, accuracy))
    return accuracy

iterations = 100
accuracyRate = []
accuracyRate10 = []
for i in range(iterations):
    trainIteration2()
    accuracy = test(i)
    accuracyRate.append(accuracy)
    if i == 0:
        accuracyRate10.append(accuracy)
    if i % 10 == 9:
        accuracyRate10.append(accuracy)

print(accuracyRate)
print(accuracyRate10)
print("--- Totally, {} iterations spent {} seconds ---".format(iterations, time.time() - startTime))
