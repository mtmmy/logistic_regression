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

inputSize = 784
numSlasses = 10
numTrains = 5
learningRate = 0.00001
size = 100

trainLbls = trainings[0][:size * 6]
trainImgs = [image.flatten() for image in trainings[1][:size * 6]]
testLbls = tests[0][:size]
testImgs = [image.flatten() for image in tests[1][:size]]

trainingData = []
testData = []

for i in range(size * 6):
    newImage = [float(p) / 100 for p in trainImgs[i]]
    trainingData.append(ImageData(trainLbls[i], np.array(newImage)))

for i in range(size):
    newImage = [float(p) / 100 for p in testImgs[i]]
    testData.append(ImageData(testLbls[i], np.array(newImage)))

weightedVectors = [[0] * 784 for _ in range(9)]

preprocessingTime = time.time()
print("--- Preprocessing spent {} seconds ---".format(preprocessingTime - startTime))

def denominatorofP(x):
    exps = []
    for w in weightedVectors:
        exps.append(np.exp(np.dot(w, x)))
    return 1 + sum(exps)

def getPYoverX(x, y):
    if y == 9:
        return 1 / denominatorofP(x)
    return np.exp(np.dot(weightedVectors[y], x)) / denominatorofP(x)

def getPrediction(d):
    p = [0] * 10
    for i in range(9):
        p[i] = getPYoverX(d.image, i)
    p[9] = 1 - sum(p)
    return np.argmax(p)

def getSumInGraidentAscent(x, guessP):
    alphas = [(x * (y - guessP)) for y in range(10)]
    return sum(alphas) 
    
def gradientAscent(weighted, imageData):
    newVector = []
    predict = getPrediction(imageData)
    predictP = getPYoverX(imageData.image, predict)
    for w, x in zip(weighted, imageData.image):
        newVector.append(w + learningRate * getSumInGraidentAscent(x, predictP))
    return newVector

def trainW():
    iterStartTime = time.time()
    for td in trainingData:
        if td.label < 9:
            weightedVectors[td.label] = gradientAscent(weightedVectors[td.label], td)
    print("--- This iteration spent {} seconds ---".format(time.time() - iterStartTime))   

def test(n):
    testStartTime = time.time()
    correct = 0
    for td in testData:
        if td.label == getPrediction(td):
            correct += 1
    print("Iterate {} times: {}".format(n, correct / size))
    print("--- This testing spent {} seconds ---".format(time.time() - testStartTime))   

iterations = 3
for i in range(iterations):
    trainW()
    test(i)

print("--- Totally, {} iterations spent {} seconds ---".format(iterations, time.time() - startTime))
