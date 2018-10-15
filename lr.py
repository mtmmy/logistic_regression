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
learningRate = 0.01
testSize = 10000
trainSize = 60000

trainingData = []
testData = []

for label, image in zip(trainings[0][:trainSize], trainings[1][:trainSize]):
    newImage = [float(p) for p in image.flatten()]
    trainingData.append(ImageData(label, np.array(newImage)))
    # trainingData.append(ImageData(label, image.flatten()))

for label, image in zip(tests[0][:testSize], tests[1][:testSize]):
    newImage = [float(p) for p in image.flatten()]
    testData.append(ImageData(label, np.array(newImage)))
    # testData.append(ImageData(label, image.flatten()))

weightedVectors = [[0] * 784 for _ in range(10)]

preprocessingTime = time.time()
print("--- Preprocessing spent {} seconds ---".format(preprocessingTime - startTime))

def getPYoverX(x, label):
    inner = [np.inner(w, x) for w in weightedVectors]
    # inner.append(0)
    inner = np.array(inner)

    b = max(inner)
    y = np.exp(inner - b)
    p = y / sum(y)
    return p[label]

def getPrediction(d):
    p = [getPYoverX(d.image, i) for i in range(10)]
    return np.argmax(p)

# def getSumInGraidentAscent(x, guess, guessP):
#     alphas = [(x * (y - guessP)) for y in range(10)]
#     return sum(alphas) 
    
# def gradientAscent(weighted, imageData):
#     newVector = []
#     predict = getPrediction(imageData)
#     predictP = getPYoverX(imageData.image, predict)
#     for w, x in zip(weighted, imageData.image):
#         newVector.append(w + learningRate * getSumInGraidentAscent(x, predict, predictP))
#     return newVector

# def trainW():
#     iterStartTime = time.time()
#     for td in trainingData:
#         if td.label < 9:
#             weightedVectors[td.label] = gradientAscent(weightedVectors[td.label], td)
#     print("--- This iteration spent {} seconds ---".format(time.time() - iterStartTime))   

def trainW2():
    global weightedVectors
    iterStartTime = time.time()
    sumW = [[0] * 784 for _ in range(10)]
    for td in trainingData:
        # if td.label < 9:
        diffP = 1 - getPYoverX(td.image, td.label)
        for i in range(784):
            sumW[td.label][i] += td.image[i] * diffP
    
    for i in range(10):
        for j in range(784):
            weightedVectors[i][j] += sumW[i][j]
    # print("--- This iteration spent {} seconds ---".format(time.time() - iterStartTime))

def test(n):
    # print(sum(weightedVectors[0]))
    testStartTime = time.time()
    correct = 0
    for td in testData:
        if td.label == getPrediction(td):
            # print(td.label)
            correct += 1
    print("Iterate {} times: {}".format(n + 1, correct / testSize))
    # print("--- This testing spent {} seconds ---".format(time.time() - testStartTime))   

iterations = 100
for i in range(iterations):
    trainW2()
    test(i)

print("--- Totally, {} iterations spent {} seconds ---".format(iterations, time.time() - startTime))
