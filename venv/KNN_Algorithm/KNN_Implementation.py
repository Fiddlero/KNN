import math as math
import operator
import pandas as pd

data = pd.read_csv('iris.data.test', header=None)
testSet = data.values
data = pd.read_csv('iris.data.learning', header=None)
trainingSet = data.values


def distance(temp1, temp2):
    length = math.sqrt(math.pow((temp1[0] - temp2[0]), 2) + math.pow((temp1[1] - temp2[1]), 2)
                       + math.pow((temp1[2] - temp2[2]), 2) + math.pow((temp1[3] - temp2[3]), 2))
    return length


def getNeighbors(data):
    distance = []
    for temp in data:
        distance.append((temp[0:4]))
    return distance


def getResponse(neighbours):
    classVotes = {}
    for x in range(len(neighbours)):
        response = neighbours[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.__iter__(), key=operator.itemgetter(-1))
    return sortedVotes[0]


class KNN:
    def __init__(self, temp, learningData):
        self.temp = temp
        self.data = learningData

    def predict(self, testingData):
        labels = []
        for y in range(len(testingData)):
            distances = []
            for x in range(len(self.data)):
                point = self.data[x]
                dist = distance(point, testingData[y])
                distances.append((point, dist))
            distances.sort(key=operator.itemgetter(1))
            neighbors = []
            for i in range(self.temp):
                neighbors.append(distances[i][0])
            labels.append(getResponse(neighbors))
        return labels

    def score(self, testingData, neightbours):
        correct = 0
        for i in range(len(testingData)):
            data = testingData[i]
            if (data[4] == neightbours[i]):
                correct += 1
        return (correct)


KNN = KNN(5, trainingSet)
testing_data_without_labels = getNeighbors(testSet)
labels_to_check = KNN.predict(testing_data_without_labels)
print("Score: ", KNN.score(testSet, labels_to_check))
print("Total rows: ", len(testSet))
