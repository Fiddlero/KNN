import math as math
import operator
import pandas as pd

data = pd.read_csv('iris.data.test', header=None)
testing_data = data.values
data = pd.read_csv('iris.data.learning', header=None)
learning_data = data.values

def distance(temp1, temp2):
    length = math.sqrt(math.pow((temp1[0] - temp2[0]), 2) + math.pow((temp1[1] - temp2[1]), 2)
                       + math.pow((temp1[2] - temp2[2]), 2) + math.pow((temp1[3] - temp2[3]), 2))
    return length

def getNeighbors(ourData):
    data = []
    for temp in data:
        data.append((temp[0:4]))
    return data


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
