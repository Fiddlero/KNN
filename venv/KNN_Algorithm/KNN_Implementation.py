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
