import pickle as pckl
from pymongo import MongoClient
import copy
import numpy as np

mc = MongoClient("mongodb://localhost:60222", username="francisco", password="francisco")
users_collection = mc.admin['users']
data_collection = mc.admin['nodered']


def printDistances(distances, token1Length, token2Length):
    for t1 in range(token1Length + 1):
        for t2 in range(token2Length + 1):
            print(int(distances[t1][t2]), end=" ")
        print()

def levenshteinDistanceDP(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    #printDistances(distances, len(token1), len(token2))
    return distances[len(token1)][len(token2)]

def calcDictDistance(lines, word, numWords):
    """
    file = open('1-1000.txt', 'r')
    lines = file.readlines()
    file.close()
    """
    dictWordDist = []
    wordIdx = 0

    for line in lines:
        wordDistance = levenshteinDistanceDP(word, line.strip())
        if wordDistance >= 10:
            wordDistance = 9
        dictWordDist.append(str(int(wordDistance)) + "-" + line.strip())
        wordIdx = wordIdx + 1

    closestWords = []
    wordDetails = []
    currWordDist = 0
    dictWordDist.sort()
    #print(dictWordDist)
    for i in range(min(numWords, len(dictWordDist))):
        currWordDist = dictWordDist[i]
        wordDetails = currWordDist.split("-")
        closestWords.append(wordDetails)
    return closestWords

data = pckl.load(open('./response_rekognition.pckl', 'rb'))

users = [copy.deepcopy(user) for user in users_collection.find({})]
usernames = [user['user'] for user in users]


def search_coincidence(rekognition_response):

    neighborhood = []
    for text in rekognition_response['TextDetections']:
        neighborhood.append(calcDictDistance(usernames, text['DetectedText'], 1)[0])
    neighborhood.sort(key=lambda x: x[0])

    id_user = None
    for neighboor in neighborhood:
        id_user = users_collection.find_one({"user": neighboor[1]})
        if id_user:
            break

    return id_user


def get_user_data(id_user):

    user_data = []

    if id_user:
        messages = data_collection.find({"chatId": id_user['id']})
        for message in messages:
            if message['show']:
                user_data.append(message['content'])

    return user_data

id_user = search_coincidence(data)
for item in get_user_data(id_user):
    print(item)