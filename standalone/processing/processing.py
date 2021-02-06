import copy
import numpy as np
from collections import defaultdict

from mongo.mongo import MongoWrapper
from common.common import Timer


MONGO_CLIENT = MongoWrapper()
USERS_COLLECTION = 'users'
DATA_COLLECTION = 'nodered'
MQTT_COLLECTION = 'mqtt'
TWTITTER_COLLECTION = 'twitter'


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

    return distances[len(token1)][len(token2)]


def calcDictDistance(lines, word, numWords):

    dictWordDist = []

    for line in lines:
        wordDistance = levenshteinDistanceDP(word, line.strip())
        if wordDistance >= 10:
            wordDistance = 9
        dictWordDist.append((int(wordDistance), line.strip()))

    closestWords = []
    dictWordDist.sort(key=lambda x: x[0])

    for i in range(min(numWords, len(dictWordDist))):
        closestWords.append(list(dictWordDist[i]))

    return closestWords


def search_match(data):

    usernames = {user['user']: user['id'] for user in [copy.deepcopy(user) for user in MONGO_CLIENT.get_many(USERS_COLLECTION, {})]}

    neighborhood = []
    for text in data['TextDetections']:
        neighborhood.append(calcDictDistance(list(usernames.keys()), text['DetectedText'], 1)[0] + [text['DetectedText']])
    neighborhood.sort(key=lambda x: x[0])

    id_user = None

    for neighboor in neighborhood:
        if float(neighboor[0]) < (len(neighboor[1])*2)/5:
            id_user = {'id': usernames.get(neighboor[1], None), 'user': neighboor[1]}
            if id_user['id']:
                break
            else:
                id_user = None
    return id_user, neighborhood


def get_user_data(id_user):

    user_data = defaultdict(list)
    user_data['application'] = 0

    if id_user:
        messages = MONGO_CLIENT.get_many(DATA_COLLECTION, {"chatId": id_user['id']})
        for message in messages:
            if message.get('show'):
                user_data['publications'].append({k: message.get(k, None) for k in ('content', 'date')})
                user_data['application'] += 1

        messages = MONGO_CLIENT.get_many(TWTITTER_COLLECTION, {'tweetId': id_user['user'].replace(' ', '_')})
        for message in messages:
            if message.get('show', True):
                user_data['opinions'].append({k: message.get(k, None) for k in ('content', 'date')})
                user_data['application'] += 1

        messages = MONGO_CLIENT.get_many(MQTT_COLLECTION, {'mqttId': id_user['id']})
        for message in messages:
            if message.get('show', True):
                user_data['sensors'].append({k: message.get(k, None) for k in ('content', 'date')})
                user_data['application'] += 1

    return user_data


def process(data, **kwargs):

    timer = Timer()

    user, neighborhood = search_match(data)
    user_data = get_user_data(user)

    if isinstance(user, dict) and 'user' in user:
        user = user['user']
        if not user_data['application']:
            user_data['application'] = ["No hay mensajes disponibles..."]
        else:
            user_data['application'] = []
    else:
        user = None
        user_data = {
            'application': ["No hay mensajes disponibles..."],
            'publications': [],
            'opinions': [],
            'sensors': []
        }

    elapsed = round(timer.value(), 3)

    response = {
        'payload': {
            'user': user,
            'matches': neighborhood,
            'messages': user_data
        },
        'elapsed': elapsed
    }

    return response