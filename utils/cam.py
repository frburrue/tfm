import cv2
import requests
import base64
import tempfile
import pytesseract
from PIL import Image, ImageDraw
import boto3
from botocore.config import Config
from io import BytesIO
from pymongo import MongoClient
import copy
import numpy as np


rekognition_client = boto3.client('rekognition', config=Config(
    region_name='us-west-2',
))

mc = MongoClient("mongodb://localhost:60222", username="francisco", password="francisco")
users_collection = mc.admin['users']
data_collection = mc.admin['nodered']


USERNAMES = [user['user'] for user in [copy.deepcopy(user) for user in users_collection.find({})]]


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

def search_coincidence(rekognition_response):

    neighborhood = []
    for text in rekognition_response['TextDetections']:
        neighborhood.append(calcDictDistance(USERNAMES, text['DetectedText'], 1)[0])
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


def get_iou(bb1, bb2):

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def post_process(img, predictions):

    fp = tempfile.NamedTemporaryFile()
    fp.write(base64.b64decode(img))

    best_hand = {}
    best_hand_confidence = -1
    best_panels = []
    best_hand_panel_overlap = -1.0
    selected_panel = None

    img = Image.open(fp.name)
    img_bckp = Image.open(fp.name)

    for p in predictions:
        print("Confidence: %d Class: %d" % (p[-2][-1], p[-2][0]))
        if p[-2][-1] >= 25:
            shape = list(map(lambda x: tuple(x), p[:2]))
            img1 = ImageDraw.Draw(img)
            img1.rectangle(shape, outline="red")

            if p[-2][0] == 0 and p[-2][-1] > best_hand_confidence:
                best_hand_confidence = p[-2][-1]
                best_hand.update(**{'x1': shape[0][0], 'x2': shape[1][0], 'y1': shape[0][1], 'y2': shape[1][1]})
            elif p[-2][0] == 1:
                best_panels.append({'x1': shape[0][0], 'x2': shape[1][0], 'y1': shape[0][1], 'y2': shape[1][1]})

    for idx, panel in enumerate(best_panels):
        iou = get_iou(best_hand, panel)
        if best_hand_panel_overlap < iou:
            best_hand_panel_overlap = iou
            selected_panel = idx

    panel = best_panels[selected_panel]
    new_img = img_bckp.crop((panel['x1'], panel['y1'], panel['x2'], panel['y2']))

    fp.close()

    return new_img

def ocr(img, preprocess=None):

    image = np.array(img)

    if preprocess == 'thres':
        gray = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
            1]

    elif preprocess == "blur":
        gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 3)

    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)

    return text + "\n\nFilter: (%s)" % preprocess

# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/rekognition.html#Rekognition.Client.detect_text
def ocr_aws(img):

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return rekognition_client.detect_text(Image={'Bytes': buffered.getvalue()})

def inference(image_path):
    try:
        url = "http://ec2-15-188-9-114.eu-west-3.compute.amazonaws.com:60211/rpc/Hands"

        payload = {}
        files = [
            ('file', open(image_path, 'rb'))
        ]
        headers = {}

        response = requests.request("POST", url, headers=headers, data=payload, files=files).json()

        print("Elapsed time: %f seconds" % response['elapsed'])
        img = response['response']['payload']['result'][0]['image']
        predictions = response['response']['payload']['result'][0]['predictions']

    except Exception as e:
        print(str(e))
        img = predictions = None

    finally:
        return img, predictions

def cam():

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        new_frame = frame
    else:
        rval = False

    while rval:

        cv2.imshow("preview", frame)
        cv2.imshow("detection", new_frame)
        rval, frame = vc.read()

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        elif key == 115:
            print("Key pressed!")
            cv2.imwrite('this.png', frame)
            img, predictions = inference('sample.jpg')
            if img and predictions:
                new_frame = post_process(img, predictions)
                rekognition_response = ocr_aws(new_frame)
                id_user = search_coincidence(rekognition_response)
                for item in get_user_data(id_user):
                    print(item)
                new_frame = np.array(new_frame)
                # print(ocr(new_frame))

    cv2.destroyWindow("preview")
    cv2.destroyWindow("detection")

cam()
