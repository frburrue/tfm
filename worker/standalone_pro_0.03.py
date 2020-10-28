import tempfile
import json
import copy
import mlflow
import boto3
import base64
import os
import pickle
import shutil
import pandas as pd
import numpy as np
import time
from utils import detect_object
from yolo3.yolo import YOLO
from mongo.mongo import MongoWrapper
from datetime import datetime

from sanic import Sanic
from sanic.response import json as sanic_json
from sanic_healthcheck import HealthCheck
from sanic_cors import CORS
import sys
import signal
import logging

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


class Timer:
    def __init__(self):
        self.init_time = time.time()

    def value(self):
        return time.time() - self.init_time


mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

print("Connecting to MLflow...")

MLFLOW_CLIENT = mlflow.tracking.MlflowClient()

print("Connecting to Mongo...")

MONGO_CLIENT = MongoWrapper()

print("Ready")

rekognition_client = boto3.client('rekognition', config=Config(
    region_name='us-west-2',
))

mc = MongoClient("mongodb://thewhitehonet.ddns.net:60222", username="francisco", password="francisco")
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
    best_hand_ok = False
    selected_panel = -1

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
                best_hand_ok = True
            elif p[-2][0] == 1:
                best_panels.append({'x1': shape[0][0], 'x2': shape[1][0], 'y1': shape[0][1], 'y2': shape[1][1]})

    if best_hand_ok:
        for idx, panel in enumerate(best_panels):
            iou = get_iou(best_hand, panel)
            if best_hand_panel_overlap < iou:
                best_hand_panel_overlap = iou
                selected_panel = idx

    if selected_panel >= 0:
        panel = best_panels[selected_panel]
        new_img = img_bckp.crop((panel['x1'], panel['y1'], panel['x2'], panel['y2']))
    else:
        new_img = None

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

def inference_response(response):
    try:
        img = response['payload']['result'][0]['image']
        predictions = response['payload']['result'][0]['predictions']

    except Exception as e:
        print(str(e))
        img = predictions = None

    finally:
        return img, predictions

def downloadDirectoryFroms3(bucketName, remoteDirectoryName):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucketName)
    for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key)

def load_model(model_path):

    anchors_path = "./model_data/yolo_anchors.txt"
    classes_path = "./model_data/data_classes.txt"

    yolo = YOLO(
        **{
            "model_path": model_path,
            "anchors_path": anchors_path,
            "classes_path": classes_path,
            "score": 0.25,
            "gpu_num": 1,
            "model_image_size": (416, 416),
        }
    )

    return yolo

def inference(yolo, image_path):

    anchors_path = "./model_data/yolo_anchors.txt"
    classes_path = "./model_data/data_classes.txt"
    save_img = True

    # Make a dataframe for the prediction outputs
    out_df = pd.DataFrame(
        columns=[
            "image",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
            "x_size",
            "y_size",
        ]
    )

    # labels to draw on images
    class_file = open(classes_path, "r")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]

    # anchors
    # anchors = get_anchors(anchors_path)
    images = []

    results = []

    for i, img_path in enumerate([image_path]):

        prediction, image = detect_object(
            yolo,
            img_path,
            save_img=save_img,
            save_img_path="./processed_images",
            postfix="",
        )

        result = {'image': base64.encodebytes(open(os.path.join('./processed_images', img_path), 'rb').read()).decode("utf-8")}
        result['predictions'] = []

        y_size, x_size, _ = np.array(image).shape
        # xmin, ymin, xmax, ymax, label, confidence
        for single_prediction in prediction:
            single_prediction[-1] *= 100
            single_prediction = list(map(lambda x: int(x), single_prediction))
            result['predictions'].append([
                (single_prediction[0], single_prediction[1]),
                (single_prediction[2], single_prediction[3]),
                (single_prediction[4], single_prediction[5]),
                (int(x_size), int(y_size)),
                 ])
        results.append(copy.deepcopy(result))

    return results

registered_models = ["Hands"]
handlers = {"Hands": load_model}
models = {}

for model_name in registered_models:
    model = None
    for mv in MLFLOW_CLIENT.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        if mv['version'] == 7:
            mv['last_updated_timestamp'] = str(datetime.fromtimestamp(int(mv['last_updated_timestamp'] / 1000)))
            bucket = mv['source'].split('//')[1].split('/')[0]
            folder = mv['source'].split('//')[1].split('/')[1]
            if os.path.exists(os.path.join('./models', folder)):
                print("Load existing model...")
                model = os.path.join(os.path.join('./models', folder), "artifacts/model/data/model.h5")
            else:
                print("Downloading model...")
                downloadDirectoryFroms3(bucket, folder)
                model = os.path.join(os.path.join('./models', folder), "artifacts/model/data/model.h5")
                if os.path.exists('./models'):
                    shutil.rmtree('./models')
                os.mkdir('./models')
                shutil.move(os.path.join(os.getcwd(), folder), './models')
            print("Using model {name} v{version} ({current_stage}) updated at {last_updated_timestamp}".format(**mv))
            response = {k: v for k, v in mv.items() if v}
            break
    models[model_name] = handlers[model_name](model)


def on_request(body):

    try:

        timer = Timer()

        data = pickle.loads(body)
        model = models[data['model']]
        response = {}

        fp = tempfile.NamedTemporaryFile(suffix='.png')
        fp.write(data['data'])

        if model:
            response['result'] = inference(model, fp.name)
        else:
            response = {'error': 'service unavailable'}

    except Exception as e:
        response = {'error': str(e)}

    finally:
        fp.close()

        elapsed = round(timer.value(), 3)

        response = {
            'payload': response,
            'worker_id': "standalone",
            'elapsed': elapsed
        }

        return response


def signal_handler(signal, action):
    os.write(2, 'Received SIGINT. Stop!\n'.encode())
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


app = Sanic(__name__)
CORS(app)

logging.basicConfig(level='INFO', filename="/mnt/log/worker.log", filemode='a',
                    format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
logger = logging.getLogger(__name__)

health = HealthCheck(app, "/healthcheck")

def worker_status():
    return True, 'worker ok'

health.add_check(worker_status)


@app.route('/rpc/<call>', methods=['GET', 'POST'])
async def rabbitmq_rpc(request, call):
    timer = Timer()
    response = on_request(pickle.dumps({'model': call, 'data': request.files["file"][0].body}))

    import base64
    from io import BytesIO

    messages = []

    img, predictions = inference_response(response)

    if img and predictions:

        new_frame = post_process(img, predictions)

        if new_frame:

            buffered = BytesIO()
            new_frame.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            buffered.close()

            rekognition_response = ocr_aws(new_frame)
            user = search_coincidence(rekognition_response)
            for item in get_user_data(user):
                messages.append(item)

            if isinstance(user, dict) and 'user' in user:
                user = user['user']
                if not len(messages):
                    messages.apppend("No hay mensajes disponibles...")

            else:
                user = None
                messages.apppend("No se ha encontrado usuario...")

        else:
            img_str = None
            user = None
            messages.append("Nada detectado...")

    response['payload'] = {
        'detection': img_str,
        'user': user,
        'messages': messages
    }

    elapsed = round(timer.value(), 3)
    logging.info(json.dumps({'response': response, 'elapsed': elapsed}))
    return sanic_json({"response": response, 'success': True, 'elapsed': elapsed}, 200)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "60211"))
    app.run(host='0.0.0.0', port=port, debug=False)
