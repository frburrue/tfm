print("Loading RPC module...")

import tempfile
import pika
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

        return json.dumps(response).encode()


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

logging.getLogger('pika').propagate = False

health = HealthCheck(app, "/healthcheck")

def backend_status():
    return True, 'backend ok'

health.add_check(backend_status)


@app.route('/rpc/<call>', methods=['GET', 'POST'])
async def rabbitmq_rpc(request, call):
    timer = Timer()
    response = on_request(pickle.dumps({'model': call, 'data': request.files["file"][0].body}))
    elapsed = round(timer.value(), 3)
    logging.info(json.dumps({'response': json.loads(response.decode('utf-8')), 'elapsed': elapsed}))
    return sanic_json({"response": json.loads(response.decode('utf-8')), 'success': True, 'elapsed': elapsed}, 200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "60211"))
    app.run(host='0.0.0.0', port=port, debug=False)