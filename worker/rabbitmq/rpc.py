print("Loading RPC module...")

import pika
import json
import mlflow
import boto3
import os
import pickle
import shutil
import pandas as pd
import numpy as np
from utils import detect_object
from yolo3.yolo import YOLO
from mongo.mongo import MongoWrapper
from datetime import datetime

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

def inference(model_path):

    anchors_path = "./model_data/yolo_anchors.txt"
    classes_path = "./model_data/data_classes.txt"
    save_img = True

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

    for i, img_path in enumerate(["./image/hand.png", "./image/cat.png", "./image/cat_hand.png"]):

        prediction, image = detect_object(
            yolo,
            img_path,
            save_img=save_img,
            save_img_path=".",
            postfix="_catface",
        )

        y_size, x_size, _ = np.array(image).shape
        for single_prediction in prediction:
            out_df = out_df.append(
                pd.DataFrame(
                    [
                        [
                            os.path.basename(img_path.rstrip("\n")),
                            img_path.rstrip("\n"),
                        ]
                        + single_prediction
                        + [x_size, y_size]
                    ],
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
                    ],
                )
            )

    print("Fin")

class RpcWorker:

    def __init__(self, id, host, port, username, password):
        self.id = id
        self.connection = pika.BlockingConnection(
            parameters=pika.ConnectionParameters(
                host=host,
                port=port,
                credentials=pika.PlainCredentials(username, password)
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

    def run(self):
        self.channel.start_consuming()

    def on_request(self, ch, method, props, body):

        data = pickle.loads(body)
        model_name = data['model']
        response = {}
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

        if model:
            # Inference

            inference(model)



            pass

        else:
            response = {'error': 'service unavailable'}

        response = {
            'payload': response,
            'worker_id': self.id
        }

        self.channel.basic_publish(
            exchange='',
            routing_key=props.reply_to,
            properties=pika.BasicProperties(
                correlation_id=props.correlation_id
            ),
            body=json.dumps(response)
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def release(self):
        self.connection.close()

    def __del__(self):
        self.release()
