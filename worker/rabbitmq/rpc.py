print("Loading RPC module...")

import pika
import json
import mlflow
import mlflow.keras
import os
import uuid
import pickle
import numpy as np
from mongo.mongo import MongoWrapper
from datetime import datetime
from keras.preprocessing import image


# dimensions of our images
img_width, img_height = 224, 224

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

print("Connecting to MLflow...")

MLFLOW_CLIENT = mlflow.tracking.MlflowClient()

print("Connecting to Mongo...")

MONGO_CLIENT = MongoWrapper()

print("Ready")

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
        
        try:
            data = pickle.loads(body)
            model_name = data['model']
            response = {}
            model = None # Model
            print("Let's go!!!")
            for mv in MLFLOW_CLIENT.search_model_versions(f"name='{model_name}'"):
                mv = dict(mv)
                if mv['current_stage'] == 'Production':
                    mv['last_updated_timestamp'] = str(datetime.fromtimestamp(int(mv['last_updated_timestamp'] / 1000)))
                    if os.path.exists(mv['last_updated_timestamp']):
                        print("Load existing model...")
                        model = mlflow.keras.load_model(mv['last_updated_timestamp'])
                    else:
                        print("Downloading model...")
                        model = mlflow.keras.load_model(mv['source'])
                        mlflow.keras.save_model(model, mv['last_updated_timestamp'])
                    print("Using model {name} v{version} ({current_stage}) updated at {last_updated_timestamp}".format(**mv))
                    response = {k: v for k, v in mv.items() if v}
                    break

            if model:
                idx = str(uuid.uuid4())
                open(str(idx), 'wb').write(data['data'])
                img = image.load_img(idx, target_size=(img_width, img_height))
                os.unlink(idx)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                response['prediction'] = int(np.argmax(model.predict_proba(x)))
                response['text'] = MONGO_CLIENT.get_one(model_name, response['prediction'])

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
        except Exception as e:
            print(str(e))

    def release(self):
        self.connection.close()

    def __del__(self):
        self.release()
