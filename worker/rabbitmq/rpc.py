import pika
import json
import mlflow
import os
import copy
from datetime import datetime

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
MLFLOW_CLIENT = mlflow.tracking.MlflowClient()

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

        model_name = body.decode('utf-8')
        response = {}
        model = None # Model

        for mv in MLFLOW_CLIENT.search_model_versions(f"name='{model_name}'"):
            mv = dict(mv)
            if mv['current_stage'] == 'Production':
                source = mv['source']  # Model path
                mv['last_updated_timestamp'] = str(datetime.fromtimestamp(int(mv['last_updated_timestamp'] / 1000)))
                print("Using model {name} v{version} ({current_stage}) updated at {last_updated_timestamp}".format(**mv))
                response = {k: v for k, v in mv.items() if v}
                break

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
