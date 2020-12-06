from rabbitmq.rpc import RpcClient
from rabbitmq.wq import WorkQueueClient
from rabbitmq.msg import MessageClient
import os
import uuid
import time

ID_BACKEND = str(uuid.uuid4())


class Timer:
    def __init__(self):
        self.init_time = time.time()

    def value(self):
        return time.time() - self.init_time


def get_rpc_client():
    return RpcClient(
        os.environ.get('RABBITMQ').split(':')[0],
        os.environ.get('RABBITMQ').split(':')[1],
        os.environ.get('RABBITMQ_USERNAME'),
        os.environ.get('RABBITMQ_PASSWORD')
    )


def get_wq_client():
    return WorkQueueClient(
        os.environ.get('RABBITMQ').split(':')[0],
        os.environ.get('RABBITMQ').split(':')[1],
        os.environ.get('RABBITMQ_USERNAME'),
        os.environ.get('RABBITMQ_PASSWORD')
    )


def get_msg_client():
    return MessageClient(
        os.environ.get('RABBITMQ').split(':')[0],
        os.environ.get('RABBITMQ').split(':')[1],
        os.environ.get('RABBITMQ_USERNAME'),
        os.environ.get('RABBITMQ_PASSWORD')
    )
