from rabbitmq.rpc import RpcWorker
from rabbitmq.wq import WorkQueueWorker
from rabbitmq.msg import MessageWorker
import os
import uuid

ID_WORKER = str(uuid.uuid4())

def get_rpc_worker():
    return RpcWorker(
        ID_WORKER,
        os.environ.get('RABBITMQ').split(':')[0],
        os.environ.get('RABBITMQ').split(':')[1],
        os.environ.get('RABBITMQ_USERNAME'),
        os.environ.get('RABBITMQ_PASSWORD')
    )

def get_wq_worker():
    return WorkQueueWorker(
        ID_WORKER,
        os.environ.get('RABBITMQ').split(':')[0],
        os.environ.get('RABBITMQ').split(':')[1],
        os.environ.get('RABBITMQ_USERNAME'),
        os.environ.get('RABBITMQ_PASSWORD')
    )

def get_msg_worker():
    return MessageWorker(
        ID_WORKER,
        os.environ.get('RABBITMQ').split(':')[0],
        os.environ.get('RABBITMQ').split(':')[1],
        os.environ.get('RABBITMQ_USERNAME'),
        os.environ.get('RABBITMQ_PASSWORD')
    )

