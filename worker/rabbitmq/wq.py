import pika
import pickle


class WorkQueueWorker:
    def __init__(self, id, host, port, username, password):
        self.id = id
        self.connection = pika.BlockingConnection(parameters=pika.ConnectionParameters(host=host, port=port,
                                                                                       credentials=pika.PlainCredentials(
                                                                                           username, password)))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='task_queue', durable=True)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='task_queue', on_message_callback=self.callback)

    def run(self):
        self.channel.start_consuming()

    def callback(self, ch, method, properties, body):

        data = pickle.loads(body)
        label = data['label']
        image = data['data']

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def release(self):
        self.connection.close()

    def __del__(self):
        self.release()
