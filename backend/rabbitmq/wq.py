import pika


class WorkQueueClient:

    def __init__(self, host, port, username, password):
        self.connection = pika.BlockingConnection(
            parameters=pika.ConnectionParameters(
                host=host,
                port=port,
                credentials=pika.PlainCredentials(username, password)
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='task_queue', durable=True)

    def enqueue(self, payload):
        self.channel.basic_publish(
            exchange='',
            routing_key='task_queue',
            body=payload,
            properties=pika.BasicProperties(
                delivery_mode=2,
            ))

    def release(self):
        self.connection.close()

    def __del__(self):
        self.release()
