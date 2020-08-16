import pika


class MessageClient:

    def __init__(self, host, port, username, password):
        self.connection = pika.BlockingConnection(
            parameters=pika.ConnectionParameters(
                host=host,
                port=port,
                credentials=pika.PlainCredentials(username, password)
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='msg')

    def send(self, payload):
        self.channel.basic_publish(exchange='', routing_key='msg', body=payload)

    def release(self):
        self.connection.close()

    def __del__(self):
        self.release()
