import pika


class MessageWorker:
    def __init__(self, id, host, port, username, password):
        self.id = id
        self.connection = pika.BlockingConnection(parameters=pika.ConnectionParameters(host=host, port=port,
                                                                                       credentials=pika.PlainCredentials(
                                                                                           username, password)))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='msg')
        self.channel.basic_consume(queue='msg', on_message_callback=self.callback, auto_ack=True)

    def run(self):
        self.channel.start_consuming()

    def callback(self, ch, method, properties, body):
        body = body.decode('utf-8')

    def release(self):
        self.connection.close()

    def __del__(self):
        self.release()