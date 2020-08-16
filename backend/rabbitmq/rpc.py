import pika
import uuid
import asyncio

class RpcClient:

    def __init__(self, host, port, username, password):

        self.connection = pika.BlockingConnection(
            parameters=pika.ConnectionParameters(
                host=host,
                port=port,
                credentials=pika.PlainCredentials(username, password)
            )
        )
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):

        if self.corr_id == props.correlation_id:
            self.response = body

    async def call(self, payload):

        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=payload
        )

        while self.response is None:
            self.connection.process_data_events()
            await asyncio.sleep(1)


        return self.response

    def release(self):

        self.connection.close()

    def __del__(self):

        self.release()
