from sanic import Sanic
from sanic.response import json as sanic_json
from sanic_healthcheck import HealthCheck
from sanic_cors import CORS
from resources.resources import (
    ID_BACKEND,
    get_rpc_client, get_wq_client, get_msg_client,
    Timer
)
import os
import json
import sys
import signal
import logging
import socket
import pickle


def signal_handler(signal, action):
    os.write(2, 'Received SIGINT. Stop!\n'.encode())
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

app = Sanic(__name__)
CORS(app)

logging.basicConfig(level='INFO', filename="/mnt/log/backend.log", filemode='a',
                    format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
logger = logging.getLogger(__name__)

logging.getLogger('pika').propagate = False

health = HealthCheck(app, "/healthcheck")

def backend_status():
    return True, 'backend ok'

health.add_check(backend_status)


@app.route('/rpc/<call>', methods=['GET', 'POST'])
async def rabbitmq_rpc(request, call):
    timer = Timer()
    rpc_client = get_rpc_client()
    response = await rpc_client.call(pickle.dumps({'model': call, 'data': request.files["file"][0].body}))
    elapsed = round(timer.value(), 3)
    logging.info(json.dumps({'response': json.loads(response.decode('utf-8')), 'elapsed': elapsed}))
    return sanic_json({"response": json.loads(response.decode('utf-8')), 'success': True}, 200)


@app.route('/wq/<enqueue>', methods=['GET', 'POST'])
async def rabbitmq_wq(request, enqueue):
    timer = Timer()
    wq_client = get_wq_client()
    try:
        wq_client.enqueue(pickle.dumps({'label': enqueue, 'data': request.files["file"][0].body}))
        response = "OK"
    except Exception as e:
        response = str(e)
    elapsed = round(timer.value(), 3)
    logging.info(json.dumps({'response': response, 'elapsed': elapsed}))
    return sanic_json({"response": response, 'success': True}, 200)


@app.route('/msg/<message>', methods=['GET'])
async def rabbitmq_msg(request, message):
    timer = Timer()
    msg_client = get_msg_client()
    try:
        msg_client.send(message)
        response = "OK"
    except Exception as e:
        response = str(e)
    elapsed = round(timer.value(), 3)
    logging.info(json.dumps({'response': response, 'elapsed': elapsed}))
    return sanic_json({"response": response, "from": "{0}_{1}".format(socket.gethostname(), ID_BACKEND)}, 200)


@app.route('/', methods=['GET', 'POST'])
async def home(request):
    logging.info("OK")
    return sanic_json({"response": "Ok", "from": "{0}_{1}".format(socket.gethostname(), ID_BACKEND), 'success': True}, 200)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "60210"))
    app.run(host='0.0.0.0', port=port, debug=False)
