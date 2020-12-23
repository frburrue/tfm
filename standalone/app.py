import json
import os
import uuid
import sys
import signal
import logging
from sanic import Sanic
from sanic.response import json as sanic_json
from sanic_healthcheck import HealthCheck
from sanic_cors import CORS
from datetime import datetime

from mlflow_handlers.mlflow_handlers import update_models
from common.common import Timer, OutputFilter
from detection.detection import update_model_detection, inference_request, inference_response
from rekognition.rekognition import rekognition_request, rekognition_response
from processing.processing import process


update_models()
update_model_detection()


def signal_handler(signal, action):
    os.write(2, 'Received SIGINT. Stop!\n'.encode())
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


app = Sanic(__name__)
CORS(app)

logging.basicConfig(level='INFO', filename="/mnt/log/worker.log", filemode='a',
                    format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
logger = logging.getLogger(__name__)

health = HealthCheck(app, "/healthcheck")

def worker_status():
    return True, 'worker ok'

health.add_check(worker_status)


@app.route('/load', methods=['POST'])
async def load(request):

    timer = Timer()

    global_response = update_status = update_models()
    update_model_detection()

    elapsed = round(timer.value(), 3)

    logging.info(json.dumps({'response': global_response, 'elapsed': elapsed}))
    return sanic_json({"response": global_response, 'success': True, 'elapsed': elapsed}, 200)


@app.route('/update', methods=['POST'])
async def update(request):

    timer = Timer()

    global_response = update_status = update_models()
    if update_status['Hands']:
        update_model_detection()

    elapsed = round(timer.value(), 3)

    logging.info(json.dumps({'response': global_response, 'elapsed': elapsed}))
    return sanic_json({"response": global_response, 'success': True, 'elapsed': elapsed}, 200)


@app.route('/detection', methods=['POST'])
async def detection(request):

    timer = Timer()
    init = datetime.now()
    global_response = {}

    if 'params' in request.form:
        control = json.loads(request.form['params'][0].replace('\\', ''))
    else:
        control = request.form

    if bool(int(control['flags'][OutputFilter.APP_IN.value])):
        global_response['id'] = control['id']
    else:
        global_response['id'] = str(uuid.uuid4())

    response_detection = inference_request({'model': 'Hands', 'data': request.files["file"][0].body}, **control)
    img, predictions = inference_response(response_detection)

    response_rekogntion = {}
    processing_response = {}

    if img or predictions:

        response_rekogntion = rekognition_request(request.files["file"][0].body, predictions, **control)
        data = rekognition_response(response_rekogntion)

        if data:

            processing_response = process(data)

    if not bool(int(control['flags'][OutputFilter.APP_OUT.value])):
        global_response['response_detection'] = {k: response_detection.get(k, None) for k in ('elapsed',)}
        global_response['response_rekognition'] = {k: response_rekogntion.get(k, None) for k in ('elapsed',)}
    else:
        global_response['response_detection'] = response_detection
        global_response['response_rekognition'] = response_rekogntion
    global_response['response_processing'] = processing_response

    end = datetime.now()
    elapsed = round(timer.value(), 3)

    logging.info(json.dumps({'response': global_response, 'elapsed': elapsed}))
    return sanic_json({"response": global_response, 'success': True, 'elapsed': elapsed, 'init': str(init), 'end': str(end)}, 200)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "60210"))
    app.run(host='0.0.0.0', port=port, debug=False)
