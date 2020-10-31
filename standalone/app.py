import json
import os
import pickle
import sys
import signal
import logging
from sanic import Sanic
from sanic.response import json as sanic_json
from sanic_healthcheck import HealthCheck
from sanic_cors import CORS

from mlflow_handlers.mlflow_handlers import update_models
from common.common import Timer
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


@app.route('/rpc/<call>', methods=['GET', 'POST'])
async def rabbitmq_rpc(request, call):

    timer = Timer()
    global_response = {}

    response_detection = inference_request(pickle.dumps({'model': call, 'data': request.files["file"][0].body}))
    img, predictions = inference_response(response_detection)

    response_rekogntion = None
    processing_response = None

    if img and predictions:

        response_rekogntion = rekognition_request(request.files["file"][0].body, predictions)
        data = rekognition_response(response_rekogntion)

        if data:

            processing_response = process(data)

    global_response['response_detection'] = response_detection
    global_response['response_rekognition'] = response_rekogntion
    global_response['response_processing'] = processing_response

    elapsed = round(timer.value(), 3)

    logging.info(json.dumps({'response': global_response, 'elapsed': elapsed}))
    return sanic_json({"response": global_response, 'success': True, 'elapsed': elapsed}, 200)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "60212"))
    app.run(host='0.0.0.0', port=port, debug=False)
