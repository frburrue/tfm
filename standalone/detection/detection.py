import os
import pickle
import base64
import tempfile
import copy
import numpy as np
from yolo3.yolo import YOLO

from .utils import detect_object
from common.common import Timer, OutputFilter
from mlflow_handlers.mlflow_handlers import get_model


DETECTION_MODEL = None


def load_model(model_path):

    anchors_path = "./model_data/yolo_anchors.txt"
    classes_path = "./model_data/data_classes.txt"

    yolo = YOLO(
        **{
            "model_path": model_path,
            "anchors_path": anchors_path,
            "classes_path": classes_path,
            "score": 0.25,
            "gpu_num": 1,
            "model_image_size": (416, 416),
        }
    )

    return yolo


def update_model_detection():

    global DETECTION_MODEL

    model_path = get_model('Hands')
    if model_path:
        DETECTION_MODEL = load_model(model_path)


def inference(yolo, image_path, **kwargs):

    # anchors_path = "./model_data/yolo_anchors.txt"
    # classes_path = "./model_data/data_classes.txt"

    # labels to draw on images
    # class_file = open(classes_path, "r")
    # input_labels = [line.rstrip("\n") for line in class_file.readlines()]

    # anchors
    # anchors = get_anchors(anchors_path)

    temp_dir = tempfile.TemporaryDirectory()
    save_fig = bool(int(kwargs['flags'][OutputFilter.DETECTION.value]))

    results = []

    for i, img_path in enumerate([image_path]):

        prediction, image = detect_object(
            yolo,
            img_path,
            save_img=save_fig,
            save_img_path=temp_dir.name,
            postfix="",
        )

        if save_fig:
            img = open(os.path.join(temp_dir.name, os.path.basename(img_path)), 'rb').read()
        else:
            img = b""

        result = {'image': base64.encodebytes(img).decode("utf-8"), 'predictions': []}

        y_size, x_size, _ = np.array(image).shape

        # xmin, ymin, xmax, ymax, label, confidence
        for single_prediction in prediction:
            single_prediction[-1] *= 100
            single_prediction = list(map(lambda x: int(x), single_prediction))
            result['predictions'].append([
                (single_prediction[0], single_prediction[1]),
                (single_prediction[2], single_prediction[3]),
                (single_prediction[4], single_prediction[5]),
                (int(x_size), int(y_size)),
                 ])
        results.append(copy.deepcopy(result))

    temp_dir.cleanup()

    return results


def inference_request(data, **kwargs):

    try:
        timer = Timer()

        model = DETECTION_MODEL
        response = {}

        fp = tempfile.NamedTemporaryFile(suffix='.png')
        fp.write(data['data'])

        if model:
            response['result'] = inference(model, fp.name, **kwargs)
        else:
            response = {'error': 'service unavailable'}

    except Exception as e:
        response = {'error': str(e)}

    finally:
        fp.close()

        elapsed = round(timer.value(), 3)

        response = {
            'payload': response,
            'elapsed': elapsed
        }

        return response


def inference_response(response):

    try:
        img = response['payload']['result'][0]['image']
        predictions = response['payload']['result'][0]['predictions']

    except Exception as e:
        img = predictions = None

    finally:
        return img, predictions



