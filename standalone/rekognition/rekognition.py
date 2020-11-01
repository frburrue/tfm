import cv2
import base64
import tempfile
import pytesseract
import boto3
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from botocore.config import Config

from common.common import Timer
from .utils import RekognitionTextParser, show_rekognition_polygons


REKOGNITION_CLIENT = boto3.client('rekognition', config=Config(
    region_name='us-west-2',
))


def ocr_tesseract(img, preprocess=None):

    image = np.array(img)

    if preprocess == 'thres':
        gray = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
            1]

    elif preprocess == "blur":
        gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 3)

    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)

    return text + "\n\nFilter: (%s)" % preprocess


# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/rekognition.html#Rekognition.Client.detect_text
def ocr_aws(img):

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return REKOGNITION_CLIENT.detect_text(Image={'Bytes': buffered.getvalue()})


def get_iou(bb1, bb2):

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def preprocess(img, predictions):

    fp = tempfile.NamedTemporaryFile()
    fp.write(img)

    best_hand = {}
    best_hand_confidence = -1
    best_panels = []
    best_hand_panel_overlap = -1.0
    best_hand_ok = False
    selected_panel = -1

    img = Image.open(fp.name)
    img_bckp = Image.open(fp.name)

    for p in predictions:
        print("Confidence: %d Class: %d" % (p[-2][-1], p[-2][0]))
        if p[-2][-1] >= 25:
            shape = list(map(lambda x: tuple(x), p[:2]))
            img1 = ImageDraw.Draw(img)
            img1.rectangle(shape, outline="red")

            if p[-2][0] == 0 and p[-2][-1] > best_hand_confidence:
                best_hand_confidence = p[-2][-1]
                best_hand.update(**{'x1': shape[0][0], 'x2': shape[1][0], 'y1': shape[0][1], 'y2': shape[1][1]})
                best_hand_ok = True
            elif p[-2][0] == 1:
                best_panels.append({'x1': shape[0][0], 'x2': shape[1][0], 'y1': shape[0][1], 'y2': shape[1][1]})

    if best_hand_ok:
        for idx, panel in enumerate(best_panels):
            iou = get_iou(best_hand, panel)
            if best_hand_panel_overlap < iou:
                best_hand_panel_overlap = iou
                selected_panel = idx

    if selected_panel >= 0:
        panel = best_panels[selected_panel]
        new_img = img_bckp.crop((panel['x1'], panel['y1'], panel['x2'], panel['y2']))
    else:
        new_img = None

    fp.close()

    return new_img


def rekognition_request(img, predictions):

    timer = Timer()

    new_frame = preprocess(img, predictions)
    img_str = None
    text = None

    if new_frame:

        text = ocr_aws(new_frame)

        buffered = BytesIO()
        new_frame.save(buffered, format="PNG")
        
        polygons = [RekognitionTextParser(item).to_dict()['polygon'] for item in text['TextDetections']]
        new_frame = show_rekognition_polygons(buffered.getvalue(), polygons, 'yellow')
        
        buffered = BytesIO()
        new_frame.save(buffered, format="PNG")
        
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        buffered.close()

    elapsed = round(timer.value(), 3)

    response = {
        'payload': {
            'image': img_str,
            'text': text
        },
        'elapsed': elapsed
    }

    return response


def rekognition_response(response):

    return response['payload']['text']
