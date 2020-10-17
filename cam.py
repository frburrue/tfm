import cv2
import numpy as np
import requests
import base64
from PIL import Image, ImageDraw
import tempfile
import pytesseract


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


def post_process(img, predictions):

    fp = tempfile.NamedTemporaryFile()
    fp.write(base64.b64decode(img))

    best_hand = {}
    best_hand_confidence = -1
    best_panels = []
    best_hand_panel_overlap = -1.0
    selected_panel = None

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
            elif p[-2][0] == 1:
                best_panels.append({'x1': shape[0][0], 'x2': shape[1][0], 'y1': shape[0][1], 'y2': shape[1][1]})

    for idx, panel in enumerate(best_panels):
        iou = get_iou(best_hand, panel)
        if best_hand_panel_overlap < iou:
            best_hand_panel_overlap = iou
            selected_panel = idx

    panel = best_panels[selected_panel]
    new_img = img_bckp.crop((panel['x1'], panel['y1'], panel['x2'], panel['y2']))

    fp.close()

    return np.array(new_img)


def ocr(img, preprocess='blur'):

    image = img

    if preprocess == 'thres':
        gray = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
            1]

    elif preprocess == "blur":
        gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 3)

    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)

    return text + "\n\nFilter: (%s)" % preprocess


def inference(image_path):
    try:
        url = "http://ec2-15-188-9-114.eu-west-3.compute.amazonaws.com:60211/rpc/Hands"

        payload = {}
        files = [
            ('file', open(image_path, 'rb'))
        ]
        headers = {}

        response = requests.request("POST", url, headers=headers, data=payload, files=files).json()

        print("Elapsed time: %f seconds" % response['elapsed'])
        img = response['response']['payload']['result'][0]['image']
        predictions = response['response']['payload']['result'][0]['predictions']

    except Exception as e:
        print(str(e))
        img = predictions = None

    finally:
        return img, predictions

def cam():

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        new_frame = frame
    else:
        rval = False

    while rval:

        cv2.imshow("preview", frame)
        cv2.imshow("detection", new_frame)
        rval, frame = vc.read()

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        elif key == 115:
            print("Key pressed!")
            cv2.imwrite('this.png', frame)
            img, predictions = inference('sample.jpg')
            if img and predictions:
                new_frame = post_process(img, predictions)
                print(ocr(new_frame))

    cv2.destroyWindow("preview")
    cv2.destroyWindow("detection")

cam()
