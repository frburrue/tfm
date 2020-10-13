import cv2
import numpy as np
import requests
import base64
from PIL import Image, ImageDraw
import tempfile

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

        fp = tempfile.NamedTemporaryFile()
        fp.write(base64.b64decode(img))

        img = Image.open(fp.name)
        for p in predictions:
            print("Confidence: %d Class: %d" % (p[-2][-1], p[-2][0]))
            if p[-2][-1] >= 25:
                shape = list(map(lambda x: tuple(x), p[:2]))
                img1 = ImageDraw.Draw(img)
                img1.rectangle(shape, outline="red")

        fp.close()

        print("Done!")

        return np.array(img)

    except Exception as e:
        print(str(e))

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
            new_frame = inference('this.png')

    cv2.destroyWindow("preview")
    cv2.destroyWindow("detection")

cam()
