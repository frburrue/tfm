import requests
import base64
from PIL import Image, ImageDraw
import tempfile

url = "http://ec2-15-188-9-114.eu-west-3.compute.amazonaws.com/rpc/Hands"

payload = {}
files = [
  ('file', open('/home/francisco/Descargas/8-Home-Remedies-To-Make-Your-Hands-Soft-624x702.jpg','rb'))
]
headers= {}

response = requests.request("POST", url, headers=headers, data = payload, files = files)

img = response.json()['response']['payload']['result'][0]['image']
predictions = response.json()['response']['payload']['result'][0]['predictions']
to_write = base64.b64decode(img)
fp = tempfile.NamedTemporaryFile()
fp.write(base64.b64decode(img))

img = Image.open(fp.name)
print(predictions)
for p in predictions:
    if p[-2][-1] > 75:
        shape = list(map(lambda x: tuple(x), p[:2]))
        img1 = ImageDraw.Draw(img)
        img1.rectangle(shape, outline="red")
img.show()

fp.close()
