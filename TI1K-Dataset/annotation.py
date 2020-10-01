import os
from PIL import Image, ImageDraw

base_path = "/home/ubuntu/TrainYourOwnYOLO/Data/Source_Images/Training_Images/vott-csv-export/"

ls = os.listdir("./vott-csv-export")

w, h = 640, 480

with open('./annotation/label.txt') as fd:
    with open('./annotation/label_parsed.txt', 'w') as fd_parsed:
        for row in fd:
            name, xmin, ymin, xmax, ymax = row.split(' ')[0:5]
            shape = [(int(float(xmin) * w), int(float(ymin) * h)), (int(float(xmax) * w), int(float(ymax) * h))]
            fd_parsed.write(base_path+name + ' ' + ','.join([str(shape[0][0]), str(shape[0][1]), str(shape[1][0]), str(shape[1][1]), "1"]) +'\n')
            img = Image.open("./vott-csv-export"+"/"+name)
            img1 = ImageDraw.Draw(img)
            img1.rectangle(shape, outline="red")
            img.save(open('./trash/'+name, "wb"), 'png')

