# %% md

# Dependencias

# %%

import argparse
import json
import pickle as pckl
import numpy as np
import os
import cv2
import pandas as pd
from PIL import Image
from scipy.special import expit
from yolo3.yolo import YOLO


# %% md

## Cargar modelo

# %%

def load_model(model_path, classes_path, anchors_path):
    yolo = YOLO(
        **{
            "model_path": model_path,
            "anchors_path": anchors_path,
            "classes_path": classes_path,
            "score": 0.5,
            "gpu_num": 1,
            "model_image_size": (416, 416),
        }
    )

    return yolo


# %% md

## Bounding boxes

# %%

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


# %% md

## Generador de lotes

# %%

class BatchGenerator():
    def __init__(self, instances, anchors, labels, batch_size=1, shuffle=True):
        self.instances = instances
        self.batch_size = batch_size
        self.labels = labels
        self.anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1]) for i in range(len(anchors) // 2)]

        if shuffle:
            np.random.shuffle(self.instances)

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def load_annotation(self, i):
        annots = []

        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.instances[i]['filename'])

    # %% md


## Detection

# %%

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def get_yolo_boxes(model, images, net_h, net_w, nms_thresh):
    batch_output, data = model.detect_image(Image.fromarray(images[0].astype('uint8')))
    boxes = []

    for bo in batch_output:
        b = [0] * 2
        b[bo[4]] = bo[5]
        box = bo[:4] + [bo[5]] + [b]
        boxes.append(BoundBox(box[0], box[1], box[2], box[3], box[4], box[5]))

    # image_h, image_w, _ = images[0].shape
    # correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    do_nms(boxes, nms_thresh)
    return [boxes]


# %%

def detection(model, generator, nms_thresh=0.5, net_h=416, net_w=416):
    # gather all detections and annotations
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = [generator.load_image(i)]

        # make the boxes and the labels
        pred_boxes = get_yolo_boxes(model, raw_image, net_h, net_w, nms_thresh)[0]

        score = np.array([box.get_score() for box in pred_boxes])
        pred_labels = np.array([box.label for box in pred_boxes])

        if len(pred_boxes) > 0:
            pred_boxes = np.array([[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in pred_boxes])
        else:
            pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
        score_sort = np.argsort(-score)
        pred_labels = pred_labels[score_sort]
        pred_boxes = pred_boxes[score_sort]

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = pred_boxes[pred_labels == label, :]

        annotations = generator.load_annotation(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

    return all_detections, all_annotations


# %% md

## Evaluation

# %%

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mrec, mpre


# %%

def evaluation(all_detections, all_annotations, generator, iou_threshold=0.5):
    average_precisions = []

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:  # Si no hay anotación de esa detección es un falso positivo
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0),
                                           annotations)  # IOU, tiene el consideración todas las anotaciones
                assigned_annotation = np.argmax(overlaps, axis=1)  # Se queda con la anotación que maximiza el IOU
                max_overlap = overlaps[0, assigned_annotation]  # Se queda con el valor del IOU se esta anotación

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:  # Comprueba si esa anotación no ha sido ya asignada a una detección (además de comprobar que el IOU supera un cierto umbral). Las detecciones están ordenadas por score descendente por lo que se quedaría primero la que tiene mayor score (aunque luego pueda tener menor IoU).
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(
                        assigned_annotation)  # Guarda la anotación para que no pueda volver a ser usada
                else:  # IOU por debajo del umbral o anotación ya utilizada
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

                    # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score (Esto lo hace para ser consistente con los vectores de anotación y detección)
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives (Esto es lo mismo que sumar unos y ceros de cada una de los vectores pero se hace así para computar el AP)
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision (Y el F1)
        recall = true_positives / num_annotations  # Es lo mismo que dividir entre TP + FN porque la suma de ambas tiene que ser el número de anotaciones (se detecten o no)
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        f1 = 2 * (precision * recall) / (precision + recall)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions.append({'label': generator.labels[label], 'AP': average_precision, 'recall': recall[-1],
                                   'precision': precision[-1], 'f1': f1[-1], 'support': num_annotations,
                                   'TP': true_positives[-1], 'FP': false_positives[-1]})

    return average_precisions


# %% md

# Evaluación

# %% md

## Carga de modelo y de datos de test

# %%

os.chdir('/home/ubuntu/tfm')
config_path = './utils/config.json'

with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())

instances = pckl.load(open(config['model']['dataset_folder'], 'rb'))
labels = config['model']['labels']
labels = sorted(labels)

valid_generator = BatchGenerator(
    instances=instances,
    anchors=config['model']['anchors'],
    labels=sorted(config['model']['labels']),
)

infer_model = load_model(config['train']['model_folder'], config['train']['classes_path'],
                         config['train']['anchors_path'])

# %% md

## Test

# %%

all_detections, all_annotations = detection(infer_model, valid_generator)

# %%

average_precisions = evaluation(all_detections, all_annotations, valid_generator)

# %% md

## Procesar salida

# %%

items = 0
precision = 0
for average_precision in average_precisions:
    items += 1
    precision += average_precision['AP']
print('mAP: {:.4f}'.format(precision / items))
print(pd.DataFrame(average_precisions))

# %%


