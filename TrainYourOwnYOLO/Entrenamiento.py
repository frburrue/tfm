import os
import mlflow
import mlflow.tensorflow

os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAIGY6XHCMG44GVO2A'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'

sql_string = 'postgresql://francisco:francisco@mlflow.c3uusfqnmup4.eu-west-3.rds.amazonaws.com/mlflow'
mlflow.set_tracking_uri(sql_string)

exp = mlflow.get_experiment_by_name("tfm-candidate-dic-augmented-local")
id_exp = exp.experiment_id

mlflow.end_run()
mlflow.tensorflow.autolog(every_n_iter=1)

import numpy as np
from psycopg2.extensions import register_adapter, AsIs

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)

def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)

def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

def addapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))

register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)
register_adapter(np.float32, addapt_numpy_float32)
register_adapter(np.int32, addapt_numpy_int32)
register_adapter(np.ndarray, addapt_numpy_array)

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    image_data = []
    box_data = []
    for b in range(batch_size):
        if i==0:
            np.random.shuffle(annotation_lines)
        image, box = get_random_data(annotation_lines[i], input_shape, random=False)
        image_data.append(image)
        box_data.append(box)
        i = (i+1) % n
    image_data = np.array(image_data)
    box_data = np.array(box_data)
    y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
    return [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

import os
import sys
import warnings

def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(),"2_Training/Train_YOLO.py")))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

src_path = os.path.join(get_parent_dir(0), "src")

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
from yolo3.model import (
    preprocess_true_boxes,
    yolo_body,
    tiny_yolo_body,
    yolo_loss,
)
from yolo3.utils import get_random_data
import tensorflow as tf

keras_path = os.path.join(src_path, "keras_yolo3")
Data_Folder = os.path.join(get_parent_dir(1), "Data")
Image_Folder = os.path.join(Data_Folder, "Source_Images", "Training_Images")
VoTT_Folder = os.path.join(Image_Folder, "vott-csv-export-augmented")
YOLO_filename = os.path.join(VoTT_Folder, "data_train.txt")

Model_Folder = os.path.join(Data_Folder, "Model_Weights")
YOLO_classname = os.path.join(Model_Folder, "data_classes.txt")

log_dir = Model_Folder
anchors_path = os.path.join(keras_path, "model_data", "yolo_anchors.txt")
weights_path = os.path.join(keras_path, "model_data", "yolo.h5")

FLAGS = None
MLFLOW_PROJECT_PARAMS = {
      'validation_split': 0.2,
      'epochs': 1,
      'batch_size': 4,
      'steps_per_epoch': -1,
      'validation_steps': -1,
      'augmentation': True,
      'normalize': True,
      'standarize': True,
      'learning_rate': 1e-4,
      'reduce_lr_factor': 0.1,
      'reduce_lr_patience': 3,
      'early_stop_patience': 5
}
MLFLOW_TAGS = {
    'dataset': VoTT_Folder
}

class TRAIN_FLAGS:
  warnings = False
  log_dir = log_dir
  classes_file=YOLO_classname
  weights_path = weights_path
  anchors_path = anchors_path
  annotation_file=YOLO_filename

FLAGS = TRAIN_FLAGS()

with open(FLAGS.annotation_file) as f:
  lines = f.readlines()

lines = [line.replace("/home/francisco/Documentos/tfm/TrainYourOwnYOLO",".") for line in lines]
if not MLFLOW_PROJECT_PARAMS['augmentation']:
    lines = [line for line in lines if 'aug' not in line.split(' ')[0]]

np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
val_split = MLFLOW_PROJECT_PARAMS['validation_split']
num_val = int(len(lines) * val_split)
num_train = len(lines) - num_val
MLFLOW_PROJECT_PARAMS['steps_per_epoch'] = (max(1, num_train // MLFLOW_PROJECT_PARAMS['batch_size']*4), max(1, num_train // MLFLOW_PROJECT_PARAMS['batch_size']))
MLFLOW_PROJECT_PARAMS['validation_steps'] = (max(1, num_val // MLFLOW_PROJECT_PARAMS['batch_size']*4), max(1, num_train // MLFLOW_PROJECT_PARAMS['batch_size']))

log_dir = FLAGS.log_dir
class_names = get_classes(FLAGS.classes_file)
num_classes = len(class_names)
anchors = get_anchors(FLAGS.anchors_path)
input_shape = (416, 416)

is_tiny_version = len(anchors) == 6  # default setting
if is_tiny_version and FLAGS.weights_path == weights_path:
  weights_path = os.path.join(os.path.dirname(FLAGS.weights_path), "yolo-tiny.h5")
if is_tiny_version and FLAGS.anchors_path == anchors_path:
  anchors_path = os.path.join(os.path.dirname(FLAGS.anchors_path), "yolo-tiny_anchors.txt")
if is_tiny_version:
  model = create_tiny_model(
      input_shape, anchors, num_classes, freeze_body=2, weights_path=weights_path
  )
else:
  model = create_model(
      input_shape, anchors, num_classes, freeze_body=2, weights_path=weights_path
  )  # make sure you know what you freeze

@tf.autograph.experimental.do_not_convert
def yolo_loss_fnc(y_true, y_pred):
  return y_pred

checkpoint = ModelCheckpoint(
    os.path.join(log_dir, "checkpoint.h5"),
    monitor="yolo_loss",
    save_weights_only=True,
    save_freq=5,
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=MLFLOW_PROJECT_PARAMS['reduce_lr_factor'], patience=MLFLOW_PROJECT_PARAMS['reduce_lr_patience'], verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=MLFLOW_PROJECT_PARAMS['early_stop_patience'], verbose=1)

while True:
    data_generator_wrapper(
          lines[:num_train], MLFLOW_PROJECT_PARAMS['batch_size']*4, input_shape, anchors, num_classes
    )
    print("Sample")