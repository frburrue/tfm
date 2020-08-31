import mlflow
import datetime
import os

# sql_string = 'postgresql://francisco:francisco@thewhitehonet.ddns.net:60220/mlflow-tfm'
sql_string = 'postgresql://francisco:francisco@mlflow.coxesqvoqcft.us-east-1.rds.amazonaws.com/mlflow-tfm'
mlflow.set_tracking_uri(sql_string)

# expname = "colab"
# mlflow.create_experiment(expname, artifact_location="./artifacts")
mlflow.create_experiment(str(datetime.datetime.now()), artifact_location='s3://mlflow-tfm')

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow.tensorflow

mlflow.tensorflow.autolog(every_n_iter=1)

# Fixed for our Cats & Dogs classes
NUM_CLASSES = len(os.listdir('./food_dataset/test'))

# Fixed for Cats & Dogs color images
CHANNELS = 3

IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 1
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1

model = Sequential()

model.add(ResNet50(weights='imagenet', include_top=False, pooling=RESNET50_POOLING_AVERAGE))
model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))
model.layers[0].trainable = False  # Say not to train first layer (ResNet) model as it is already trained

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)

cb_early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath='./best.hdf5', monitor='val_loss', save_best_only=False, mode='auto',
                                  verbose=1)

image_size = IMAGE_RESIZE

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
    './food_dataset/train',
    target_size=(image_size, image_size),
    batch_size=BATCH_SIZE_TRAINING,
    class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    './food_dataset/validation',
    target_size=(image_size, image_size),
    batch_size=BATCH_SIZE_VALIDATION,
    class_mode='categorical')

checker = {}
for k, v in train_generator.class_indices.items():
    checker[v] = k

fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=STEPS_PER_EPOCH_VALIDATION,
    callbacks=[cb_checkpointer, cb_early_stopper]
)