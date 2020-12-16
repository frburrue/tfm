import mlflow
import os
import shutil
import boto3
from datetime import datetime


S3_CLIENT = boto3.resource('s3')

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
MLFLOW_CLIENT = mlflow.tracking.MlflowClient()

REGISTERED_MODELS = ["Hands"]
CURRENT_MODEL = "Unknown"
MODELS = {}


def downlod_model(bucket_name, remoteDirectory_name):

    bucket = S3_CLIENT.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=remoteDirectory_name):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key)


def update_models():

    global CURRENT_MODEL

    update = {}

    for model_name in REGISTERED_MODELS:
        model = None
        update[model_name] = 0
        for mv in MLFLOW_CLIENT.search_model_versions(f"name='{model_name}'"):
            mv = dict(mv)
            if mv['current_stage'] == 'Production':
                mv['last_updated_timestamp'] = str(datetime.fromtimestamp(int(mv['last_updated_timestamp'] / 1000)))
                bucket = mv['source'].split('//')[1].split('/')[0]
                folder = mv['source'].split('//')[1].split('/')[1]
                if os.path.exists(os.path.join('./models', folder)):
                    print("Load existing model...")
                    update[model_name] = not (CURRENT_MODEL == model)
                    CURRENT_MODEL = model = os.path.join(os.path.join('./models', folder), "artifacts/model/data/model.h5")
                else:
                    print("Downloading model...")
                    downlod_model(bucket, folder)
                    update[model_name] = 1
                    CURRENT_MODEL = model = os.path.join(os.path.join('./models', folder), "artifacts/model/data/model.h5")
                    if os.path.exists('./models'):
                        shutil.rmtree('./models')
                    os.mkdir('./models')
                    shutil.move(os.path.join(os.getcwd(), folder), './models')
                print("Using model {name} v{version} ({current_stage}) updated at {last_updated_timestamp}".format(**mv))
                #response = {k: v for k, v in mv.items() if v}
                break
        if model:
            MODELS[model_name] = model

    return update


def get_model(model_name):

    return MODELS.get(model_name, None)
