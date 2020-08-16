import os
import mlflow
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
MLFLOW_CLIENT = mlflow.tracking.MlflowClient()


@app.route("/predict", methods=["POST"])
def predict():

    req = request.json
    model_name = req['model']

    model = None # Model
    for mv in MLFLOW_CLIENT.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        if mv['current_stage'] == 'Production':
            source = mv['source'] # Model path
            mv['last_updated_timestamp'] = datetime.fromtimestamp(int(mv['last_updated_timestamp'] / 1000))
            print("Using model {name} v{version} ({current_stage}) updated at {last_updated_timestamp}".format(**mv))
            return mv

    return jsonify(result="Not model found...")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8991, threaded=False)