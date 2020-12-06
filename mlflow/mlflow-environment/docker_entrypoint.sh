#!/bin/bash

tail -f /dev/null

mlflow server -h 0.0.0.0 -p 8990 --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root $MLFLOW_ARTIFACTS_PATH