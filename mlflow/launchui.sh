#!/bin/bash

mlflow server -h 0.0.0.0 -p 61001 --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root s3://mlflow-tfm
