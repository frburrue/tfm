#!/bin/bash

mlflow server -h 0.0.0.0 -p 8990 --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root s3://mlflow-tfm
