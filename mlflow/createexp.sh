#!/bin/bash

mkdir $(pwd)/artifacts

mlflow experiments create -n fba-mlflow-exp-local -l $(pwd)/artifacts

