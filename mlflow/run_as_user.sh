#!/bin/bash

docker run --rm --gpus all -u $(id -u):$(id -g) -v $(pwd):/app -t -i --network docker_franburruezo -p 8990-8991:8990-8991 docker_mlflow-environment bash
