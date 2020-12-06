#!/bin/bash

docker run --rm --gpus all -v $(pwd):/app -t -i --network docker_franburruezo -p 60260:8990 -p 60261:8888 docker_mlflow-environment bash
