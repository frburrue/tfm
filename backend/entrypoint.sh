#!/bin/bash

gunicorn --access-logfile - --bind 0.0.0.0:${BACKEND_PORT} -w ${BACKEND_NUM_WORKERS} --worker-class sanic.worker.GunicornWorker --max-requests 100 app:app
