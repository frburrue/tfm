#!/bin/bash

gunicorn --bind 0.0.0.0:8991 src.api.backend_rest:app
