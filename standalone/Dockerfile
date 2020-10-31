FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    libpq-dev \
	libsm6 \
	libxext6 \
	libxrender-dev \
	libglib2.0-0 \
	libgl1-mesa-glx \
	git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/keras_yolo3

RUN python setup.py build && \
    mkdir ./build/lib/yolo3/font && \
    cp resources/* ./build/lib/yolo3/font/ && \
    python setup.py install

RUN mkdir /mnt/log

WORKDIR /app

ENTRYPOINT [ "/bin/sh", "entrypoint.sh"]