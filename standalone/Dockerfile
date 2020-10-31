FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && \
    apt-get upgrade && \
    apt-get -y install --no-install-recommends \
    build-essential \
    libpq-dev \
	libsm6 \
	libxext6 \
	libxrender-dev \
	libglib2.0-0 \
	git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT [ "/bin/sh", "entrypoint.sh"]