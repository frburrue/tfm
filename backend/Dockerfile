FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT [ "/bin/sh", "entrypoint.sh"]
