FROM python:3.7-slim

RUN apt update && apt install -y \
	build-essential \
	libpq-dev \
	libsm6 \
	libxext6 \
	libxrender-dev \
	libglib2.0-0 \
	git

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["/bin/sh", "docker_entrypoint.sh"]