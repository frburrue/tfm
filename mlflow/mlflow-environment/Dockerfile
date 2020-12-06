FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential wget nano libpq-dev libsm6 libxext6 libxrender-dev libglib2.0-0 git

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    ln -s /miniconda3/bin/conda /usr/local/bin/conda && \
    conda init && \
    conda create --name mlflow python=3.7 && \
    echo ". /usr/local/bin/conda" >> ~/.bashrc && \
    echo "conda activate mlflow" >> ~/.bashrc && \
    ln -s /miniconda3/envs/mlflow/bin/python /usr/local/bin/python

COPY requirements.txt .
RUN /miniconda3/envs/mlflow/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/miniconda3/envs/mlflow/bin:${PATH}"

COPY . .

ENTRYPOINT ["/bin/sh", "docker_entrypoint.sh"]
