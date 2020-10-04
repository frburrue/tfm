FROM tensorflow/tensorflow:2.2.0-gpu-jupyter

RUN apt install wget

WORKDIR /workspace

COPY resources .

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh

RUN chmod +x Miniconda3-latest-Linux-x86_64.sh && \
 ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
 /workspace/miniconda3/bin/conda init bash && \
 /workspace/miniconda3/bin/conda create --name mlflow python=3.7 && \
 /workspace/miniconda3/bin/conda install --name mlflow -c conda-forge/label/cf202003 mlflow && \
 /workspace/miniconda3/bin/conda install --name mlflow psycopg2
ENV PATH="/workspace/miniconda3/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y nano libsm6 libxext6 libxrender-dev
