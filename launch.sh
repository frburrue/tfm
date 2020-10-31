#/bin/bash

CMD=${1:-default}

echo "Executing with ${CMD} option"

DOCKER_MOUNTPOINT_PATH="/mnt/docker"
POSTGRES_RESOURCES_PATH="${DOCKER_MOUNTPOINT_PATH}/postgres"
PGADMIN_RESOURCES_PATH="${DOCKER_MOUNTPOINT_PATH}/pgadmin"
MONGO_RESOURCES_PATH="${DOCKER_MOUNTPOINT_PATH}/mongo"
NODE_RED_RESOURCES_PATH="${DOCKER_MOUNTPOINT_PATH}/node-red"
REQUIRED_FOLDERS=($DOCKER_MOUNTPOINT_PATH $POSTGRES_RESOURCES_PATH $PGADMIN_RESOURCES_PATH $MONGO_RESOURCES_PATH $NODE_RED_RESOURCES_PATH)

ELASTICSEARCH_VOLUME=elasticsearch
REQUIRED_VOLUMES=($ELASTICSEARCH_VOLUME)

LOG_VOLUME=log
LOG_MOUNTPOINT=/mnt/log

case "$CMD" in

  "clean" )
  
  echo "Cleaning ${REQUIRED_FOLDERS[*]} if exists..."

  for folder in "${REQUIRED_FOLDERS[@]}";
  do
  if [ -d "${folder}" ]; then
      rm -r $folder
  fi
  done

  echo "Cleaning ALL containers if exists..."

  docker ps -a | grep -v "CONTAINER" | awk -F " " '{print $1}' | while read a; do docker stop $a && docker container rm $a; done

  echo "Cleaning ${REQUIRED_VOLUMES[*]} if exists..."

  for volume in "${REQUIRED_VOLUMES[@]}";
  do
      docker volume rm `pwd | awk -F "/" '{print $NF"_"}'`${volume}
  done

  docker volume rm `pwd | awk -F "/" '{print $NF"_"}'`${LOG_VOLUME}

  ;;

 * )

  ;;

esac

echo "Checking..."

for folder in "${REQUIRED_FOLDERS[@]}";
do
if [ ! -d "${folder}" ]; then
  mkdir -p $folder
  echo "Create ${folder}..."
else
  echo "Found ${folder}!"
fi
done

echo "Ready!"

ARCH=$(uname -m) \
LOG_VOLUME=${LOG_VOLUME} \
LOG_MOUNTPOINT=${LOG_MOUNTPOINT} \
NETWORK=franburruezo \
BACKEND_HOST=backend \
BACKEND_PORT=60210 \
BACKEND_NUM_WORKERS=1 \
WORKER_HOST=worker \
WORKER_PORT=60211 \
WORKER_NUM_WORKERS=1 \
STANDALONE_HOST=worker \
STANDALONE_PORT=60212 \
STANDALONE_NUM_WORKERS=1 \
POSTGRES_HOST=postgres \
POSTGRES_PORT=60220 \
POSTGRES_RESOURCES_PATH=${POSTGRES_RESOURCES_PATH} \
PGADMIN_HOST=pgadmin \
PGADMIN_PORT=61220 \
PGADMIN_RESOURCES_PATH=${PGADMIN_RESOURCES_PATH} \
MONGO_HOST=mongo \
MONGO_PORT=60222 \
MONGO_RESOURCES_PATH=${MONGO_RESOURCES_PATH} \
REDIS_HOST=redis \
REDIS_PORT=60223 \
NODE_RED_RESOURCES_PATH=${NODE_RED_RESOURCES_PATH} \
ELASTICSEARCH_HOST=elasticsearch \
ELASTICSEARCH_PORT=60230 \
ELASTICSEARCH_VOLUME=${ELASTICSEARCH_VOLUME} \
KIBANA_HOST=kibana \
KIBANA_PORT=60232 \
FILEBEAT_HOST=filebeat \
FILEBEAT_PORT=60231 \
GRAFANA_HOST=grafana \
GRAFANA_PORT=60233 \
RABBITMQ_HOST=rabbitmq \
RABBITMQ_PORT=60240 \
NODE_RED_HOST=node-red \
NODE_RED_PORT=60250 \
MLFLOW_HOST=mlflow \
MLFLOW_PORT_UI=60260 \
MLFLOW_PORT_BACKEND=60261 \
MLFLOW_PORT_NOTEBOOK=60262 \
docker-compose up --build -d standalone
