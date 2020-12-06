#/bin/bash

CMD=${1:-default}

echo "Executing with ${CMD} option"

DOCKER_MOUNTPOINT_PATH="/mnt/docker"
POSTGRES_RESOURCES_PATH="${DOCKER_MOUNTPOINT_PATH}/postgres"
MONGO_RESOURCES_PATH="${DOCKER_MOUNTPOINT_PATH}/mongo"
NODE_RED_RESOURCES_PATH="${DOCKER_MOUNTPOINT_PATH}/node-red"
REQUIRED_FOLDERS=($DOCKER_MOUNTPOINT_PATH $POSTGRES_RESOURCES_PATH $MONGO_RESOURCES_PATH $NODE_RED_RESOURCES_PATH)

case "$CMD" in

  "clean" )

  echo "Cleaning ALL containers if exists..."

  docker ps -a | grep -v "CONTAINER" | awk -F " " '{print $1}' | while read a; do docker stop $a && docker container rm $a; done
  
  echo "Cleaning ${REQUIRED_FOLDERS[*]} if exists..."

  for folder in "${REQUIRED_FOLDERS[@]}";
  do
  if [ -d "${folder}" ]; then
      rm -r $folder
  fi
  done

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
NETWORK=franburruezo \
STANDALONE_HOST=standalone \
STANDALONE_PORT=60210 \
STANDALONE_NUM_WORKERS=1 \
POSTGRES_HOST=postgres \
POSTGRES_PORT=60220 \
POSTGRES_RESOURCES_PATH=${POSTGRES_RESOURCES_PATH} \
MONGO_HOST=mongo \
MONGO_PORT=60222 \
MONGO_RESOURCES_PATH=${MONGO_RESOURCES_PATH} \
NODE_RED_RESOURCES_PATH=${NODE_RED_RESOURCES_PATH} \
NODE_RED_HOST=node-red \
NODE_RED_PORT=60250 \
MLFLOW_HOST=mlflow \
MLFLOW_PORT_UI=60260 \
MLFLOW_PORT_NOTEBOOK=60261 \
docker-compose up --build -d pgadmin mlflow-environment
