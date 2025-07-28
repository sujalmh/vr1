#!/bin/bash

SERVICE=$1        # loader or retrieval
STAGE=$2          # test or prod
PORT=$3           # 6000 or 7000

IMAGE_NAME="${SERVICE}:${STAGE}"
ENV_FILE=".env.${STAGE}"

echo "Building Docker image for $SERVICE in $STAGE..."

docker build -t $IMAGE_NAME ./$SERVICE

echo "Stopping and removing old container..."
docker rm -f ${SERVICE}-${STAGE}

echo "Starting new container on port $PORT..."
docker run -d --env-file $ENV_FILE -p $PORT:$PORT --name ${SERVICE}-${STAGE} $IMAGE_NAME