#!/bin/bash
# EXPOSE 8686

set -e  # Exit immediately on any command failure
set -o pipefail

SERVICE=$1        # loader or retrieval
STAGE=$2          # test or prod
PORT=$3           # 6000 or 7000
ENTRYPOINT=$4  # Optional, not used in this script

IMAGE_NAME="${SERVICE}:${STAGE}"
CONTAINER_NAME="${SERVICE}-${STAGE}"
ENV_FILE=".env.${STAGE}"
SERVICE_DIR="./$SERVICE"

if [ ! -d "$SERVICE_DIR" ]; then
  echo "❌ Error: Directory $SERVICE_DIR does not exist."
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "❌ Error: Env file $ENV_FILE not found."
  exit 1
fi

echo "🔧 Building Docker image: $IMAGE_NAME"
docker build --no-cache -t "$IMAGE_NAME" "$SERVICE_DIR"

echo "🧹 Stopping and removing old container if it exists..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "🚀 Starting new container: $CONTAINER_NAME on port $PORT"
docker run -d \
  --env-file "$ENV_FILE" \
  -p "$PORT:$PORT" \
  --name "$CONTAINER_NAME" \
  "$IMAGE_NAME" \
  uvicorn "$ENTRYPOINT":app --host 0.0.0.0 --port "$PORT"

echo "✅ Deployment of $SERVICE in $STAGE stage complete."