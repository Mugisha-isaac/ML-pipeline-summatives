#!/bin/bash

# Docker Build and Run Script
# This script helps test the Docker build locally before deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="audio-talent-api"
PORT=${1:-8000}
DOCKERFILE_PATH="backend/Dockerfile"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Audio Talent Classification Docker Build${NC}"
echo -e "${YELLOW}========================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker found${NC}"

# Build the Docker image
echo -e "\n${YELLOW}Building Docker image...${NC}"
docker build \
    -t $APP_NAME:latest \
    -t $APP_NAME:$(date +%Y%m%d-%H%M%S) \
    -f $DOCKERFILE_PATH \
    backend/

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Run the Docker container
echo -e "\n${YELLOW}Starting Docker container...${NC}"
echo -e "${YELLOW}API will be available at: http://localhost:$PORT${NC}"
echo -e "${YELLOW}Health check endpoint: http://localhost:$PORT/health${NC}"
echo -e "${YELLOW}Docs endpoint: http://localhost:$PORT/docs${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}\n"

docker run \
    --rm \
    -p $PORT:8000 \
    -e PYTHONUNBUFFERED=1 \
    -e DISABLE_LOAD_TESTING=true \
    --name $APP_NAME \
    $APP_NAME:latest

echo -e "\n${GREEN}✓ Container stopped${NC}"
