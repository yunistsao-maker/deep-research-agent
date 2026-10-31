#!/usr/bin/env bash
# Build the arm64 agent image and push it to ECR.
#   ./infra/build_push.sh us-west-2            -> prints the image URI to pass to setup.py
set -euo pipefail
REGION="${1:-us-west-2}"
REPO="deep-research-agent"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
TAG=$(date +%Y%m%d-%H%M%S)
docker buildx build --platform linux/arm64 -t "${URI}:${TAG}" -t "${URI}:latest" --push "$(dirname "$0")/../agent"
echo "${URI}:${TAG}"
