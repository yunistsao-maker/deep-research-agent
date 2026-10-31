# AgentCore Runtime requires linux/arm64 images.
# Build:  docker buildx build --platform linux/arm64 -t <ecr>/deep-research-agent:latest --push .
FROM --platform=linux/arm64 python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
# Playwright only needs its client/driver here: the browser itself is the AgentCore-managed Chromium
# reached over CDP, so no `playwright install` (keeps the image well under the 1 GB limit).
RUN pip install -r requirements.txt && pip install aws-opentelemetry-distro

COPY deep_research/ deep_research/
COPY app.py .

RUN useradd -m -u 1000 agent
USER agent
EXPOSE 8080
# opentelemetry-instrument -> traces/metrics in CloudWatch GenAI Observability (AgentCore Observability)
CMD ["opentelemetry-instrument", "python", "app.py"]
