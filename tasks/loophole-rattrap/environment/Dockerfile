# Agent container ("main"). Deliberately contains no simulator, puzzle or solver:
# the agent can only reach the environment through the MCP server.
FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*
# MCP client library (also used by the oracle's trajectory replayer) and the compute stack
# an agent may use to process images or plan (the run itself has no internet)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mcp==1.30.0 numpy==2.4.6 scipy==1.17.1 pillow==13.0.0 opencv-python-headless==5.0.0.93
WORKDIR /app
