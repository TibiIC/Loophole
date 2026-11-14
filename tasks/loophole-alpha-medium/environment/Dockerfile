# Agent container ("main"). Deliberately contains no simulator, puzzle or solver:
# the agent can only reach the environment through the MCP server.
FROM python:3.11-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*
# MCP client library (also used by the oracle's trajectory replayer)
RUN pip install --no-cache-dir mcp==1.30.0
WORKDIR /app
