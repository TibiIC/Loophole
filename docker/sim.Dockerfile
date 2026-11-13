# Loophole simulation sidecar (MCP server) and verifier image.
# Build context: repository root.  docker build -f docker/sim.Dockerfile -t loophole-sim:local .
# Bookworm pinned: Mesa >= 25.1 (trixie+) dropped OSMesa.
FROM python:3.11-slim-bookworm

ARG MUJOCO_GL=osmesa
ENV MUJOCO_GL=${MUJOCO_GL} \
    PYOPENGL_PLATFORM=${MUJOCO_GL} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOOPHOLE_LOG_DIR=/var/loophole/episode \
    LOOPHOLE_MESH_CACHE=/opt/loophole/.mesh_cache

RUN apt-get update \
 && apt-get install -y --no-install-recommends libosmesa6 libgl1 libegl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/loophole
COPY pyproject.toml ./
COPY loophole ./loophole
RUN pip install ".[mesh]"

# Verifier entrypoint (used when this image runs as Harbor's separate verifier).
COPY verifier /tests
RUN chmod +x /tests/test.sh && mkdir -p /var/loophole/episode

EXPOSE 8000
CMD ["python", "-m", "loophole.mcp_server"]
