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
    LOOPHOLE_MESH_CACHE=/opt/loophole/.mesh_cache \
    LOOPHOLE_OMPL_DIR=/opt/loophole/.ompl_alpha

RUN apt-get update \
 && apt-get install -y --no-install-recommends libosmesa6 libgl1 libegl1 libglib2.0-0 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/loophole
COPY pyproject.toml ./
COPY loophole ./loophole
RUN pip install -e ".[mesh]"

# Canonical OMPL.app Alpha meshes (Rice licence: non-commercial, no redistribution) are
# fetched here from a pinned commit with sha256 checks - never committed to the repo.
# Derived OBJs and the convex (face-wedge) decomposition are generated inside the image.
COPY scripts/fetch_alpha.sh ./scripts/fetch_alpha.sh
RUN scripts/fetch_alpha.sh "$LOOPHOLE_OMPL_DIR" \
 && python -c "from loophole.puzzles.ompl_alpha import prepare; prepare('1.5')" \
 && python -c "from loophole.sim.env import LoopholeEnv; LoopholeEnv('alpha-1.5')"

# Verifier entrypoint (used when this image runs as Harbor's separate verifier).
COPY verifier /tests
RUN chmod +x /tests/test.sh && mkdir -p /var/loophole/episode

EXPOSE 8000
CMD ["python", "-m", "loophole.mcp_server"]
