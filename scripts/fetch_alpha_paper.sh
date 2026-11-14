#!/bin/bash
# Download the Alpha 1.0 puzzle used in Zhang, Belfer, Kry & Vouga, "C-Space Tunnel
# Discovery for Puzzle Path Planning", ACM TOG 39(4), SIGGRAPH 2020:
# meshes, OMPL problem file and the authors' solution path (OMPL format), from their
# puzzle repository at a pinned commit, with sha256 checks. Byte-identical to the
# paper's supplementary material (supp-material/puzzles/alpha/).
#
# The geometry derives from the OMPL.app Alpha puzzle (Rice University licence:
# non-commercial, no redistribution) and the repository states no licence of its own,
# so these files are NEVER committed: fetched here (`make fetch-alpha-paper`) or at
# Docker build time. Usage: scripts/fetch_alpha_paper.sh [DEST] (default <repo>/.alpha_paper)
set -euo pipefail
COMMIT=cd05c5da442a526b13b0abb1c072e6e1c061aede   # gitlab.com/xinyazhang/puzzle-geometry
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/.alpha_paper}"
mkdir -p "$DEST"
cd "$DEST"
cat > SHA256SUMS <<'SUMS'
b29b952f65d1b3acc768411779cd84edc1e9c3eee9d1761acdb25533bb0017a1  alpha-1.0.cfg
1ac388f879ef64e24440d23ee57463e78fcdf5e8a07f57054340fb06aefdaee9  alpha-1.0.dt.tcp.obj
ba5edbeb78589b1e0cdec4bc3b4f1bf5d4d9a0c5fcde640625b7a7bd3a9b8f53  alpha_env-1.0.dt.tcp.obj
501dfe586861427a7cf9561a032637878e4e8904d7a4c47826667afdea387a27  alpha-1.0.ompl.txt
SUMS
for f in $(awk '{print $2}' SHA256SUMS); do
  curl -sfL "https://gitlab.com/xinyazhang/puzzle-geometry/-/raw/${COMMIT}/puzzles/alpha/${f}" -o "$f"
done
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
echo "Alpha 1.0 (Zhang et al. 2020) files in $DEST"
