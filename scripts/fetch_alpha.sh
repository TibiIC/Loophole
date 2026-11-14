#!/bin/bash
# Download the canonical OMPL.app Alpha puzzle files (pinned commit, sha256-checked).
#
# These files are under the Rice University Software Distribution License
# (non-commercial use; "Redistribution is not permitted"). They are NEVER committed:
# they are fetched here (locally: `make fetch-alpha`) or at Docker build time, and the
# derived convex hulls are generated next to them (gitignored / not in the repo).
# Usage: scripts/fetch_alpha.sh [DEST]   (default: <repo>/.ompl_alpha)
set -euo pipefail
COMMIT=958884d6593809ee6a564907ff4e796691d48ffc   # ompl/omplapp main, 2026-11-14
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/.ompl_alpha}"
mkdir -p "$DEST"
cd "$DEST"
cat > SHA256SUMS <<'SUMS'
253df4e9c474c519cb953ef5db0cd65bd8dcb5438a6cc034decc3b132a583896  alpha-1.0.cfg
4e5ec59b62063c8865d379a14e1dd9c532f39bb27bc0c4c2ebf8c3d06d6497ff  alpha-1.2.cfg
29d0f1ca11dd4955d039fb6a4ccf8101e1c67acd0f2d8409eebf73f4f712f197  alpha-1.5.cfg
b09b7866d4251ba0d4fbc0f47a5cfd40a50865852dffc9db1282c030100c0fb9  alpha-1.5.path
1434fef6507f7408109c4b5bc438b4fe0eff0f8d282955a55aedaeb81a8c00b5  alpha_env-1.0.dae
17d56eee3322d4a5e5535dfefbe6965e824c10385e41e67606addcf6fd935004  alpha_env-1.2.dae
503c5715b4fa1384ff9c5433b08873fb7c243426cf5a39e1323a4a7206170ea8  alpha_env-1.5.dae
d01ec840996df502ef292e19fd6e7ad8cb8708c47a4b17df05a5afc0923ab89e  alpha_robot.dae
SUMS
for f in $(awk '{print $2}' SHA256SUMS); do
  curl -sfL "https://raw.githubusercontent.com/ompl/omplapp/${COMMIT}/resources/3D/${f}" -o "$f"
done
curl -sfL "https://raw.githubusercontent.com/ompl/omplapp/${COMMIT}/LICENSE" -o LICENSE
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
echo "Canonical files in $DEST (licence: $DEST/LICENSE)"
