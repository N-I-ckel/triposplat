#!/usr/bin/env bash
set -euo pipefail

# TripoSplat Apple-Silicon hybrid setup.
# MLX-port source is pinned for reproducibility and installed without its broad
# dependency set; this repo supplies the original TripoSplat implementation.
MLX_PORT_COMMIT="821c1a8f090770265e8b02e8981fd1c4466a30c9"
MLX_PORT_URL="git+https://github.com/Jup33Q/TripoSplat-Klein-MLX.git@${MLX_PORT_COMMIT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m pip install -r requirements-mlx.txt
"${PYTHON_BIN}" -m pip install --no-deps "${MLX_PORT_URL}"
"${PYTHON_BIN}" scripts/download_mlx_weights.py "$@"

cat <<'EOF'

Apple-Silicon TripoSplat setup complete.
Try:
  python3 run_mlx.py input.png -o output.splat
EOF
