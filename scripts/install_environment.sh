#!/usr/bin/env bash
# ============================================================================
# install_environment.sh — Bootstrap the drawing-to-CAD system environment
#
# Designed for rootless Debian containers (e.g., running as uid=1000 "node").
# No sudo/root, no curl/wget, no apt-get required.
#
# Prerequisites:
#   - Node.js >= 18 (used as HTTPS download shim when curl/wget absent)
#   - Write access to $HOME (~)
#   - /app mounted with the project source (can be read-only except training_data/)
#
# What it installs:
#   - A Node.js "curl" shim (if curl/wget are missing)
#   - uv (Astral's Python package manager)
#   - Python 3.12 (via uv)
#   - Python venv at ~/cad-venv with all pip dependencies
#   - micromamba + conda-forge system libraries (cairo, GL, mesa, X11)
#   - Activation script at ~/activate_cad.sh
#
# Usage:
#   bash scripts/install_environment.sh
#   source ~/activate_cad.sh
# ============================================================================
set -euo pipefail

HOME_DIR="$HOME"
PROJECT_ROOT="/app"
VENV_DIR="$HOME_DIR/cad-venv"
MAMBA_ROOT="$HOME_DIR/micromamba"
MAMBA_ENV="syslibs"
ACTIVATE_SCRIPT="$HOME_DIR/activate_cad.sh"
CURL_SHIM="$HOME_DIR/.local/bin/curl-shim"

echo "=== Drawing-to-CAD Environment Installer (rootless) ==="
echo "Home:         $HOME_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Venv:         $VENV_DIR"
echo ""

# ========================================================================== #
# Step 0: Ensure we have an HTTPS download tool
# ========================================================================== #
echo "[0/7] Setting up download tool..."

if command -v curl &>/dev/null; then
    FETCH="curl"
    echo "  curl found at $(command -v curl)"
elif command -v wget &>/dev/null; then
    FETCH="wget"
    echo "  wget found at $(command -v wget)"
elif command -v node &>/dev/null; then
    # Create a curl-compatible shim using Node.js
    echo "  No curl/wget found. Creating Node.js HTTPS download shim..."
    mkdir -p "$(dirname "$CURL_SHIM")"
    cat > "$CURL_SHIM" << 'NODECURL'
#!/bin/bash
# Minimal curl replacement using Node.js for HTTPS downloads.
# Supports: -o FILE URL (download to file) and bare URL (stdout)
OUTPUT=""
URL=""
SILENT=0
CHECK_MODE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK_MODE=1; shift ;;
        -o) OUTPUT="$2"; shift 2 ;;
        -sSfL|-sSf|-sS|-s|-S|-f|-L) SILENT=1; shift ;;
        -*) shift ;;
        *) URL="$1"; shift ;;
    esac
done

[ "$CHECK_MODE" -eq 1 ] && exit 0
[ -z "$URL" ] && { echo "Error: No URL provided" >&2; exit 1; }

if [ -n "$OUTPUT" ]; then
    node -e "
const https = require('https');
const http = require('http');
const fs = require('fs');
function download(u, dest, depth) {
    if (depth > 10) { process.exit(1); }
    const mod = u.startsWith('https') ? https : http;
    mod.get(u, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            let loc = res.headers.location;
            if (loc.startsWith('/')) { const p = new URL(u); loc = p.protocol+'//'+p.host+loc; }
            download(loc, dest, depth + 1);
            return;
        }
        if (res.statusCode !== 200) { process.exit(22); }
        const file = fs.createWriteStream(dest);
        res.pipe(file);
        file.on('finish', () => file.close());
    }).on('error', () => process.exit(1));
}
download(process.argv[1], process.argv[2], 0);
" "$URL" "$OUTPUT"
else
    node -e "
const https = require('https');
const http = require('http');
function download(u, depth) {
    if (depth > 10) { process.exit(1); }
    const mod = u.startsWith('https') ? https : http;
    mod.get(u, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            let loc = res.headers.location;
            if (loc.startsWith('/')) { const p = new URL(u); loc = p.protocol+'//'+p.host+loc; }
            download(loc, depth + 1);
            return;
        }
        if (res.statusCode !== 200) { process.exit(22); }
        res.pipe(process.stdout);
    }).on('error', () => process.exit(1));
}
download(process.argv[1], 0);
" "$URL"
fi
NODECURL
    chmod +x "$CURL_SHIM"
    FETCH="$CURL_SHIM"
    echo "  Created shim at $CURL_SHIM"
else
    echo "ERROR: No download tool available (need curl, wget, or node)" >&2
    exit 1
fi

# Helper: download URL to file
fetch_to_file() {
    local url="$1" dest="$2"
    case "$FETCH" in
        curl|*curl-shim) "$FETCH" -sSfL -o "$dest" "$url" ;;
        wget) wget -q -O "$dest" "$url" ;;
    esac
}

# Helper: download URL to stdout
fetch_to_stdout() {
    local url="$1"
    case "$FETCH" in
        curl|*curl-shim) "$FETCH" -sSf "$url" ;;
        wget) wget -q -O- "$url" ;;
    esac
}

echo ""

# ========================================================================== #
# Step 1: Install uv (Python package manager)
# ========================================================================== #
echo "[1/7] Installing uv (Astral Python package manager)..."

if command -v uv &>/dev/null; then
    echo "  uv already installed: $(uv --version)"
else
    mkdir -p "$HOME_DIR/.local/bin"
    export PATH="$HOME_DIR/.local/bin:$PATH"

    # Download and run the uv installer
    fetch_to_stdout "https://astral.sh/uv/install.sh" | sh
    echo "  Installed: $(uv --version)"
fi

export PATH="$HOME_DIR/.local/bin:$PATH"
echo ""

# ========================================================================== #
# Step 2: Install Python 3.12 via uv
# ========================================================================== #
echo "[2/7] Installing Python 3.12..."

uv python install 3.12
PYTHON_PATH="$(uv python find 3.12)"
echo "  Python: $PYTHON_PATH ($($PYTHON_PATH --version))"
echo ""

# ========================================================================== #
# Step 3: Create Python virtual environment
# ========================================================================== #
echo "[3/7] Creating virtual environment at $VENV_DIR..."

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python3" ]; then
    echo "  Existing venv found, reusing."
else
    # Create venv in home dir (avoids permission issues on mounted /app)
    uv venv --python 3.12 "$VENV_DIR"
fi

export PATH="$VENV_DIR/bin:$HOME_DIR/.local/bin:$PATH"
export VIRTUAL_ENV="$VENV_DIR"

echo "  venv python: $(python3 --version)"
echo ""

# ========================================================================== #
# Step 4: Install Python dependencies
# ========================================================================== #
echo "[4/7] Installing Python dependencies..."

# Core dependencies from pyproject.toml
echo "  Installing core deps..."
uv pip install --python "$VENV_DIR/bin/python3" \
    "pydantic>=2.0" \
    "pyyaml>=6.0" \
    "structlog>=23.0" \
    "httpx>=0.25.0" \
    "numpy>=1.24" \
    "Pillow>=10.0" \
    "cairosvg>=2.7.0"

# build123d pulls in cadquery-ocp (OpenCASCADE Python bindings)
echo "  Installing build123d + OCP (large download, ~400MB)..."
uv pip install --python "$VENV_DIR/bin/python3" "build123d>=0.7.0"

# trimesh with STEP support via cascadio backend
echo "  Installing trimesh + cascadio..."
uv pip install --python "$VENV_DIR/bin/python3" "trimesh>=4.0" "cascadio"

# Optional dependencies
echo "  Installing optional deps (optuna, pytest)..."
uv pip install --python "$VENV_DIR/bin/python3" \
    "optuna>=3.4" \
    "pytest>=7.0" \
    "pytest-asyncio>=0.21" \
    2>/dev/null || echo "  Note: some optional deps may have failed"

echo "  Python dependencies installed."
echo ""

# ========================================================================== #
# Step 5: Install system libraries via micromamba (rootless)
# ========================================================================== #
echo "[5/7] Installing system libraries via micromamba (rootless conda-forge)..."

# OCP/build123d and cairosvg need shared libraries (libGL, libcairo, X11)
# that are normally installed via apt. Without root, we use micromamba to
# get them from conda-forge into a local prefix.

if [ -d "$MAMBA_ROOT/envs/$MAMBA_ENV/lib" ] && \
   [ -f "$MAMBA_ROOT/envs/$MAMBA_ENV/lib/libGL.so.1" ] && \
   [ -f "$MAMBA_ROOT/envs/$MAMBA_ENV/lib/libcairo.so.2" ]; then
    echo "  System libs already installed, skipping."
else
    MAMBA_BIN="$HOME_DIR/.local/bin/micromamba"
    if [ ! -f "$MAMBA_BIN" ]; then
        echo "  Downloading micromamba..."
        fetch_to_file \
            "https://micro.mamba.pm/api/micromamba/linux-64/latest" \
            "/tmp/micromamba.tar.bz2"
        mkdir -p "$(dirname "$MAMBA_BIN")"
        tar -xjf /tmp/micromamba.tar.bz2 -C /tmp "bin/micromamba"
        mv /tmp/bin/micromamba "$MAMBA_BIN"
        chmod +x "$MAMBA_BIN"
        rm -rf /tmp/bin /tmp/micromamba.tar.bz2
        echo "  micromamba installed at $MAMBA_BIN"
    fi

    export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"

    echo "  Creating syslibs environment with cairo, GL, mesa, X11..."
    "$MAMBA_BIN" create -n "$MAMBA_ENV" -y -c conda-forge \
        cairo \
        mesalib \
        libgl \
        libglu \
        xorg-libx11 \
        xorg-libxext \
        xorg-libxrender
fi

export CONDA_PREFIX="$MAMBA_ROOT/envs/$MAMBA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "  System libs: $CONDA_PREFIX/lib"
echo ""

# ========================================================================== #
# Step 6: Validate the installation
# ========================================================================== #
echo "[6/7] Validating installation..."

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python3 "$PROJECT_ROOT/scripts/validate_install.py"
echo ""

# ========================================================================== #
# Step 7: Create activation script
# ========================================================================== #
echo "[7/7] Creating activation script..."

cat > "$ACTIVATE_SCRIPT" << EOF
#!/usr/bin/env bash
# Source this file to activate the drawing-to-CAD environment:
#   source ~/activate_cad.sh
export PATH="\$HOME/.local/bin:$VENV_DIR/bin:\$PATH"
export PYTHONPATH="$PROJECT_ROOT:\$PYTHONPATH"
export CONDA_PREFIX="$MAMBA_ROOT/envs/$MAMBA_ENV"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "CAD environment activated. Python: \$(python3 --version)"
EOF
chmod +x "$ACTIVATE_SCRIPT" 2>/dev/null || true

echo "  Created $ACTIVATE_SCRIPT"
echo ""

# ========================================================================== #
# Step 8: Verify training data
# ========================================================================== #
echo "[Bonus] Checking training data..."

SVG_COUNT=$(ls "$PROJECT_ROOT/training_data/drawings_svg/"*.svg 2>/dev/null | wc -l)
STEP_COUNT=$(ls "$PROJECT_ROOT/training_data/shapes_step/"*.step 2>/dev/null | wc -l)

echo "  SVG files:  $SVG_COUNT"
echo "  STEP files: $STEP_COUNT"

if [ "$SVG_COUNT" -eq 0 ] || [ "$STEP_COUNT" -eq 0 ]; then
    echo "  WARNING: Training data directories are empty or missing."
else
    echo "  Training data present."
fi
echo ""

# ========================================================================== #
# Done
# ========================================================================== #
echo "============================================"
echo "  Installation complete."
echo ""
echo "  To activate:    source ~/activate_cad.sh"
echo ""
echo "  Next steps:"
echo "    1. source ~/activate_cad.sh"
echo "    2. python3 scripts/preprocess_training_data.py"
echo "    3. python3 -m pytest tests/test_training.py -v"
echo "    4. python3 scripts/run_optimizer.py --mock --max-iterations 3"
echo "============================================"
