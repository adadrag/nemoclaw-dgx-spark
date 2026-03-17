#!/bin/bash
# NemoClaw + Ollama setup for DGX Spark
# Automates Phase 1 (cloud quick-start) and Phase 2 (local Ollama inference)
set -euo pipefail

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*"; exit 1; }

OLLAMA_MODEL="nemotron-3-super:120b"

# ── Pre-flight checks ──
info "=== NemoClaw + Ollama Setup for DGX Spark ==="
echo ""

command -v docker > /dev/null || error "Docker not found. DGX Spark should have Docker pre-installed."
nvidia-smi > /dev/null 2>&1 || error "nvidia-smi not found. GPU not detected."

# ── Phase 1: Install NemoClaw ──
info "Phase 1: Installing NemoClaw..."

if command -v nemoclaw > /dev/null 2>&1; then
    info "NemoClaw already installed: $(command -v nemoclaw)"
else
    info "Downloading and running NemoClaw installer..."
    curl -fsSL https://nvidia.com/nemoclaw.sh | bash
fi

# DGX Spark compatibility fix
info "Running DGX Spark setup (cgroup v2 + Docker permissions)..."
info "This requires sudo — you may be prompted for your password."
sudo nemoclaw setup-spark

info "Phase 1 complete. NemoClaw installed."
echo ""

# ── Phase 2: Install Ollama + Local Inference ──
info "Phase 2: Setting up local inference with Ollama..."

if command -v ollama > /dev/null 2>&1; then
    info "Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
else
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Pull the model
info "Pulling $OLLAMA_MODEL (~87GB download)..."
ollama pull $OLLAMA_MODEL

# Verify Ollama is serving
info "Verifying Ollama is serving..."
if curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; models=[m['name'] for m in json.load(sys.stdin).get('models',[])]; print('Models:', models)" 2>/dev/null; then
    info "Ollama is serving."
else
    warn "Ollama may not be running. Start it with: ollama serve"
fi

# Configure NemoClaw for local Ollama
info "Configuring NemoClaw for local Ollama inference..."
NEMOCLAW_EXPERIMENTAL=1 nemoclaw onboard \
    --endpoint ollama \
    --model $OLLAMA_MODEL

info "Phase 2 complete."
echo ""
info "=== Setup Complete ==="
info ""
info "Connect to your agent:  nemoclaw my-assistant connect"
info "Monitor sandbox:        openshell term"
info ""
info "To switch to Atlas (faster inference), run: scripts/setup-atlas.sh"
