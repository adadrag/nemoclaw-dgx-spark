#!/bin/bash
# Atlas inference engine setup for DGX Spark
# Automates Phase 3 — replaces Ollama with Atlas for faster inference
#
# IMPORTANT: This stops Ollama. Do not run both engines simultaneously.
set -euo pipefail

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*"; exit 1; }

# ── Configuration (update these as new versions are released) ──
ATLAS_IMAGE="avarok/atlas-alpha2:latest"
MODEL_ID="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
PORT=8001

info "=== Atlas Inference Engine Setup ==="
info "Image: $ATLAS_IMAGE"
info "Model: $MODEL_ID"
info "Port:  $PORT"
echo ""

# ── Pre-flight checks ──
command -v docker > /dev/null || error "Docker not found."
nvidia-smi > /dev/null 2>&1 || error "GPU not detected."

# ── Stop Ollama ──
info "Stopping Ollama to free GPU memory..."
sudo systemctl stop ollama 2>/dev/null || true
docker stop atlas 2>/dev/null || true
docker rm atlas 2>/dev/null || true
sleep 2

# ── Pull Atlas image ──
info "Pulling Atlas Docker image ($ATLAS_IMAGE)..."
docker pull $ATLAS_IMAGE

# ── Download NVFP4 model weights ──
info "Checking NVFP4 model weights..."
if python3 -c "
from huggingface_hub import try_to_load_from_cache
result = try_to_load_from_cache('$MODEL_ID', 'config.json')
assert result is not None
" 2>/dev/null; then
    info "Model weights already cached."
else
    info "Downloading NVFP4 model weights (this may take a while)..."
    # Fix permissions if needed
    docker run --rm -v "$HOME/.cache/huggingface:/hf" alpine chown -R "$(id -u):$(id -g)" /hf 2>/dev/null || true
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')"
fi

# ── Launch Atlas ──
info "Launching Atlas..."
docker run -d --name atlas \
    --gpus all --ipc=host --network host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    $ATLAS_IMAGE serve $MODEL_ID \
    --port $PORT \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.88 \
    --scheduling-policy slai \
    --max-seq-len 8192 \
    --max-batch-size 16 \
    --tool-call-parser hermes \
    --ssm-cache-slots 8

# ── Wait for health ──
info "Waiting for Atlas to be ready..."
SECONDS=0
while [ $SECONDS -lt 300 ]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        info "Atlas ready in ${SECONDS}s"
        curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null
        break
    fi
    sleep 3
done

if [ $SECONDS -ge 300 ]; then
    error "Atlas did not start within 300s. Check: docker logs atlas"
fi

# ── Warmup ──
info "Running warmup (10 requests)..."
for i in $(seq 1 10); do
    curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":32}" > /dev/null
    echo "  warmup $i/10 done"
done

# ── Configure NemoClaw ──
info "Configuring NemoClaw to use Atlas..."
NEMOCLAW_EXPERIMENTAL=1 nemoclaw onboard \
    --endpoint vllm \
    --endpoint-url "http://host.openshell.internal:$PORT/v1" \
    --model "$MODEL_ID" \
    --api-key dummy 2>/dev/null || warn "NemoClaw onboard failed — configure manually if needed"

echo ""
info "=== Atlas Setup Complete ==="
info ""
info "Connect to your agent:  nemoclaw my-assistant connect"
info ""
info "To switch back to Ollama:"
info "  docker stop atlas"
info "  sudo systemctl start ollama"
info "  NEMOCLAW_EXPERIMENTAL=1 nemoclaw onboard --endpoint ollama --model nemotron-3-super:120b"
