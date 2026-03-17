# Design Spec: NemoClaw on DGX Spark

**Date:** 2026-03-17
**Status:** Approved
**Author:** tati + Claude

---

## 1. Purpose

Create a public GitHub repository (`nemoclaw-dgx-spark`) that serves as a comprehensive setup guide for running NemoClaw — NVIDIA's sandboxed OpenClaw agent framework — on a DGX Spark (GB10, 128GB unified memory, CUDA Compute Capability 12.1).

The guide covers cloud inference as a quick start, local inference via Ollama (Nemotron 3 Super 120B) as the primary path, and Atlas inference engine as an advanced high-performance alternative. It includes benchmarks comparing Ollama vs Atlas for Nemotron 3 Super 120B on this hardware.

## 2. Target Audience

- DGX Spark owners who want to run sandboxed AI agents locally
- Developers exploring NemoClaw and OpenClaw
- Users who want local inference instead of cloud-only

## 3. Repository Structure

```
nemoclaw-dgx-spark/
├── README.md                          # Comprehensive guide (main deliverable)
├── LICENSE                            # Apache 2.0
├── CLAUDE.md                          # Claude Code project instructions (WAT framework)
├── .gitignore
├── benchmarks/
│   ├── benchmark-nemotron.py          # Nemotron 3 Super 120B benchmark script
│   └── results/                       # Raw JSON benchmark results
└── scripts/
    ├── setup-nemoclaw.sh              # Automated NemoClaw + Ollama setup
    └── setup-atlas.sh                 # Atlas alternative inference setup
```

## 4. README Content Design

The README follows the actual install journey as a linear walkthrough.

### 4.1 Overview Section
- What NemoClaw is: OpenClaw plugin for NVIDIA OpenShell, sandboxed agent environment
- Architecture diagram (Mermaid format, renders natively on GitHub):
  ```
  DGX Spark > Docker (cgroupns=host) > OpenShell gateway > k3s > NemoClaw sandbox > OpenClaw agent
  ```
- What the reader will end up with: a sandboxed AI agent running on local GPU inference
- Model selection rationale: Nemotron 3 Super 120B (120B total, 12B active MoE, fits in 128GB)

### 4.2 Hardware & Prerequisites
- DGX Spark hardware specs table (GB10, 128GB, CUDA Compute Capability 12.1, ARM64)
- Software requirements:
  - Docker (pre-installed on DGX Spark, v28.x)
  - Node.js 22+
  - NVIDIA OpenShell CLI
  - NVIDIA API key (for initial cloud setup)
- OpenShell CLI installation (the NemoClaw bootstrap script installs it, but document manual install as fallback):
  ```bash
  ARCH=$(uname -m)
  curl -fsSL "https://github.com/NVIDIA/OpenShell/releases/latest/download/openshell-linux-${ARCH}" \
    -o /usr/local/bin/openshell && chmod +x /usr/local/bin/openshell
  ```
- Known DGX Spark quirks to address:
  - cgroup v2 incompatibility with k3s-in-Docker (needs `cgroupns=host`)
  - Docker socket permissions (user must be in docker group)
  - CoreDNS CrashLoop (needs gateway IP fix)
- "Tested With" table listing exact versions used during authoring (NemoClaw, Ollama, Atlas, Docker, Node.js, DGX OS)

### 4.3 Phase 1: Quick Start with NVIDIA Cloud
- Purpose: get NemoClaw running in 5 minutes before dealing with local inference
- Steps:
  1. Run `curl -fsSL https://nvidia.com/nemoclaw.sh | bash` — this handles Node.js, OpenShell, and NemoClaw installation. On DGX Spark, follow up with `sudo nemoclaw setup-spark` to fix cgroup v2 and Docker permissions (verified command from NemoClaw's `spark-install.md` and `scripts/setup-spark.sh`)
  2. Onboard with NVIDIA Build endpoint (cloud inference, Nemotron 3 Super 120B)
  3. Connect to sandbox: `nemoclaw my-assistant connect`
  4. Test via TUI: `openclaw tui`
  5. Verify: send a test message, confirm response
- Expected output: working agent using cloud inference

### 4.4 Phase 2: Local Inference with Ollama
- Purpose: move inference from cloud to local GPU for privacy, speed, and zero-cost
- Steps:
  1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
  2. Pull model: `ollama pull nemotron-3-super:120b` (~87GB download). Note: the exact Ollama model tag must be verified during implementation — if not in the Ollama library, document the manual GGUF import path as fallback
  3. Verify Ollama is serving: `curl http://localhost:11434/api/tags`
  4. Reconfigure NemoClaw for local Ollama:
     ```bash
     NEMOCLAW_EXPERIMENTAL=1 nemoclaw onboard \
       --endpoint ollama \
       --model nemotron-3-super:120b
     ```
  5. Verify agent works with local inference
- Document: the `host.openshell.internal:11434/v1` routing through the OpenShell gateway
- Expected: agent using local GPU inference, no cloud dependency

### 4.5 Phase 3: Advanced — Atlas Inference Engine (Optional)
- Purpose: 3x faster inference than Ollama for users who want maximum performance
- Prerequisites: must stop Ollama to free VRAM (cannot coexist)
- Steps:
  1. Stop Ollama: `systemctl stop ollama`
  2. Pull Atlas image: `docker pull avarok/atlas-alpha2:latest` (1.9GB)
  3. Download NVFP4 model: depends on available Nemotron 3 Super NVFP4 weights
  4. Launch Atlas with appropriate flags
  5. Configure NemoClaw to use Atlas as custom/vllm endpoint on port 8001
  6. Verify agent works through Atlas
- Include: warmup behavior (8-10 requests to reach full speed), operational notes
- Note: Atlas is AGPL-3.0 licensed, closed source, alpha software

### 4.6 Phase 4: Benchmarks
- Benchmark script (`benchmarks/benchmark-nemotron.py`) adapted from Atlas benchmark suite
- Tests to run on Nemotron 3 Super 120B:
  - Single request speed (tok/s, TTFT)
  - Medium generation (1024 tokens, 10 iterations for statistics)
  - Concurrency (1, 5, 10, 20 users with RAG-style prompts)
  - GPU memory usage snapshots
- Comparison table: Ollama vs Atlas (side by side)
- Warmup phase: 5 requests discarded before measurement begins (Atlas CUDA graph compilation, Ollama model warm-up)
- All raw JSON results saved in `benchmarks/results/`

### 4.7 Troubleshooting
- OOM when running multiple inference engines simultaneously (stop one first)
- cgroup v2 / k3s issues (cgroupns=host fix)
- HuggingFace cache permissions (Docker chown fix)
- CoreDNS CrashLoop (fix-coredns.sh)
- Docker permission denied (docker group)
- NemoClaw experimental endpoints not showing (NEMOCLAW_EXPERIMENTAL=1)

### 4.8 References
- NemoClaw docs: https://docs.nvidia.com/nemoclaw/latest/
- NemoClaw GitHub: https://github.com/NVIDIA/NemoClaw
- Atlas Discord (for CLI reference and updates)
- NVIDIA forums: Atlas inference engine thread
- Existing Qwen3.5 benchmarks: https://github.com/adadrag/qwen3.5-dgx-spark

## 5. Technical Decisions

### 5.1 NemoClaw Experimental Mode
Local inference (Ollama, vLLM) is behind `NEMOCLAW_EXPERIMENTAL=1` in the onboard wizard. The guide documents this explicitly. When enabled, the wizard detects running Ollama and auto-selects it, or offers it as an option.

### 5.2 Atlas Integration
Atlas is not a built-in NemoClaw provider. Integration options:
- Use `vllm` profile with `--endpoint-url http://host.openshell.internal:8001/v1`
- Or use `custom` endpoint type
The guide will document the exact non-interactive onboard flags.

### 5.3 Nemotron 3 Super 120B on Atlas
Atlas supports Nemotron models. The Discord docs list `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` as supported. The NVFP4 version for Super 120B exists on HuggingFace as `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` — this must be verified during implementation (Step 1 of the plan).

**If NVFP4 weights exist and Atlas loads them:** benchmark Ollama vs Atlas head-to-head on the same model.

**If NVFP4 weights do not exist or Atlas cannot load them:** the Atlas section becomes a reference to the existing `qwen3.5-dgx-spark` Atlas benchmarks, with a note that Atlas support for Nemotron 3 Super is pending. The benchmark section will only cover Ollama for Nemotron 3 Super, and link to the Atlas benchmark results from the sibling repo for cross-engine comparison.

### 5.4 Benchmark Script
Adapted from the Atlas benchmark suite (`atlas-benchmark/benchmark.py`). Changes:
- Default model set to Nemotron 3 Super 120B
- Ollama endpoint support (port 11434)
- Uses API-reported `response_token/s` when available (Atlas), falls back to client-side estimation (Ollama)

### 5.5 Script Scope Definitions

- **`scripts/setup-nemoclaw.sh`**: Automates Phase 1 + Phase 2 — installs NemoClaw, runs `setup-spark`, installs Ollama, pulls the model, and runs `nemoclaw onboard` with local Ollama endpoint. Users can also follow the README manually step by step.
- **`scripts/setup-atlas.sh`**: Automates Phase 3 — stops Ollama, pulls Atlas image, downloads NVFP4 model weights, launches Atlas, and reconfigures NemoClaw endpoint. Designed to be run after Phase 2 is working.

### 5.6 Atlas Image Tag Strategy
The Atlas image tag (`avarok/atlas-alpha2:latest`) is alpha software and will change. The `setup-atlas.sh` script uses a variable (`ATLAS_IMAGE`) at the top so users can update it in one place. The README includes a note pointing to the Atlas Discord for current image tags.

## 6. CLAUDE.md Design
Follows the WAT framework pattern from the existing qwen3.5-dgx-spark repo:
- Layer 1 (Workflows): markdown SOPs in the README phases
- Layer 2 (Agents): Claude Code as the orchestrator
- Layer 3 (Tools): scripts in `scripts/` and `benchmarks/`
- Self-improvement loop for discovered issues

## 7. GitHub Repository Standards
- Public repository under user's GitHub account
- Apache 2.0 license
- .gitignore for Python, Node.js, and common artifacts
- Clean initial commit structure

## 8. Out of Scope
- Nemotron Ultra 253B (does not fit on single DGX Spark — needs ~142GB NVFP4, only 128GB available)
- Multi-node clustering (2x DGX Spark)
- Production deployment hardening
- OpenClaw agent customization (focus is on infrastructure, not agent config)
