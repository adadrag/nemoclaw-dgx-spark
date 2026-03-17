# Agent Instructions

You're working inside the **WAT framework** (Workflows, Agents, Tools).

## Project Context

This repository is a setup guide for running **NemoClaw** (NVIDIA's sandboxed OpenClaw agent framework) on a **DGX Spark** (GB10, 128GB unified memory, CUDA CC 12.1, ARM64).

- **DGX Spark IP**: 192.168.42.2
- **SSH alias**: `dgx-spark` (user: adadrag, key: ~/.ssh/dgx_spark)
- **Sibling repo**: `qwen3.5-dgx-spark` (Atlas benchmarks for Qwen3.5-35B-A3B)

## The WAT Architecture

**Layer 1: Workflows (The Instructions)**
- The README phases are the SOPs: cloud quick-start, Ollama local, Atlas advanced
- Each phase is a linear walkthrough with exact commands

**Layer 2: Agents (The Decision-Maker)**
- You orchestrate the setup, run benchmarks, and update documentation
- Read the relevant phase, execute commands via SSH, handle failures

**Layer 3: Tools (The Execution)**
- `scripts/setup-nemoclaw.sh` — automates NemoClaw + Ollama install
- `scripts/setup-atlas.sh` — automates Atlas inference engine setup
- `benchmarks/benchmark-nemotron.py` — Nemotron 3 Super 120B benchmark suite

## Critical Rules

1. **Never run multiple inference engines simultaneously** on the DGX Spark. Stop one before starting another (OOM risk — causes system freeze requiring power cycle).
2. **Always stop Ollama before starting Atlas**: `systemctl stop ollama`
3. **Always stop Atlas before starting Ollama**: `docker stop atlas`
4. **HuggingFace cache is owned by root** — fix with: `docker run --rm -v $HOME/.cache/huggingface:/hf alpine chown -R $(id -u):$(id -g) /hf`
5. **NemoClaw local inference requires**: `NEMOCLAW_EXPERIMENTAL=1`

## How to Operate

1. Look for existing tools first — check `scripts/` and `benchmarks/`
2. When things fail: read the error, fix the script, retest, update the README
3. Keep the README current with any new findings
