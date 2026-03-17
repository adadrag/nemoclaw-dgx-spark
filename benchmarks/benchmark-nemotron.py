#!/usr/bin/env python3
"""
Nemotron 3 Super 120B benchmark suite for DGX Spark.
Compares Ollama vs Atlas inference engines.
Adapted from https://github.com/adadrag/qwen3.5-dgx-spark
"""

import time
import json
import requests
import concurrent.futures
import statistics
import argparse
from datetime import datetime

ENGINE_DEFAULTS = {
    "ollama": {"url": "http://localhost:11434", "model": "nemotron-3-super:120b"},
    "atlas": {"url": "http://localhost:8001", "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"},
}


def chat_completion(base_url, model, prompt, max_tokens=128, stream=False):
    """Send a chat completion request and measure timing."""
    # Ollama uses /api/chat natively but also supports /v1/chat/completions
    api_url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
    }

    start = time.perf_counter()

    if stream:
        payload["stream"] = True
        resp = requests.post(api_url, json=payload, stream=True, timeout=300)
        first_token_time = None
        chunks = []
        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line != "data: [DONE]":
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            chunks.append(delta["content"])
                    except Exception:
                        pass
        end = time.perf_counter()
        content = "".join(chunks)
        ttft = (first_token_time - start) if first_token_time else None
        # Try to get server-reported metrics
        api_tps = None
    else:
        resp = requests.post(api_url, json=payload, timeout=300)
        end = time.perf_counter()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        ttft = None
        # Check for Atlas-style extended usage metrics
        usage = data.get("usage", {})
        api_tps = usage.get("response_token/s")

    elapsed = end - start
    word_count = len(content.split())
    estimated_tokens = int(word_count * 1.3)

    if api_tps:
        tok_per_sec = api_tps
    else:
        tok_per_sec = estimated_tokens / elapsed if elapsed > 0 else 0

    return {
        "elapsed": round(elapsed, 2),
        "output_chars": len(content),
        "estimated_tokens": estimated_tokens,
        "tok_per_sec": round(tok_per_sec, 1),
        "api_tps": round(api_tps, 1) if api_tps else None,
        "ttft": round(ttft, 3) if ttft else None,
        "content_preview": content[:100] + "..." if len(content) > 100 else content,
    }


def run_warmup(base_url, model, count=5):
    """Discard warmup requests to let CUDA graphs compile."""
    print(f"\n  Warming up ({count} requests)...")
    for i in range(count):
        chat_completion(base_url, model, "Say hello.", max_tokens=32)
        print(f"    warmup {i+1}/{count} done")
    print("  Warmup complete.\n")


def run_single_benchmarks(base_url, model):
    """Single-request speed tests."""
    print("\n" + "=" * 60)
    print("SINGLE REQUEST BENCHMARKS")
    print("=" * 60)

    tests = [
        ("Short response", "Say hello and introduce yourself in one sentence.", 128),
        ("Medium response", "Explain quantum computing in detail. Cover qubits, superposition, entanglement, and current applications.", 1024),
        ("Long response", "Write a comprehensive guide to setting up a production Kubernetes cluster, including networking, storage, monitoring, security hardening, and disaster recovery.", 4096),
        ("Code generation", "Write a Python async web scraper using aiohttp that handles rate limiting, retries, and saves results to SQLite. Include error handling and logging.", 2048),
        ("Reasoning", "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Think step by step.", 256),
    ]

    results = []
    for name, prompt, max_tokens in tests:
        print(f"\n--- {name} (max_tokens={max_tokens}) ---")
        result = chat_completion(base_url, model, prompt, max_tokens, stream=True)
        tps_label = f"{result['api_tps']} tok/s (API)" if result["api_tps"] else f"~{result['tok_per_sec']} tok/s (est)"
        print(f"  Time: {result['elapsed']}s | {tps_label} | TTFT: {result['ttft']}s")
        results.append({"test": name, **result})

    return results


def run_concurrency_benchmark(base_url, model, num_users_list=None):
    """Concurrency benchmark with RAG-style prompts."""
    if num_users_list is None:
        num_users_list = [1, 5, 10, 20]

    print("\n" + "=" * 60)
    print("CONCURRENCY BENCHMARKS")
    print("=" * 60)

    system_prompt = "You are a helpful company assistant. Answer based on the provided context."
    context = "Our company was founded in 2020. We have 150 employees across London, Berlin, and Tokyo. " * 50
    question = "How many employees does the company have and where are they located?"
    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}"

    results = []
    for n in num_users_list:
        print(f"\n--- {n} concurrent users ---")

        def single_request(_):
            return chat_completion(base_url, model, prompt, max_tokens=200)

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            futures = [executor.submit(single_request, i) for i in range(n)]
            individual = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.perf_counter() - start

        per_user_speeds = [r["tok_per_sec"] for r in individual]
        total_tokens = sum(r["estimated_tokens"] for r in individual)
        aggregate_tps = total_tokens / total_time

        print(
            f"  Per-user avg: {statistics.mean(per_user_speeds):.1f} tok/s | "
            f"Aggregate: {aggregate_tps:.1f} tok/s | "
            f"Avg latency: {statistics.mean([r['elapsed'] for r in individual]):.1f}s"
        )

        results.append({
            "concurrent_users": n,
            "per_user_avg_tps": round(statistics.mean(per_user_speeds), 1),
            "aggregate_tps": round(aggregate_tps, 1),
            "avg_latency": round(statistics.mean([r["elapsed"] for r in individual]), 1),
            "total_time": round(total_time, 1),
        })

    return results


def run_claims_validation(base_url, model):
    """10-iteration speed test for statistical validation."""
    print("\n" + "=" * 60)
    print("SPEED VALIDATION (10 iterations, 1024 tokens)")
    print("=" * 60)

    prompt = "Explain quantum computing in detail. Cover qubits, superposition, entanglement, and current applications."
    speeds = []

    for i in range(10):
        result = chat_completion(base_url, model, prompt, max_tokens=1024, stream=True)
        tps = result["api_tps"] or result["tok_per_sec"]
        speeds.append(tps)
        print(f"  Run {i+1}/10: {tps} tok/s ({result['estimated_tokens']} est. tokens in {result['elapsed']}s)")

    print(f"\n  Mean:   {statistics.mean(speeds):.1f} tok/s")
    print(f"  Median: {statistics.median(speeds):.1f} tok/s")
    print(f"  Stddev: {statistics.stdev(speeds):.1f} tok/s")
    print(f"  Min:    {min(speeds):.1f} tok/s")
    print(f"  Max:    {max(speeds):.1f} tok/s")

    return {
        "iterations": 10,
        "mean": round(statistics.mean(speeds), 1),
        "median": round(statistics.median(speeds), 1),
        "stddev": round(statistics.stdev(speeds), 1),
        "min": round(min(speeds), 1),
        "max": round(max(speeds), 1),
        "individual_runs": speeds,
    }


def main():
    parser = argparse.ArgumentParser(description="Nemotron 3 Super 120B benchmark")
    parser.add_argument("--engine", choices=["ollama", "atlas"], default="ollama",
                       help="Inference engine to benchmark")
    parser.add_argument("--url", help="Override server URL")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--test", choices=["all", "single", "concurrency", "claims"],
                       default="all", help="Which test suite to run")
    parser.add_argument("--output", help="Output JSON file (default: <engine>-results.json)")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup phase")
    args = parser.parse_args()

    defaults = ENGINE_DEFAULTS[args.engine]
    url = args.url or defaults["url"]
    model = args.model or defaults["model"]
    output = args.output or f"{args.engine}-results.json"

    all_results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "engine": args.engine,
            "url": url,
            "model": model,
            "hardware": "NVIDIA DGX Spark, GB10, 128GB unified, CC 12.1",
        },
        "results": {},
    }

    print(f"\n{'#' * 60}")
    print(f"# BENCHMARKING: {args.engine.upper()} @ {url}")
    print(f"# Model: {model}")
    print(f"{'#' * 60}")

    # Verify server is up
    try:
        r = requests.get(f"{url}/v1/models", timeout=10)
        models = r.json()
        model_id = models.get("data", [{}])[0].get("id", "unknown")
        print(f"Server OK — model: {model_id}")
    except Exception as e:
        print(f"Server at {url} not reachable: {e}")
        print("Make sure the inference engine is running.")
        return

    # Warmup
    if not args.skip_warmup:
        run_warmup(url, model)

    # Run tests
    if args.test in ("all", "single"):
        all_results["results"]["single"] = run_single_benchmarks(url, model)

    if args.test in ("all", "concurrency"):
        all_results["results"]["concurrency"] = run_concurrency_benchmark(url, model)

    if args.test in ("all", "claims"):
        all_results["results"]["claims"] = run_claims_validation(url, model)

    # Save results
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
