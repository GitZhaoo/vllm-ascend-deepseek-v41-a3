#!/usr/bin/env python3
"""Experiment benchmark suite for DeepSeek-V4.1-Flash on Ascend A3.

Runs the same measurement set for every config so A/B numbers are comparable:
  * single-stream hot-prefix (C=1): TTFT, decode tok/s, ms/step, tokens/step, acceptance
  * single-stream cold-prefix TTFT (unique first block each time)
  * dual-prefix concurrency sweep: aggregate wall-clock tok/s, per-stream tok/s, TTFT
  * /metrics deltas for iteration count (ms/step) and spec-decode acceptance

Usage:
  python3 bench_suite.py --base-url http://127.0.0.1:8021 --out /cache/exp/x.json
Requires: aiohttp.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import urllib.request

import aiohttp

PARAGRAPH = (
    "Transformer架构的核心是自注意力机制。对于输入序列X，通过可学习的权重矩阵W_Q、W_K、W_V"
    "将其投影到Query、Key、Value三个表示空间。注意力权重通过Q和K的点积计算，除以√d_k缩放后"
    "经softmax归一化，再与V相乘。多头注意力将QKV投影到多个子空间独立计算后拼接。"
    "位置编码注入位置信息，RoPE对Q和K应用旋转矩阵编码相对位置，外推性好。"
    "每层包含注意力子层、残差连接、LayerNorm、前馈网络FFN。现代变体用RMSNorm替代LayerNorm，"
    "用SwiGLU替代ReLU。GPT系列只用Decoder，BERT只用Encoder。大语言模型都基于Transformer。"
)

METRICS = [
    "vllm:iteration_tokens_total_count",
    "vllm:iteration_tokens_total_sum",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:request_inference_time_seconds_sum",
    "vllm:request_inference_time_seconds_count",
]


def snapshot(url: str) -> dict:
    try:
        txt = urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10).read().decode()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    out: dict[str, float] = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        for name in METRICS:
            if line.startswith(name + "{") or line.startswith(name + " "):
                try:
                    out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
                break
    return out


async def one_request(session, url, payload):
    started = time.perf_counter()
    ttft = None
    last = started
    usage = 0
    chunks = 0
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=1800)) as resp:
        if resp.status != 200:
            return {"error": f"HTTP {resp.status}: {(await resp.text())[:200]}"}
        async for raw in resp.content:
            line = raw.strip()
            if not line or line == b"data: [DONE]" or not line.startswith(b"data: "):
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"].get("completion_tokens", usage)
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - started
                last = time.perf_counter()
                chunks += 1
    tokens = usage or chunks
    decode = max(last - started - (ttft or 0.0), 1e-9)
    return {"ttft": ttft or 0.0, "tokens": tokens,
            "decode_tps": (tokens - 1) / decode if tokens > 1 else 0.0,
            "total": last - started,
            "first_ts": started, "last_ts": last}


def payload_for(model, prompt, max_tokens):
    return {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}


def derive_step_metrics(before, after, requests):
    """Combine /metrics deltas with wall-clock token windows."""
    iters = (after.get("vllm:iteration_tokens_total_count", 0) - before.get("vllm:iteration_tokens_total_count", 0))
    draft = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - before.get("vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    gens = after.get("vllm:generation_tokens_total", 0) - before.get("vllm:generation_tokens_total", 0)
    gen_tokens = sum(r["tokens"] for r in requests if "error" not in r)
    # decode window: from first token to last token of the concurrent batch
    t0 = min(r["first_ts"] for r in requests if "error" not in r)
    t1 = max(r["last_ts"] for r in requests if "error" not in r)
    window = max(t1 - t0, 1e-9)
    # subtract the single scheduler step that produced the first tokens
    decode_iters = max(iters - 1, 1)
    pf_sum = after.get("vllm:request_prefill_time_seconds_sum", 0) - before.get("vllm:request_prefill_time_seconds_sum", 0)
    pf_cnt = after.get("vllm:request_prefill_time_seconds_count", 0) - before.get("vllm:request_prefill_time_seconds_count", 0)
    ttft_sum = after.get("vllm:time_to_first_token_seconds_sum", 0) - before.get("vllm:time_to_first_token_seconds_sum", 0)
    ttft_cnt = after.get("vllm:time_to_first_token_seconds_count", 0) - before.get("vllm:time_to_first_token_seconds_count", 0)
    return {
        "engine_prefill_ms": (pf_sum / pf_cnt * 1000.0) if pf_cnt else None,
        "engine_ttft_ms": (ttft_sum / ttft_cnt * 1000.0) if ttft_cnt else None,
        "iters": iters,
        "decode_window_s": window,
        "ms_per_step": window / decode_iters * 1000.0,
        "tokens_per_gen": gen_tokens,
        "acceptance": (accepted / draft) if draft else None,
        "tokens_per_step_metric": (gens / iters) if iters else None,
    }


async def single_stream(base_url, model, headers, prompt, max_tokens, repeats):
    url = base_url.rstrip("/") + "/v1/completions"
    async with aiohttp.ClientSession(headers=headers) as session:
        # warm-up (also fills prefix cache for the hot-prefix case)
        await one_request(session, url, payload_for(model, prompt, 16))
        await asyncio.sleep(1.0)
        rows = []
        for _ in range(repeats):
            before = snapshot(base_url)
            r = await one_request(session, url, payload_for(model, prompt, max_tokens))
            after = snapshot(base_url)
            if "error" in r:
                rows.append(r)
                continue
            step = derive_step_metrics(before, after, [r])
            rows.append({**r, **step})
            await asyncio.sleep(1.0)
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"error": rows[0]["error"] if rows else "no data"}
    return {
        "runs": [{k: v for k, v in r.items() if k not in ("first_ts", "last_ts")} for r in ok],
        "ttft_ms": statistics.median(r["ttft"] for r in ok) * 1000,
        "decode_tps": statistics.median(r["decode_tps"] for r in ok),
        "ms_per_step": statistics.median(r["ms_per_step"] for r in ok),
        "acceptance": (statistics.median([r["acceptance"] for r in ok if r["acceptance"]])
                       if any(r["acceptance"] for r in ok) else None),
        "tokens": statistics.median(r["tokens"] for r in ok),
        "engine_prefill_ms": statistics.median([r["engine_prefill_ms"] for r in ok if r.get("engine_prefill_ms")]),
        "engine_ttft_ms": statistics.median([r["engine_ttft_ms"] for r in ok if r.get("engine_ttft_ms")]),
    }


async def cold_ttft(base_url, model, headers, max_tokens, repeats):
    url = base_url.rstrip("/") + "/v1/completions"
    rows = []
    async with aiohttp.ClientSession(headers=headers) as session:
        for i in range(repeats):
            prompt = f"COLD-{time.time_ns()}-{i} " + PARAGRAPH * 3
            r = await one_request(session, url, payload_for(model, prompt, max_tokens))
            rows.append(r)
            await asyncio.sleep(0.5)
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"error": rows[0]["error"] if rows else "no data"}
    return {"ttft_ms": statistics.median(r["ttft"] for r in ok) * 1000,
            "decode_tps": statistics.median(r["decode_tps"] for r in ok)}


async def dual_level(base_url, model, headers, prompts, max_tokens, concurrency, repeats):
    url = base_url.rstrip("/") + "/v1/completions"
    agg, streams, ttfts = [], [], []
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        for _ in range(repeats):
            payloads = [payload_for(model, prompts[i % len(prompts)], max_tokens)
                        for i in range(concurrency)]
            started = time.perf_counter()
            results = await asyncio.gather(*[one_request(session, url, p) for p in payloads])
            wall = time.perf_counter() - started
            ok = [r for r in results if "error" not in r]
            if not ok:
                return {"concurrency": concurrency, "error": results[0].get("error", "unknown")}
            agg.append(sum(r["tokens"] for r in ok) / wall)
            streams.append(statistics.median(r["decode_tps"] for r in ok))
            ttfts.append(statistics.median(r["ttft"] for r in ok))
    return {"concurrency": concurrency,
            "aggregate": statistics.median(agg),
            "stream": statistics.median(streams),
            "ttft_ms": statistics.median(ttfts) * 1000}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("EXP_BASE_URL", "http://127.0.0.1:8021"))
    ap.add_argument("--model", default=os.environ.get("DSV41_SERVED_NAME", "DeepSeek-V4.1-Flash-1017"))
    ap.add_argument("--api-key", default=os.environ.get("DSV41_API_KEY", ""))
    ap.add_argument("--out", default="/cache/exp/bench.json")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dual-concurrency", default="2,8,16,32,64")
    ap.add_argument("--skip-dual", action="store_true")
    args = ap.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    hot_prompt = "HOT-" + PARAGRAPH * 3
    result = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "base_url": args.base_url}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def save(a, r):
        with open(a.out, "w") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=2)

    result["single_hot"] = await single_stream(
        args.base_url, args.model, headers, hot_prompt, args.max_tokens, max(args.repeats, 3))
    save(args, result)
    print("single_hot:", json.dumps(result["single_hot"], ensure_ascii=False)[:600], flush=True)

    # tiny fully-cached prompt: reproduces the engine's fixed first-token floor
    short_prompt = "Hello, how are you? " * 4
    result["single_short"] = await single_stream(
        args.base_url, args.model, headers, short_prompt, 8, 3)
    print("single_short:", json.dumps(result["single_short"], ensure_ascii=False)[:400], flush=True)

    result["single_cold"] = await cold_ttft(args.base_url, args.model, headers, 64, args.repeats)
    save(args, result)
    print("single_cold:", json.dumps(result["single_cold"], ensure_ascii=False)[:400], flush=True)

    if not args.skip_dual:
        prompt_a = "ALPHA-7f3c9a2e unique workload block. " * 20 + PARAGRAPH * 3
        prompt_b = "BETA-91d4e6b0 unique workload block. " * 20 + PARAGRAPH * 3
        prompts = [prompt_a, prompt_b]
        async with aiohttp.ClientSession(headers=headers) as session:
            warm = [payload_for(args.model, p, 8) for p in prompts]
            await asyncio.gather(*[one_request(session, args.base_url.rstrip("/") + "/v1/completions", p)
                                   for p in warm])
        await asyncio.sleep(2)
        dual = []
        for level in [int(x) for x in args.dual_concurrency.split(",") if x.strip()]:
            row = await dual_level(args.base_url, args.model, headers, prompts,
                                   args.max_tokens, level, max(2, args.repeats - 1))
            dual.append(row)
            print(f"dual c={level}:", json.dumps(row, ensure_ascii=False), flush=True)
        result["dual"] = dual
        save(args, result)

    save(args, result)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
