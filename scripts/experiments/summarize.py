#!/usr/bin/env python3
"""Collect /cache/exp/json/*.json into a markdown comparison table."""
import glob
import json
import os
import sys

EXP = os.environ.get("EXP", "/cache/exp")
ORDER = [
    "exp0_baseline", "exp4_piecewise", "exp2b_fused_mc2", "exp1_static_kernel",
    "exp5_npugraph_ex", "exp6_maxseqs64", "exp7_kvpool_parity", "exp9_bat8192",
    "exp3b_prefill_mc2_bat8192", "exp10_combo", "exp11_tp16",
]


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def row(name, d):
    h = d.get("single_hot") or {}
    s = d.get("single_short") or {}
    c = d.get("single_cold") or {}
    dual = {r.get("concurrency"): r for r in d.get("dual") or []}
    return {
        "cfg": name,
        "decode_tps": h.get("decode_tps"),
        "ttft_ms": h.get("ttft_ms"),
        "eng_prefill_ms": h.get("engine_prefill_ms"),
        "eng_ttft_ms": h.get("engine_ttft_ms"),
        "short_ttft_ms": s.get("ttft_ms"),
        "cold_ttft_ms": c.get("ttft_ms"),
        "accept": h.get("acceptance"),
        "tps": h.get("tokens"),
        "dual2": (dual.get(2) or {}).get("aggregate"),
        "dual16": (dual.get(16) or {}).get("aggregate"),
        "dual64": (dual.get(64) or {}).get("aggregate"),
        "dual16_stream": (dual.get(16) or {}).get("stream"),
        "dual16_ttft": (dual.get(16) or {}).get("ttft_ms"),
    }


def fmt(v, digits=1):
    if v is None:
        return "?"
    if isinstance(v, str):
        return v
    return f"{v:.{digits}f}"


def main():
    files = sorted(glob.glob(os.path.join(EXP, "json", "*.json")))
    rows = []
    for f in files:
        name = os.path.basename(f)[:-5]
        d = load(f)
        if d:
            rows.append(row(name, d))
    rows.sort(key=lambda r: ORDER.index(r["cfg"]) if r["cfg"] in ORDER else 99)

    print("| 配置 | 单流 tok/s | TTFT ms | 引擎prefill ms | 引擎TTFT ms | 短promptTTFT | 冷TTFT | 接受率 | 双前缀 c2 | c16 | c64 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['cfg']} | {fmt(r['decode_tps'])} | {fmt(r['ttft_ms'],0)} | "
              f"{fmt(r['eng_prefill_ms'],0)} | {fmt(r['eng_ttft_ms'],0)} | {fmt(r['short_ttft_ms'],0)} | "
              f"{fmt(r['cold_ttft_ms'],0)} | {fmt(r['accept'],3)} | {fmt(r['dual2'],0)} | "
              f"{fmt(r['dual16'],0)} | {fmt(r['dual64'],0)} |")


if __name__ == "__main__":
    main()
