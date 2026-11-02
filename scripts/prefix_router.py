#!/usr/bin/env python3
"""Prefix/KV-aware router for an externally load-balanced vLLM DP deployment.

Architecture
------------
    client -> this router (:8021) -> vLLM API rank0 (:8022)
                                   -> vLLM API rank1 (:8023)

Each vLLM rank publishes KV cache events over ZMQ. The router tracks the
sequence of prefix blocks cached by each rank and sends a request to the rank
that already holds the longest matching token prefix. When no rank has a
matching prefix, the request goes to the least-loaded rank and the assignment
is remembered as a sticky affinity.

The router is OpenAI-compatible: it transparently proxies chat/completions,
completions, models, metrics and streaming SSE responses. It does not require
duplicated model output logic.

Dependencies (host Python):
    aiohttp, pyzmq, msgspec

Usage:
    python3 scripts/prefix_router.py \
        --listen 8021 \
        --upstream http://127.0.0.1:8022 \
        --upstream http://127.0.0.1:8023 \
        --kv-events tcp://127.0.0.1:5557 \
        --kv-events tcp://127.0.0.1:5558 \
        --model DeepSeek-V4.1-Flash-1017
"""

from __future__ import annotations

import argparse
import array
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import msgspec
import zmq
import zmq.asyncio
from aiohttp import web

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}
COMPLETION_PATHS = {"/v1/chat/completions", "/v1/completions"}
BLOCK_SIZE = 128
MAX_DIGESTS = 200_000
MAX_STICKY = 50_000
MAX_TOKEN_CACHE_ENTRIES = 1_024
MAX_TOKEN_CACHE_TOKENS = 2_000_000
MAX_CACHED_PROMPT_TOKENS = 16_384

logger = logging.getLogger("prefix_router")


def block_digest(previous: bytes, tokens: list[int]) -> bytes:
    payload = array.array("I", tokens).tobytes()
    return hashlib.blake2b(previous + payload, digest_size=16).digest()


def prefix_digests(tokens: list[int], block_size: int = BLOCK_SIZE) -> list[bytes]:
    out: list[bytes] = []
    previous = b""
    blocks = len(tokens) // block_size
    for i in range(blocks):
        previous = block_digest(previous, tokens[i * block_size:(i + 1) * block_size])
        out.append(previous)
    return out


@dataclass
class Rank:
    index: int
    url: str
    active: int = 0
    healthy: bool = False  # set by the first /health probe
    digests: set[bytes] = field(default_factory=set)
    digest_order: deque[bytes] = field(default_factory=deque)
    digest_counts: dict[bytes, int] = field(default_factory=dict)
    hash_to_digest: dict[str, bytes] = field(default_factory=dict)

    def add_digest(self, digest: bytes) -> None:
        # The same block hash is published once per KV-cache group. Count
        # references so BlockRemoved for one group does not hide the prefix
        # while other groups still hold it.
        self.digest_counts[digest] = self.digest_counts.get(digest, 0) + 1
        if digest in self.digests:
            return
        if len(self.digest_order) >= MAX_DIGESTS:
            self.reset()
        self.digests.add(digest)
        self.digest_order.append(digest)

    def remove_digest(self, digest: bytes) -> None:
        count = self.digest_counts.get(digest, 0) - 1
        if count > 0:
            self.digest_counts[digest] = count
            return
        self.digest_counts.pop(digest, None)
        self.digests.discard(digest)

    def reset(self) -> None:
        self.digests.clear()
        self.digest_order.clear()
        self.digest_counts.clear()
        self.hash_to_digest.clear()

    def matched_blocks(self, digests: list[bytes]) -> int:
        matched = 0
        for digest in digests:
            if digest not in self.digests:
                break
            matched += 1
        return matched


class Router:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.block_size = args.block_size
        self.ranks = [Rank(i, url.rstrip("/")) for i, url in enumerate(args.upstream)]
        if len(self.ranks) < 2:
            raise SystemExit("at least two --upstream endpoints are required")
        self.session: aiohttp.ClientSession | None = None
        self.zmq_ctx: zmq.asyncio.Context | None = None
        self.sticky: OrderedDict[bytes, tuple[int, float]] = OrderedDict()
        self.token_cache: OrderedDict[bytes, list[int]] = OrderedDict()
        self.token_cache_tokens = 0
        self.stats = {"requests": 0, "affinity_hits": 0, "fallback": 0, "events": 0}

    async def start(self) -> None:
        connector = aiohttp.TCPConnector(
            limit=256, limit_per_host=128, keepalive_timeout=120,
            enable_cleanup_closed=True, ttl_dns_cache=300,
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None, sock_read=3600),
            trust_env=False,
            auto_decompress=False,
        )
        self.zmq_ctx = zmq.asyncio.Context.instance()
        for rank, endpoint in zip(self.ranks, self.args.kv_events or []):
            asyncio.create_task(self._kv_event_loop(rank, endpoint), name=f"kv-events-{rank.index}")
        asyncio.create_task(self._health_loop(), name="health")

    async def close(self) -> None:
        if self.session:
            await self.session.close()

    # ---- KV cache events -------------------------------------------------
    async def _kv_event_loop(self, rank: Rank, endpoint: str) -> None:
        assert self.zmq_ctx is not None
        socket = self.zmq_ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.RCVHWM, 100_000)
        socket.connect(endpoint)
        logger.info("rank %d subscribing KV events at %s", rank.index, endpoint)
        try:
            while True:
                frames = await socket.recv_multipart()
                if len(frames) < 3:
                    continue
                try:
                    batch = msgspec.msgpack.decode(frames[-1])
                except Exception as exc:  # noqa: BLE001
                    logger.debug("bad KV event payload: %s", exc)
                    continue
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                for event in batch[1] or []:
                    self._handle_event(rank, event)
                    self.stats["events"] += 1
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("KV event loop for rank %d stopped: %s", rank.index, exc)
        finally:
            socket.close(linger=0)

    def _handle_event(self, rank: Rank, event: Any) -> None:
        if isinstance(event, dict):
            kind = event.get("type")
        else:
            return
        if kind == "AllBlocksCleared":
            rank.reset()
            return
        if kind == "BlockRemoved":
            for block_hash in event.get("block_hashes") or []:
                digest = rank.hash_to_digest.get(block_hash)
                if digest is None:
                    continue
                rank.remove_digest(digest)
                # Keep the hash -> digest mapping until every cache group has
                # removed it; otherwise the remaining group events cannot
                # decrement the reference count.
                if rank.digest_counts.get(digest, 0) <= 0:
                    rank.hash_to_digest.pop(block_hash, None)
            return
        if kind != "BlockStored":
            return

        hashes = event.get("block_hashes") or []
        tokens = event.get("token_ids") or []
        block_size = int(event.get("block_size") or self.block_size)
        parent = event.get("parent_block_hash")
        previous = rank.hash_to_digest.get(parent, b"") if parent else b""
        if not hashes:
            return
        chunks = [tokens[i:i + block_size] for i in range(0, len(tokens), block_size)]
        # Null/masked blocks can make lengths differ; pair what we can safely.
        pair_count = min(len(hashes), len(chunks))
        for block_hash, chunk in zip(hashes[:pair_count], chunks[:pair_count]):
            if len(chunk) != block_size:
                break
            previous = block_digest(previous, chunk)
            rank.hash_to_digest[block_hash] = previous
            rank.add_digest(previous)
        # Bound the hash table as well as the digest set.
        if len(rank.hash_to_digest) > MAX_DIGESTS * 2:
            rank.reset()

    # ---- health ----------------------------------------------------------
    async def _health_loop(self) -> None:
        while True:
            await asyncio.gather(*(self._check_health(rank) for rank in self.ranks))
            await asyncio.sleep(self.args.health_interval)

    async def _check_health(self, rank: Rank) -> None:
        assert self.session is not None
        try:
            async with self.session.get(rank.url + "/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                rank.healthy = resp.status == 200
        except Exception:  # noqa: BLE001
            rank.healthy = False

    # ---- tokenization / routing -----------------------------------------
    def _tokenize_payload(self, body: dict[str, Any]) -> dict[str, Any] | None:
        payload: dict[str, Any] = {}
        if body.get("model"):
            payload["model"] = body["model"]
        if "messages" in body:
            payload["messages"] = body["messages"]
            for key in ("tools", "chat_template", "chat_template_kwargs",
                        "continue_final_message", "add_generation_prompt"):
                if key in body:
                    payload[key] = body[key]
        elif "prompt" in body:
            payload["prompt"] = body["prompt"]
            if "add_special_tokens" in body:
                payload["add_special_tokens"] = body["add_special_tokens"]
        else:
            return None
        return payload

    @staticmethod
    def _body_key(payload: dict[str, Any]) -> bytes:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")).encode()
        return hashlib.blake2b(canonical, digest_size=16).digest()

    async def _tokenize(self, payload: dict[str, Any], headers: dict[str, str]) -> list[int] | None:
        key = self._body_key(payload)
        cached = self.token_cache.get(key)
        if cached is not None:
            self.token_cache.move_to_end(key)
            return cached
        assert self.session is not None
        url = self.ranks[0].url + "/tokenize"
        try:
            async with self.session.post(url, json=payload, headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
        except Exception:  # noqa: BLE001
            return None
        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            return None
        # Never cache huge prompts: a Python list of 1M ints costs ~36 MB and
        # would quickly exhaust the router process.
        if len(tokens) <= MAX_CACHED_PROMPT_TOKENS:
            old = self.token_cache.pop(key, None)
            if old is not None:
                self.token_cache_tokens -= len(old)
            self.token_cache[key] = tokens
            self.token_cache_tokens += len(tokens)
            while (len(self.token_cache) > MAX_TOKEN_CACHE_ENTRIES
                   or self.token_cache_tokens > MAX_TOKEN_CACHE_TOKENS):
                _, old = self.token_cache.popitem(last=False)
                self.token_cache_tokens -= len(old)
        return tokens

    def _choose_rank(self, body_key: bytes, matched: list[int]) -> int:
        best = max(matched) if matched else 0
        candidates = [r for r in self.ranks if r.healthy] or self.ranks
        if best > 0:
            ranked = [r.index for r in candidates if matched[r.index] == best]
            choice = min(ranked, key=lambda i: self.ranks[i].active)
            self.stats["affinity_hits"] += 1
        else:
            sticky = self.sticky.get(body_key)
            if sticky and self.ranks[sticky[0]].healthy:
                choice = sticky[0]
            else:
                choice = min(candidates, key=lambda r: r.active).index
                self.stats["fallback"] += 1
        # Avoid dumping a hot prefix onto one rank forever.
        other = min((r for r in candidates if r.index != choice),
                    key=lambda r: r.active, default=None)
        if other is not None and self.ranks[choice].active > other.active + self.args.max_skew:
            choice = other.index
            self.stats["fallback"] += 1
        self.sticky[body_key] = (choice, time.monotonic())
        self.sticky.move_to_end(body_key)
        while len(self.sticky) > MAX_STICKY:
            self.sticky.popitem(last=False)
        return choice

    async def route(self, request: web.Request, body: bytes) -> int:
        try:
            data = json.loads(body)
        except Exception:  # noqa: BLE001
            return 0
        payload = self._tokenize_payload(data)
        if payload is None:
            return 0
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        tokens = await self._tokenize(payload, headers)
        matched = [0] * len(self.ranks)
        if tokens:
            digests = prefix_digests(tokens, self.block_size)
            matched = [rank.matched_blocks(digests) for rank in self.ranks]
        return self._choose_rank(self._body_key(payload), matched)

    # ---- proxies ---------------------------------------------------------
    async def proxy(self, request: web.Request, forced_rank: int | None = None) -> web.StreamResponse:
        body = await request.read()
        rank_index = forced_rank
        if rank_index is None and request.path in COMPLETION_PATHS and request.method == "POST":
            rank_index = await self.route(request, body)
        if rank_index is None:
            rank_index = 0
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP]
        order = [rank_index] + [r.index for r in self.ranks if r.index != rank_index]
        last_error: Exception | None = None
        for idx in order:
            rank = self.ranks[idx]
            if not rank.healthy and len(order) > 1:
                continue
            rank.active += 1
            self.stats["requests"] += 1
            try:
                return await self._forward(request, body, rank, headers)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                logger.warning("upstream rank %d failed: %s", idx, exc)
            finally:
                rank.active -= 1
        return web.Response(status=502, text=f"all upstreams failed: {last_error}")

    async def _forward(self, request: web.Request, body: bytes, rank: Rank,
                       headers: list[tuple[str, str]]) -> web.StreamResponse:
        assert self.session is not None
        url = rank.url + request.rel_url.raw_path_qs
        upstream = await self.session.request(
            request.method, url, headers=headers, data=body if body else None,
        )
        out = web.StreamResponse(status=upstream.status, reason=upstream.reason)
        for key, value in upstream.headers.items():
            if key.lower() not in HOP_BY_HOP:
                out.headers[key] = value
        if "text/event-stream" in upstream.headers.get("Content-Type", ""):
            out.headers.setdefault("Cache-Control", "no-cache")
        await out.prepare(request)
        try:
            async for chunk in upstream.content.iter_any():
                await out.write(chunk)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            upstream.close()
        try:
            await out.write_eof()
        except Exception:  # noqa: BLE001
            pass
        return out

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.path == "/health":
            healthy = all(rank.healthy for rank in self.ranks)
            return web.Response(status=200 if healthy else 503,
                                text="ok" if healthy else "not ready")
        if request.path == "/router/stats":
            if self.args.stats_token:
                auth = request.headers.get("Authorization", "")
                if auth != f"Bearer {self.args.stats_token}":
                    return web.Response(status=401, text="unauthorized")
            return web.json_response({
                "ranks": [{"url": r.url, "active": r.active, "healthy": r.healthy,
                           "cached_prefix_digests": len(r.digests)} for r in self.ranks],
                **self.stats,
            })
        if request.path == "/metrics":
            return await self._aggregate_metrics()
        return await self.proxy(request)

    async def _aggregate_metrics(self) -> web.Response:
        assert self.session is not None
        texts: list[str] = []
        for rank in self.ranks:
            try:
                async with self.session.get(rank.url + "/metrics",
                                            timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        texts.append(await resp.text())
            except Exception:  # noqa: BLE001
                pass
        if not texts:
            return web.Response(status=503, text="metrics unavailable")
        seen_meta: set[str] = set()
        lines: list[str] = []
        for text in texts:
            for line in text.splitlines():
                if line.startswith("# HELP") or line.startswith("# TYPE"):
                    if line in seen_meta:
                        continue
                    seen_meta.add(line)
                lines.append(line)
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


def build_app(router: Router) -> web.Application:
    async def startup(app: web.Application) -> None:
        await router.start()

    async def cleanup(app: web.Application) -> None:
        await router.close()

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    app.router.add_route("*", "/{tail:.*}", router.handle)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", type=int, default=8021)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--upstream", action="append", required=True,
                        help="vLLM rank endpoint, repeat for each rank")
    parser.add_argument("--kv-events", action="append", default=[],
                        help="ZMQ KV event endpoint per rank, same order as --upstream")
    parser.add_argument("--model", default=os.environ.get("DSV41_SERVED_NAME", ""))
    parser.add_argument("--max-skew", type=int, default=8,
                        help="break prefix affinity when one rank has this many more active requests")
    parser.add_argument("--block-size", type=int, default=128,
                        help="KV cache block size; must match vLLM --block-size")
    parser.add_argument("--stats-token", default=None,
                        help="require this bearer token for /router/stats "
                             "(default: $DSV41_ROUTER_STATS_TOKEN or $DSV41_API_KEY)")
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    args.stats_token = (args.stats_token
                        or os.environ.get("DSV41_ROUTER_STATS_TOKEN")
                        or os.environ.get("DSV41_API_KEY"))

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    logger.info("router listening on %s:%d, upstreams=%s, kv_events=%s",
                args.host, args.listen, args.upstream, args.kv_events)
    router = Router(args)
    web.run_app(build_app(router), host=args.host, port=args.listen, access_log=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
