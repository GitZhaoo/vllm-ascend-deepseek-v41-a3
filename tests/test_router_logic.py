#!/usr/bin/env python3
"""Unit tests for the prefix/KV-aware router (no network required).

Run:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import prefix_router
except ImportError as exc:  # pragma: no cover - optional host dependencies
    raise unittest.SkipTest(f"router dependencies unavailable: {exc}") from exc


class DigestTest(unittest.TestCase):
    def test_block_boundaries(self) -> None:
        tokens = list(range(128 * 3))
        digests = prefix_router.prefix_digests(tokens, 128)
        self.assertEqual(len(digests), 3)
        self.assertEqual(len(set(digests)), 3)
        # Changing a token before the last block must change that block and all
        # following digests, but not earlier ones.
        changed = tokens.copy()
        changed[0] = 999
        changed_digests = prefix_router.prefix_digests(changed, 128)
        self.assertNotEqual(digests[0], changed_digests[0])
        self.assertNotEqual(digests[1], changed_digests[1])

    def test_short_prompt_has_no_full_block(self) -> None:
        self.assertEqual(prefix_router.prefix_digests([1, 2, 3], 128), [])


class RankTest(unittest.TestCase):
    def test_longest_prefix_stops_at_first_miss(self) -> None:
        rank = prefix_router.Rank(0, "http://127.0.0.1:1")
        digests = [b"a", b"b", b"c"]
        rank.digests.update(digests)
        self.assertEqual(rank.matched_blocks(digests), 3)
        self.assertEqual(rank.matched_blocks([b"a", b"x", b"c"]), 1)

    def test_refcount_keeps_digest_until_all_groups_removed(self) -> None:
        rank = prefix_router.Rank(0, "http://127.0.0.1:1")
        digest = b"x" * 16
        # vLLM publishes the same block once per KV-cache group.
        rank.add_digest(digest)
        rank.add_digest(digest)
        rank.remove_digest(digest)
        self.assertIn(digest, rank.digests)
        rank.remove_digest(digest)
        self.assertNotIn(digest, rank.digests)


class RoutingTest(unittest.TestCase):
    def make_router(self, max_skew: int = 8) -> prefix_router.Router:
        args = SimpleNamespace(
            upstream=["http://127.0.0.1:18022", "http://127.0.0.1:18023"],
            block_size=128,
            max_skew=max_skew,
        )
        return prefix_router.Router(args)

    def test_prefers_longest_prefix(self) -> None:
        router = self.make_router()
        self.assertEqual(router._choose_rank(b"key", [3, 1]), 0)

    def test_tie_uses_least_active(self) -> None:
        router = self.make_router()
        router.ranks[0].active = 4
        router.ranks[1].active = 1
        self.assertEqual(router._choose_rank(b"key", [2, 2]), 1)

    def test_skew_breaks_sticky_affinity(self) -> None:
        router = self.make_router(max_skew=2)
        router.ranks[0].active = 10
        router.ranks[1].active = 0
        self.assertEqual(router._choose_rank(b"key", [3, 0]), 1)

    def test_tokenize_payload_excludes_sampling_fields(self) -> None:
        router = self.make_router()
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
            "max_tokens": 32,
            "temperature": 0,
        }
        payload = router._tokenize_payload(body)
        assert payload is not None
        self.assertEqual(payload["messages"], body["messages"])
        self.assertEqual(payload["tools"], body["tools"])
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)


if __name__ == "__main__":
    unittest.main()
