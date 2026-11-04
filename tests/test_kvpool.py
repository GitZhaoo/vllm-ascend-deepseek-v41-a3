"""Tests for the optional Mooncake KV pool (AscendStoreConnector) wiring.

The pool is enabled by default (DSV41_KV_POOL=1): serve.sh asks
scripts/kvpool.sh to write mooncake.json, starts mooncake_master if needed and
passes --kv-transfer-config to vLLM. These tests stay hermetic — they only
exercise config rendering and the static wiring, never a live master.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KVPOOL = ROOT / "scripts" / "kvpool.sh"
SERVE = ROOT / "scripts" / "serve.sh"
COMMON = ROOT / "scripts" / "common.sh"


def run_kvpool(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("DSV41_")  # keep tests independent of local.env
    }
    full_env.update(env or {})
    return subprocess.run(
        ["bash", str(KVPOOL), *args],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=ROOT,
        timeout=60,
    )


class RenderConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name) / "kvpool"

    def config(self, **extra: str) -> dict:
        env = {"DSV41_KVPOOL_STATE_DIR": str(self.state), **extra}
        result = run_kvpool("render-config", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        path = Path(result.stdout.strip())
        self.assertTrue(path.is_file())
        return json.loads(path.read_text())

    def test_defaults_match_reference_deployment(self) -> None:
        cfg = self.config()
        self.assertEqual(cfg["metadata_server"], "P2PHANDSHAKE")
        self.assertEqual(cfg["protocol"], "ascend")
        self.assertEqual(cfg["device_name"], "")
        self.assertEqual(cfg["master_server_address"], "127.0.0.1:50088")
        self.assertEqual(cfg["global_segment_size"], "64GB")
        self.assertEqual(cfg["tenant_id"], "default")
        self.assertFalse(cfg["enable_ssd_offload"])
        self.assertTrue(cfg["prefer_alloc_in_same_node"])
        self.assertFalse(cfg["preferred_segment"])

    def test_overrides_are_honoured(self) -> None:
        cfg = self.config(
            DSV41_MOONCAKE_MASTER="10.0.0.5:50089",
            DSV41_MOONCAKE_SEGMENT_SIZE="16GB",
            DSV41_MOONCAKE_TENANT="team-a",
        )
        self.assertEqual(cfg["master_server_address"], "10.0.0.5:50089")
        self.assertEqual(cfg["global_segment_size"], "16GB")
        self.assertEqual(cfg["tenant_id"], "team-a")

    def test_ssd_offload_requires_path(self) -> None:
        result = run_kvpool(
            "render-config",
            env={
                "DSV41_KVPOOL_STATE_DIR": str(self.state),
                "DSV41_MOONCAKE_SSD_OFFLOAD": "1",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DSV41_MOONCAKE_SSD_PATH", result.stderr)

    def test_ssd_offload_path_is_rendered(self) -> None:
        cfg = self.config(
            DSV41_MOONCAKE_SSD_OFFLOAD="1",
            DSV41_MOONCAKE_SSD_PATH="/nvme/mooncake_offload",
        )
        self.assertTrue(cfg["enable_ssd_offload"])
        self.assertEqual(cfg["ssd_offload_path"], "/nvme/mooncake_offload")

    def test_print_env_reports_effective_values(self) -> None:
        result = run_kvpool(
            "print-env",
            env={
                "DSV41_KVPOOL_STATE_DIR": str(self.state),
                "DSV41_MOONCAKE_SEGMENT_SIZE": "16GB",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DSV41_KV_POOL=1", result.stdout)
        self.assertIn("DSV41_MOONCAKE_SEGMENT_SIZE=16GB", result.stdout)


class WiringTest(unittest.TestCase):
    """Guard the default-on behaviour and the engine flag wiring."""

    def test_default_is_enabled(self) -> None:
        text = COMMON.read_text()
        self.assertIn(': "${DSV41_KV_POOL:=1}"', text)
        self.assertIn(': "${DSV41_MOONCAKE_SEGMENT_SIZE:=64GB}"', text)

    def test_serve_appends_kv_transfer_config(self) -> None:
        text = SERVE.read_text()
        self.assertIn("AscendStoreConnector", text)
        self.assertIn('ARGS+=(--kv-transfer-config "$KVPOOL_JSON")', text)
        self.assertIn('export PYTHONHASHSEED="$DSV41_PYTHONHASHSEED"', text)
        self.assertIn("MOONCAKE_CONFIG_PATH", text)

    def test_rollback_switch_documented(self) -> None:
        text = COMMON.read_text()
        self.assertIn("DSV41_KV_POOL=0", text)


if __name__ == "__main__":
    unittest.main()
