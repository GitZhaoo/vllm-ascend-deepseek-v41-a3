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
    # Hermetic: ignore config/dsv41*.env so production values never leak in.
    full_env["DSV41_SKIP_CONFIG_FILES"] = "1"
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
        self.assertEqual(cfg["global_segment_size"], "32GB")
        self.assertEqual(cfg["tenant_id"], "default")
        self.assertFalse(cfg["enable_ssd_offload"])
        self.assertTrue(cfg["prefer_alloc_in_same_node"])
        # A3: writes must stay in the writer's own segment (no cross-NPU P2P).
        self.assertTrue(cfg["preferred_segment"])

    def test_preferred_segment_can_be_disabled(self) -> None:
        cfg = self.config(DSV41_MOONCAKE_PREFERRED_SEGMENT="0")
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
        self.assertIn("DSV41_KVPOOL_MAX_MEM_PCT=40", result.stdout)


class CapacityTest(unittest.TestCase):
    """The preflight must catch an over-committed pool before the engine loads.

    A 64GB/卡 × 16 dies pool passes HcommMemReg in isolation but failed on the
    real node once the engine's own host footprint was counted, so the budget
    is checked against MemTotal/MemAvailable up front.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name) / "kvpool"
        self.meminfo = Path(self.tmp.name) / "meminfo"

    def meminfo_with(self, total_gib: int, avail_gib: int) -> str:
        self.meminfo.write_text(
            "MemTotal:       %d kB\n"
            "MemFree:        %d kB\n"
            "MemAvailable:   %d kB\n"
            % (total_gib * 1048576, avail_gib * 1048576 // 2, avail_gib * 1048576)
        )
        return str(self.meminfo)

    def check(self, meminfo: str, **extra: str) -> subprocess.CompletedProcess[str]:
        return run_kvpool(
            "check-capacity",
            env={"DSV41_KVPOOL_STATE_DIR": str(self.state), "DSV41_MEMINFO_FILE": meminfo, **extra},
        )

    def test_default_passes_on_this_node_class(self) -> None:
        result = self.check(self.meminfo_with(2010, 1900))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("512 GiB", result.stderr)

    def test_one_tib_pool_is_rejected(self) -> None:
        result = self.check(self.meminfo_with(2010, 1900), DSV41_MOONCAKE_SEGMENT_SIZE="64GB")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("1024 GiB", result.stderr)
        self.assertIn("DSV41_MOONCAKE_SEGMENT_SIZE", result.stderr)

    def test_small_node_gets_a_smaller_suggestion(self) -> None:
        result = self.check(self.meminfo_with(1024, 1000))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("降到", result.stderr)
        # 40% of 1 TiB / 16 ranks = 25 GiB → rounded down to a multiple of 4.
        self.assertIn("24GB", result.stderr)

    def test_invalid_size_is_rejected(self) -> None:
        result = self.check(self.meminfo_with(2010, 1900), DSV41_MOONCAKE_SEGMENT_SIZE="lots")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DSV41_MOONCAKE_SEGMENT_SIZE", result.stderr)

    def test_ensure_fails_before_touching_the_master(self) -> None:
        result = run_kvpool(
            "ensure",
            env={
                "DSV41_KVPOOL_STATE_DIR": str(self.state),
                "DSV41_MEMINFO_FILE": self.meminfo_with(2010, 1900),
                "DSV41_MOONCAKE_SEGMENT_SIZE": "64GB",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("容量预检未通过", result.stderr)
        self.assertFalse((self.state / "mooncake.json").exists())


class WiringTest(unittest.TestCase):
    """Guard the default-on behaviour and the engine flag wiring."""

    def test_default_is_enabled(self) -> None:
        text = COMMON.read_text()
        self.assertIn(': "${DSV41_KV_POOL:=1}"', text)
        self.assertIn(': "${DSV41_MOONCAKE_SEGMENT_SIZE:=32GB}"', text)

    def test_serve_appends_kv_transfer_config(self) -> None:
        text = SERVE.read_text()
        self.assertIn("AscendStoreConnector", text)
        self.assertIn('ARGS+=(--kv-transfer-config "$KVPOOL_JSON")', text)
        self.assertIn('export PYTHONHASHSEED="$DSV41_PYTHONHASHSEED"', text)
        self.assertIn("MOONCAKE_CONFIG_PATH", text)
        self.assertIn("ASCEND_TRANSFER_TIMEOUT", text)
        self.assertIn("ASCEND_CONNECT_TIMEOUT", text)

    def test_rollback_switch_documented(self) -> None:
        text = COMMON.read_text()
        self.assertIn("DSV41_KV_POOL=0", text)


if __name__ == "__main__":
    unittest.main()
