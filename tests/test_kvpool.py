"""Tests for the optional external KV pool (AscendStoreConnector + MemCache).

This mirrors the reference deployment CJYLZS/vllm-ascend-deepseek-v41-a2:
serve.sh asks scripts/kvpool.sh to render mmc-local.conf/mmc-meta.conf, starts
(or reuses) a CPU-only memcache_hybrid MetaService, then passes
--kv-transfer-config with backend=memcache and MMC_LOCAL_CONFIG_PATH.

The tests are hermetic: they exercise config rendering, the 0101 correctness
gate and static wiring, but never start a live MetaService.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KVPOOL = ROOT / "scripts" / "kvpool.sh"
SERVE = ROOT / "scripts" / "serve.sh"
COMMON = ROOT / "scripts" / "common.sh"
DOC = ROOT / "docs" / "kv-pool.md"


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


def parse_conf(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


class RenderConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name) / "kvpool"

    def config(self, **extra: str) -> tuple[dict[str, str], dict[str, str]]:
        env = {"DSV41_KVPOOL_STATE_DIR": str(self.state), **extra}
        result = run_kvpool("render-config", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        path = Path(result.stdout.strip())
        self.assertTrue(path.is_file())
        self.assertEqual(path, self.state / "mmc-local.conf")
        return parse_conf(path), parse_conf(self.state / "mmc-meta.conf")

    def test_defaults_match_reference_deployment(self) -> None:
        local, meta = self.config()
        self.assertEqual(local["ock.mmc.meta_service_url"], "tcp://127.0.0.1:5100")
        self.assertEqual(local["ock.mmc.local_service.config_store_url"], "tcp://127.0.0.1:6100")
        self.assertEqual(local["ock.mmc.local_service.world_size"], "256")
        self.assertEqual(local["ock.mmc.local_service.protocol"], "host_shm")
        self.assertEqual(local["ock.mmc.local_service.dram.size"], "2GB")
        self.assertEqual(meta["ock.mmc.meta_service.metrics_url"], "http://127.0.0.1:8100")
        self.assertEqual(meta["ock.mmc.meta_service_url"], "tcp://127.0.0.1:5100")

    def test_overrides_are_honoured(self) -> None:
        local, meta = self.config(
            DSV41_KV_DRAM_GB="8",
            DSV41_KV_META_PORT="15100",
            DSV41_KV_STORE_PORT="16100",
            DSV41_KV_METRICS_PORT="18100",
            DSV41_KV_WORLD_SIZE="32",
        )
        self.assertEqual(local["ock.mmc.local_service.dram.size"], "8GB")
        self.assertEqual(local["ock.mmc.local_service.world_size"], "32")
        self.assertEqual(local["ock.mmc.meta_service_url"], "tcp://127.0.0.1:15100")
        self.assertEqual(meta["ock.mmc.meta_service_url"], "tcp://127.0.0.1:15100")
        self.assertEqual(meta["ock.mmc.meta_service.metrics_url"], "http://127.0.0.1:18100")

    def test_invalid_dram_is_rejected(self) -> None:
        result = run_kvpool(
            "render-config",
            env={"DSV41_KVPOOL_STATE_DIR": str(self.state), "DSV41_KV_DRAM_GB": "lots"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DSV41_KV_DRAM_GB", result.stderr)

    def test_unknown_protocol_is_rejected(self) -> None:
        result = run_kvpool(
            "render-config",
            env={"DSV41_KVPOOL_STATE_DIR": str(self.state), "DSV41_KV_PROTOCOL": "tcp"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DSV41_KV_PROTOCOL", result.stderr)

    def test_port_collision_with_vllm_is_rejected(self) -> None:
        result = run_kvpool(
            "render-config",
            env={
                "DSV41_KVPOOL_STATE_DIR": str(self.state),
                "DSV41_PORT": "8100",
                "DSV41_KV_METRICS_PORT": "8100",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("与 vLLM --port 相同", result.stderr)

    def test_print_env_reports_effective_values(self) -> None:
        result = run_kvpool(
            "print-env",
            env={
                "DSV41_KVPOOL_STATE_DIR": str(self.state),
                "DSV41_KV_DRAM_GB": "8",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DSV41_KV_POOL=0", result.stdout)
        self.assertIn("DSV41_KV_DRAM_GB=8", result.stdout)
        self.assertIn("DSV41_KV_PROTOCOL=host_shm", result.stdout)
        self.assertIn("DSV41_KVPOOL_MAX_MEM_PCT=25", result.stdout)



class CapacityTest(unittest.TestCase):
    """The preflight catches an over-committed pool before the engine loads."""

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
        self.assertIn("32.0 GiB", result.stderr)

    def test_one_tib_pool_is_rejected(self) -> None:
        result = self.check(self.meminfo_with(2010, 1900), DSV41_KV_DRAM_GB="64")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("1024.0 GiB", result.stderr)
        self.assertIn("DSV41_KV_DRAM_GB", result.stderr)

    def test_small_node_gets_a_smaller_suggestion(self) -> None:
        result = self.check(self.meminfo_with(1024, 1000), DSV41_KV_DRAM_GB="32")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("降到", result.stderr)
        # 25% of 1 TiB / 16 ranks = 16 GiB.
        self.assertIn("16GB", result.stderr)

    def test_invalid_size_is_rejected(self) -> None:
        result = self.check(self.meminfo_with(2010, 1900), DSV41_KV_DRAM_GB="lots")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DSV41_KV_DRAM_GB", result.stderr)

    def test_ensure_fails_before_touching_meta(self) -> None:
        result = run_kvpool(
            "ensure",
            env={
                "DSV41_KV_POOL": "1",
                "DSV41_KVPOOL_STATE_DIR": str(self.state),
                "DSV41_KV_META_LOG": str(Path(self.tmp.name) / "meta.log"),
                "DSV41_KV_META_PID_FILE": str(Path(self.tmp.name) / "meta.pid"),
                "DSV41_MEMINFO_FILE": self.meminfo_with(2010, 1900),
                "DSV41_KV_DRAM_GB": "64",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("容量校验失败", result.stderr)
        self.assertFalse((self.state / "mmc-local.conf").exists())
        self.assertFalse((Path(self.tmp.name) / "meta.pid").exists())


class TreeFixTest(unittest.TestCase):
    def test_missing_0101_refuses_to_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_kvpool(
                "start",
                env={
                    "DSV41_KV_POOL": "1",
                    "DSV41_ROOTFS": tmp,
                    "DSV41_KV_META_LOG": str(Path(tmp) / "meta.log"),
                    "DSV41_KV_META_PID_FILE": str(Path(tmp) / "meta.pid"),
                },
            )
        self.assertEqual(result.returncode, 8, result.stderr)
        self.assertIn("0101", result.stderr)
        self.assertIn("MRV2", result.stderr)

    def test_allow_unfixed_tree_is_an_explicit_escape_hatch(self) -> None:
        text = KVPOOL.read_text()
        self.assertIn("DSV41_KV_ALLOW_UNFIXED_TREE", text)
        self.assertIn("_preempted_req_ids.discard(request.req_id)", text)


class WiringTest(unittest.TestCase):
    def test_default_is_opt_in(self) -> None:
        text = COMMON.read_text()
        self.assertIn(': "${DSV41_KV_POOL:=0}"', text)
        self.assertIn(': "${DSV41_KV_DRAM_GB:=2}"', text)
        self.assertIn(': "${DSV41_KV_PROTOCOL:=host_shm}"', text)

    def test_serve_uses_memcache_not_mooncake(self) -> None:
        text = SERVE.read_text()
        self.assertIn("AscendStoreConnector", text)
        self.assertIn('"backend":"memcache"', text)
        self.assertIn("MMC_LOCAL_CONFIG_PATH", text)
        self.assertIn('ARGS+=(--kv-transfer-config "$KVPOOL_JSON")', text)
        # Regression: the old Mooncake/HIXL wiring must not come back.
        self.assertNotIn("MOONCAKE_CONFIG_PATH", text)
        self.assertNotIn("ASCEND_TRANSFER_TIMEOUT", text)
        self.assertNotIn("PYTHONHASHSEED", text)

    def test_rollback_switch_is_documented(self) -> None:
        text = COMMON.read_text()
        self.assertIn(': "${DSV41_KV_POOL:=0}"', text)

    def test_docs_describe_the_reference_design(self) -> None:
        text = DOC.read_text()
        self.assertIn("AscendStoreConnector", text)
        self.assertIn("MemCache", text)
        self.assertIn("host_shm", text)
        self.assertIn("0101", text)
        self.assertNotIn("Mooncake", text)


if __name__ == "__main__":
    unittest.main()
