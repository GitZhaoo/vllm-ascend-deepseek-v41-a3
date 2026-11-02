#!/usr/bin/env python3
"""Unit tests for scripts/patch_utils.py and scripts/oci_extract.py.

Run:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import os
import pathlib
import sys
import tarfile
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import oci_extract  # noqa: E402
import patch_utils  # noqa: E402


class PatchUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.patch = self.root / "test.patch"
        self.patch.write_text(
            "diff --git a/mod.py b/mod.py\n"
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,1 +1,2 @@\n"
            "+_GATE = os.environ.get(\"VLLM_ASCEND_V41_TEST\", \"0\") == \"1\"\n"
            "+_HELPER = \"distinctive_patch_marker_line\"\n",
            encoding="utf-8",
        )
        self.target = self.root / "mod.py"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_not_applied(self) -> None:
        self.target.write_text("pass\n", encoding="utf-8")
        self.assertEqual(patch_utils.content_state(str(self.root), str(self.patch)), "not_applied")

    def test_applied_with_pinned_gate_default(self) -> None:
        # The apply script rewrites the default from 0 to 1 after patching.
        self.target.write_text(
            "_GATE = os.environ.get(\"VLLM_ASCEND_V41_TEST\", \"1\") == \"1\"\n"
            "_HELPER = \"distinctive_patch_marker_line\"\n",
            encoding="utf-8",
        )
        self.assertEqual(patch_utils.content_state(str(self.root), str(self.patch)), "applied")

    def test_partial(self) -> None:
        multi = self.root / "multi.patch"
        multi.write_text(
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n"
            "+MARKER_A = \"unique marker alpha\"\n"
            "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1,2 @@\n"
            "+MARKER_B = \"unique marker beta\"\n",
            encoding="utf-8",
        )
        (self.root / "a.py").write_text("MARKER_A = \"unique marker alpha\"\n", encoding="utf-8")
        (self.root / "b.py").write_text("pass\n", encoding="utf-8")
        self.assertEqual(patch_utils.content_state(str(self.root), str(multi)), "partial")


class OciExtractTest(unittest.TestCase):
    def _layer(self, path: pathlib.Path, entries: list[tuple[str, str | None, str]]) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, data, kind in entries:
                info = tarfile.TarInfo(name)
                if kind == "sym":
                    info.type = tarfile.SYMTYPE
                    info.linkname = data or ""
                    tf.addfile(info)
                elif kind == "whiteout":
                    tf.addfile(info)
                else:
                    payload = (data or "").encode()
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))
        path.write_bytes(buf.getvalue())

    def test_whiteout_symlink_and_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp, "root")
            root.mkdir()
            (root / "old").write_text("old", encoding="utf-8")
            (root / "keep").write_text("keep", encoding="utf-8")
            layer = pathlib.Path(tmp, "layer.tar.gz")
            self._layer(layer, [
                (".wh.old", None, "whiteout"),
                ("new", "new", "file"),
                ("link-rel", "new", "sym"),
                ("link-abs", "/usr/local/target", "sym"),
                ("../escape", "bad", "file"),
            ])
            files, whiteouts = oci_extract.extract_layer(str(layer), str(root))
            self.assertFalse((root / "old").exists())
            self.assertEqual((root / "new").read_text(encoding="utf-8"), "new")
            self.assertEqual((root / "keep").read_text(encoding="utf-8"), "keep")
            self.assertEqual(os.readlink(root / "link-rel"), "new")
            self.assertEqual(os.readlink(root / "link-abs"), "/usr/local/target")
            self.assertFalse((pathlib.Path(tmp) / "escape").exists())
            self.assertGreaterEqual(files, 3)
            self.assertEqual(whiteouts, 1)


if __name__ == "__main__":
    unittest.main()
