#!/usr/bin/env python3
"""Regression tests for scripts/apply_patches.sh content fingerprinting.

These exercise the *fingerprint* behaviour added in 0.2.2: a patch whose
content changed after it was applied must be reported as "[补丁已变更]" and the
next `apply` must refuse, instead of trusting a stale marker. The script is run
against a throwaway git tree via SRC= / PATCH_DIR= (single-component mode), so
no real deployment is touched.

Run:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APPLY = os.path.join(ROOT, "scripts", "apply_patches.sh")

PATCH_A = (
    "diff --git a/a.py b/a.py\n"
    "--- a/a.py\n"
    "+++ b/a.py\n"
    "@@ -1 +1,2 @@\n"
    " line1\n"
    "+DISTINCTIVE_FINGERPRINT_LINE_ALPHA\n"
)


def run(*args, src, patch_dir, **env):
    full = dict(os.environ)
    full.update(env)
    full["DSV41_SKIP_CONFIG_FILES"] = "1"
    full["DSV41_RUNTIME"] = "native"
    full["SRC"] = src
    full["PATCH_DIR"] = patch_dir
    return subprocess.run(
        ["bash", APPLY, *args],
        env=full,
        capture_output=True,
        text=True,
    )


class ApplyFingerprintTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        self.src = os.path.join(self.tmp, "src")
        os.makedirs(self.src)
        subprocess.run(["git", "init", "-q", self.src], check=True)
        subprocess.run(
            ["git", "-C", self.src, "config", "user.email", "t@t"], check=True
        )
        subprocess.run(
            ["git", "-C", self.src, "config", "user.name", "t"], check=True
        )
        with open(os.path.join(self.src, "a.py"), "w", encoding="utf-8") as fh:
            fh.write("line1\n")
        subprocess.run(["git", "-C", self.src, "add", "a.py"], check=True)
        subprocess.run(
            ["git", "-C", self.src, "commit", "-qm", "init"], check=True
        )

        self.patches = os.path.join(self.tmp, "patches")
        os.makedirs(self.patches)
        with open(os.path.join(self.patches, "0001-test.patch"), "w", encoding="utf-8") as fh:
            fh.write(PATCH_A)
        with open(os.path.join(self.patches, "series.conf"), "w", encoding="utf-8") as fh:
            fh.write("0001-test.patch\n")

    def _marker(self, name="0001-test.patch"):
        return os.path.join(self.src, ".patches-applied.d", f"{name}.done")

    def test_apply_then_status_is_idempotent(self) -> None:
        r = run("apply", src=self.src, patch_dir=self.patches)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(self._marker()), r.stdout)

        # marker stores the patch sha256
        with open(self._marker(), encoding="utf-8") as fh:
            self.assertIn("sha256=", fh.read())

        r2 = run("apply", src=self.src, patch_dir=self.patches)
        self.assertEqual(r2.returncode, 0, r2.stderr)

    def test_changed_patch_is_reported_and_rejected(self) -> None:
        run("apply", src=self.src, patch_dir=self.patches)
        # mutate the patch content on disk
        with open(os.path.join(self.patches, "0001-test.patch"), "a", encoding="utf-8") as fh:
            fh.write("# trailing noise makes the sha differ\n")

        r = run("status", src=self.src, patch_dir=self.patches)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("补丁已变更", r.stdout)

        r2 = run("apply", src=self.src, patch_dir=self.patches)
        self.assertEqual(r2.returncode, 1, r2.stdout + r2.stderr)
        self.assertIn("补丁内容已变化", r2.stderr)

    def test_series_fingerprint_changes_when_patch_changes(self) -> None:
        run("apply", src=self.src, patch_dir=self.patches)
        with open(os.path.join(self.patches, "0001-test.patch"), "a", encoding="utf-8") as fh:
            fh.write("# noise\n")
        r = run("status", src=self.src, patch_dir=self.patches)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("序列已变化", r.stdout)

    def test_revert_cleans_markers_and_tree(self) -> None:
        run("apply", src=self.src, patch_dir=self.patches)
        r = run("revert", "--force", src=self.src, patch_dir=self.patches)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.src, ".patches-applied.d")))
        with open(os.path.join(self.src, "a.py"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "line1\n")


if __name__ == "__main__":
    unittest.main()
