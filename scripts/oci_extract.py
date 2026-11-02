#!/usr/bin/env python3
"""Extract an OCI/Docker image from a registry into a root filesystem.

Purpose
-------
The Ascend container images are published as OCI images, but this project runs
the rootfs natively (no docker / no container runtime). A naive ``tar -x`` is
not sufficient: OCI layers use whiteout markers (``.wh.<name>`` and
``.wh..wh..opq``) to express deletions. Ignoring them leaves stale files from
lower layers behind, which is how a broken ``torch`` tree causes
``ImportError: cannot import name 'is_fake_tensor'``.

This script:
  * resolves a multi-arch manifest and selects linux/arm64 by default,
  * downloads each blob with retry + sha256 verification,
  * applies layers in order while honouring whiteouts,
  * records completed layers so a interrupted run can resume,
  * writes provenance metadata to ``<target>/.dsv41-image``.

Usage:
    python3 scripts/oci_extract.py \
        --registry quay.nju.edu.cn --repo ascend/vllm-ascend \
        --tag v0.30.0rc2-a3 --target /cache/dsv41-rc2-rootfs

Requires only the Python 3 standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

INDEX_TYPES = [
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
]
ACCEPT = ", ".join(INDEX_TYPES)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never use env proxies
TOKEN_CACHE: dict[str, str] = {}


def log(msg: str) -> None:
    print(f"[rootfs] {msg}", flush=True)


def _parse_www_authenticate(header: str) -> dict[str, str]:
    scheme, _, rest = header.partition(" ")
    if scheme.lower() != "bearer":
        return {}
    out: dict[str, str] = {}
    for part in rest.split(","):
        key, _, value = part.strip().partition("=")
        out[key.strip().lower()] = value.strip().strip('"')
    return out


def _fetch_token(challenge: dict[str, str]) -> str:
    realm = challenge.get("realm")
    if not realm:
        return ""
    params = {}
    if challenge.get("service"):
        params["service"] = challenge["service"]
    if challenge.get("scope"):
        params["scope"] = challenge["scope"]
    url = realm + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with OPENER.open(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data.get("token") or data.get("access_token") or ""


def http_get(url: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": "dsv41-rootfs/1.0"}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    try:
        with OPENER.open(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise
        challenge = _parse_www_authenticate(exc.headers.get("WWW-Authenticate", ""))
        scope = challenge.get("scope", "")
        token = TOKEN_CACHE.get(scope) or _fetch_token(challenge)
        if not token:
            raise
        TOKEN_CACHE[scope] = token
        headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        with OPENER.open(req, timeout=120) as resp:
            return resp.read()


def resolve_manifest(registry: str, repo: str, ref: str, arch: str, os_name: str) -> tuple[dict, str]:
    url = f"https://{registry}/v2/{repo}/manifests/{ref}"
    raw = http_get(url, ACCEPT)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    doc = json.loads(raw)
    if "manifests" not in doc:
        return doc, digest

    candidates = doc["manifests"]
    chosen = None
    for item in candidates:
        plat = item.get("platform") or {}
        if plat.get("os") == os_name and plat.get("architecture") == arch:
            chosen = item
            break
    if chosen is None:
        platforms = sorted(
            f"{m.get('platform', {}).get('os')}/{m.get('platform', {}).get('architecture')}"
            for m in candidates
        )
        raise SystemExit(f"no {os_name}/{arch} manifest; available: {', '.join(platforms)}")

    child_digest = chosen["digest"]
    log(f"selected {os_name}/{arch} manifest {child_digest}")
    raw = http_get(f"https://{registry}/v2/{repo}/manifests/{child_digest}", ACCEPT)
    return json.loads(raw), child_digest


def download_blob(registry: str, repo: str, digest: str, dest: str, retries: int = 5) -> None:
    url = f"https://{registry}/v2/{repo}/blobs/{digest}"
    tmp = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dsv41-rootfs/1.0"})
            hasher = hashlib.sha256()
            written = 0
            try:
                resp = OPENER.open(req, timeout=120)
            except urllib.error.HTTPError as exc:
                if exc.code != 401:
                    raise
                challenge = _parse_www_authenticate(exc.headers.get("WWW-Authenticate", ""))
                scope = challenge.get("scope", "")
                token = TOKEN_CACHE.get(scope) or _fetch_token(challenge)
                TOKEN_CACHE[scope] = token
                req.add_header("Authorization", f"Bearer {token}")
                resp = OPENER.open(req, timeout=120)
            with resp, open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    hasher.update(chunk)
                    written += len(chunk)
            actual = "sha256:" + hasher.hexdigest()
            if actual != digest:
                raise ValueError(f"digest mismatch: {actual} != {digest}")
            os.replace(tmp, dest)
            log(f"blob {digest[:19]}… ok ({written / 1e6:.1f} MB)")
            return
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            log(f"blob {digest[:19]}… attempt {attempt}/{retries} failed: {exc}")
            try:
                os.remove(tmp)
            except OSError:
                pass
            if attempt == retries:
                raise
            time.sleep(min(5 * attempt, 30))


def extract_layer(layer_path: str, rootfs: str) -> tuple[int, int]:
    """Extract one layer honouring OCI whiteouts. Returns (files, whiteouts)."""
    tf = tarfile.open(layer_path, mode="r:*")
    members = tf.getmembers()
    whiteouts = 0

    # Pass 1: deletions from lower layers.
    for mem in members:
        name = mem.name.rstrip("/")
        base = os.path.basename(name)
        parent = os.path.dirname(name)
        norm_parent = os.path.normpath(parent) if parent else "."
        if os.path.isabs(name) or norm_parent == ".." or norm_parent.startswith("../"):
            log(f"skip unsafe whiteout path {mem.name}")
            continue
        if base == ".wh..wh..opq":
            target_dir = os.path.join(rootfs, parent)
            if os.path.isdir(target_dir):
                for entry in os.listdir(target_dir):
                    path = os.path.join(target_dir, entry)
                    if os.path.isdir(path) and not os.path.islink(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
            whiteouts += 1
        elif base.startswith(".wh."):
            target = os.path.join(rootfs, parent, base[4:])
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target, ignore_errors=True)
            elif os.path.exists(target) or os.path.islink(target):
                try:
                    os.remove(target)
                except OSError:
                    pass
            whiteouts += 1

    # Pass 2: materialise regular members.
    extracted = 0
    for mem in members:
        base = os.path.basename(mem.name.rstrip("/"))
        if base == ".wh..wh..opq" or base.startswith(".wh."):
            continue
        # Reject path traversal and special files; the image itself is trusted
        # (blobs are sha256-verified), so legitimate absolute symlinks are kept.
        norm = os.path.normpath(mem.name)
        if os.path.isabs(mem.name) or norm == ".." or norm.startswith("../"):
            log(f"skip unsafe path {mem.name}")
            continue
        if mem.isdev() or mem.isfifo():
            log(f"skip special file {mem.name}")
            continue
        try:
            try:
                tf.extract(mem, path=rootfs, set_attrs=False, filter="fully_trusted")
            except TypeError:  # Python < 3.12 without extraction filters
                tf.extract(mem, path=rootfs, set_attrs=False)
            extracted += 1
        except Exception as exc:  # noqa: BLE001 - permissions vary by container
            log(f"skip {mem.name}: {exc}")
    tf.close()
    return extracted, whiteouts


def disk_free_gib(path: str) -> float:
    probe = path
    while not os.path.exists(probe):
        probe = os.path.dirname(probe) or "/"
    return shutil.disk_usage(probe).free / (1024 ** 3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default="quay.nju.edu.cn")
    ap.add_argument("--repo", default="ascend/vllm-ascend")
    ap.add_argument("--tag", default="v0.30.0rc2-a3")
    ap.add_argument("--platform-arch", default="arm64")
    ap.add_argument("--platform-os", default="linux")
    ap.add_argument("--target", required=True, help="rootfs destination directory")
    ap.add_argument("--layers-dir", default=None, help="layer download cache (default: <target>.layers)")
    ap.add_argument("--keep-layers", action="store_true", help="keep downloaded layer archives")
    ap.add_argument("--force", action="store_true", help="discard layer markers and re-extract")
    ap.add_argument("--dry-run", action="store_true", help="list layers only")
    args = ap.parse_args()

    rootfs = os.path.abspath(os.path.expanduser(args.target))
    layers_dir = os.path.abspath(os.path.expanduser(args.layers_dir or rootfs + ".layers"))
    # Markers are tag-specific: layer ordering and whiteouts are meaningful
    # only within one image revision.
    safe_ref = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in args.tag)
    markers = os.path.join(layers_dir, ".done", safe_ref)

    log(f"registry=https://{args.registry} repo={args.repo} ref={args.tag}")
    log(f"target={rootfs} platform={args.platform_os}/{args.platform_arch}")
    try:
        manifest, resolved = resolve_manifest(
            args.registry, args.repo, args.tag, args.platform_arch, args.platform_os
        )
    except Exception as exc:  # noqa: BLE001
        log(f"failed to resolve manifest: {exc}")
        return 1

    layers = manifest.get("layers", [])
    total = sum(int(layer.get("size", 0)) for layer in layers)
    log(f"{len(layers)} layers, uncompressed size ≈ {total / 1e9:.1f} GB, resolved {resolved}")

    if args.dry_run:
        for i, layer in enumerate(layers):
            log(f"L{i:02d} {layer['digest']} {int(layer.get('size', 0)) / 1e6:.1f} MB")
        return 0

    os.makedirs(rootfs, exist_ok=True)
    os.makedirs(markers, exist_ok=True)
    free = disk_free_gib(layers_dir)
    if free < 25:
        log(f"⚠ 仅剩 {free:.1f} GiB 空间；镜像下载 + 解包建议至少 25 GiB")

    extracted_any = False
    for i, layer in enumerate(layers):
        digest = layer["digest"]
        size = int(layer.get("size", 0))
        marker = os.path.join(markers, digest.replace(":", "_"))
        if size == 0:
            log(f"L{i:02d} empty, skip")
            touch(marker)
            continue
        if os.path.exists(marker) and not args.force:
            log(f"L{i:02d} already applied, skip")
            continue

        archive = os.path.join(layers_dir, f"L{i:02d}.tar")
        log(f"L{i:02d} {digest[:19]}… {size / 1e6:.1f} MB")
        download_blob(args.registry, args.repo, digest, archive)
        files, whiteouts = extract_layer(archive, rootfs)
        log(f"L{i:02d} extracted {files} files, {whiteouts} whiteouts")
        extracted_any = True
        touch(marker)
        if not args.keep_layers:
            try:
                os.remove(archive)
            except OSError:
                pass

    provenance = {
        "registry": args.registry,
        "repository": args.repo,
        "reference": args.tag,
        "resolved_manifest": resolved,
        "platform": f"{args.platform_os}/{args.platform_arch}",
        "layers": [layer["digest"] for layer in layers],
        "extracted_at_unix": int(time.time()),
        "extracted_files_last_run": extracted_any,
    }
    with open(os.path.join(rootfs, ".dsv41-image"), "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
        fh.write("\n")

    log("ALL DONE")
    return 0


def touch(path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))


if __name__ == "__main__":
    if platform.machine() not in ("aarch64", "arm64"):
        print(f"[rootfs] warning: host arch is {platform.machine()}, image is arm64", file=sys.stderr)
    raise SystemExit(main())
