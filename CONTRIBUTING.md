# Contributing

## Scope

This repository is a **deployment** project, not a fork of vLLM Ascend. Changes
should stay within bootstrapping, configuration, observability, documentation
and the small optimization patch series.

## Ground rules

- Do not commit model weights, rootfs directories, downloaded layers or logs.
- Do not commit API keys or machine-specific configuration. Use
  `config/dsv41.env.example` for defaults and `config/dsv41.local.env` locally.
- Keep scripts POSIX-ish bash 4+ and dependency-free where possible.

## Changing patches

1. Apply the current series to a scratch rootfs.
2. Edit the source tree and validate accuracy + performance.
3. Regenerate the patch:

   ```bash
   cd "$DSV41_ROOTFS/vllm-workspace/vllm-ascend"
   git diff -- vllm_ascend/<path> > /tmp/new.patch
   ```

4. Replace the relevant file under `scripts/patches/vllm-ascend-v41-rc2/`,
   keeping upstream path prefixes (`a/vllm_ascend/...`, `b/vllm_ascend/...`).
5. Update `series.conf` and `docs/patches.md` if the ordering or rationale
   changes.
6. Run `bash scripts/apply_patches.sh check` against a clean rootfs.

Guidelines:

- Prefer opt-in env gates over unconditional behaviour changes.
- Every patch must have a measurable benefit or fix a correctness issue;
  otherwise drop it.
- Keep patches small and rebase-friendly; upstream moves quickly.

## Validation checklist

Before opening a pull request:

```bash
make lint
bash scripts/apply_patches.sh verify
python3 tests/smoke_test.py
python3 scripts/bench_2k.py --concurrency 1,4,16 --repeats 1
```

For performance changes, include before/after per-step latency at fixed
concurrency and prompt, plus the speculative-decoding acceptance rate.

## Commit messages

Use an imperative subject line, for example:

```
bench: add 8K acceptance-rate accounting
fix(watchdog): do not restart while /health is warming up
docs: explain why gate defaults are pinned to 1
```

## License

By contributing you agree that your contributions are licensed under
Apache-2.0, matching the rest of the repository.
