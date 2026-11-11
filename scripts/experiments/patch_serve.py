#!/usr/bin/env python3
"""Patch the *test copy* of scripts/serve.sh so graph / MC2 / kernel knobs can be
selected from the environment while keeping byte-identical defaults.

Only used against /cache/exp/repo-test on the test node; production serve.sh is
never touched by this script.
"""
import sys

path = sys.argv[1]

with open(path) as fh:
    text = fh.read()

if "DSV41_CUDAGRAPH_MODE" in text:
    print("already patched")
    raise SystemExit(0)

anchor = ': "${DSV41_EXTRA_SERVE_ARGS:=}"'
assert anchor in text, "anchor not found"
knobs = anchor + '''
# ---- experiment knobs (defaults reproduce the production configuration) ----
: "${DSV41_CUDAGRAPH_MODE:=FULL_DECODE_ONLY}"
: "${DSV41_STATIC_KERNEL:=false}"
: "${DSV41_NPUGRAPH_EX:=false}"
: "${DSV41_PREFILL_MC2:=false}"
: "${DSV41_FUSED_MC2:=false}"
: "${DSV41_MULTISTREAM_SHARED_EXPERT:=true}"
: "${DSV41_DRAFT_GRAPH:=1}"'''
text = text.replace(anchor, knobs, 1)

old_comp = '''COMP_JSON=$(printf '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[%s]}' "$DSV41_CUDAGRAPH_SIZES")'''
new_comp = '''COMP_JSON=$(printf '{"cudagraph_mode":"%s","cudagraph_capture_sizes":[%s]}' "$DSV41_CUDAGRAPH_MODE" "$DSV41_CUDAGRAPH_SIZES")'''
assert old_comp in text, "comp json not found"
text = text.replace(old_comp, new_comp, 1)

old_add = '''ADDITIONAL_JSON='{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false},"multistream_overlap_shared_expert":true}\''''
new_add = '''ADDITIONAL_JSON=$(printf '{"enable_cpu_binding":true,"enable_prefill_mc2":%s,"enable_fused_mc2":%s,"ascend_compilation_config":{"enable_npugraph_ex":%s,"enable_static_kernel":%s},"multistream_overlap_shared_expert":%s}' "$DSV41_PREFILL_MC2" "$DSV41_FUSED_MC2" "$DSV41_NPUGRAPH_EX" "$DSV41_STATIC_KERNEL" "$DSV41_MULTISTREAM_SHARED_EXPERT")'''
assert old_add in text, "additional json not found"
text = text.replace(old_add, new_add, 1)

with open(path, "w") as fh:
    fh.write(text)
print("patched", path)
