"""Check pinned external KLD receipts; does not run or qualify the PR model."""

import argparse
import hashlib
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("evidence_dir", type=Path)
root = parser.parse_args().evidence_dir
manifest = json.loads((root / "SHA256.json").read_text())
for name, expected in manifest.items():
    path = root / name
    assert path.parent == root and path.is_file(), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
current = json.loads((root / "comparison.json").read_text())
audit = json.loads((root / "audit.json").read_text())
historical = json.loads((root / "historical-fp8-audit.json").read_text())
assert audit["status"] == historical["status"] == "passed"
assert current["windows"] == historical["windows"] == 32
assert current["true_decode_rows_per_window"] == 2046
assert historical["true_decode_rows_per_window"] == 2046
assert historical["conditions"]["kv_dtype"] == "fp8_ds_mla"
assert historical["conditions"]["attention"] == "FLASHINFER_MLA_SPARSE_SM120"
assert historical["dcp"] == 1 and not historical["mtp"]
assert current["capture_image_local_id"] == (
    "sha256:a7fde8169ec24fff3d7f1a7a8a50375a90e6c9e8cf71254680a281f54be37ca6"
)
ids = None
for arm in ["nvfp4", "fp8"]:
    rows = current["arms"][arm]["per_window"]
    now = [r["window_id"] for r in rows]
    assert len(now) == len(set(now)) == 32
    if ids is None:
        ids = now
    assert now == ids
    for row in rows:
        assert row["prediction_rows"] == 2047 and row["true_decode_rows"] == 2046
        assert len(row["score_sha256"]) == len(row["raw_sha256"]) == 64
    mean = math.fsum(r["true_decode_mean_kld"] for r in rows) / 32
    assert math.isclose(
        mean, current["arms"][arm]["mean_true_decode_kld"], abs_tol=1e-15, rel_tol=0
    )
old = historical["per_window"]
assert [r["window_id"] for r in old] == ids
for old_row, row in zip(old, current["arms"]["fp8"]["per_window"]):
    assert old_row["token_values_sha256"] == row["token_sha256"]
assert math.isclose(
    math.fsum(r["true_decode_mean_kld"] for r in old) / 32,
    historical["mean_true_decode_kld"],
    abs_tol=1e-15,
    rel_tol=0,
)
delta = (
    current["arms"]["fp8"]["mean_true_decode_kld"]
    - current["arms"]["nvfp4"]["mean_true_decode_kld"]
)
assert math.isclose(delta, current["fp8_minus_nvfp4"], abs_tol=1e-15, rel_tol=0)
print(
    "PASS: pinned external evidence, 64 current and 32 historical FP8 window receipts; no PR-head GPU qualification"
)
