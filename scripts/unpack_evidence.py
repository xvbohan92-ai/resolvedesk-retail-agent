"""Restore losslessly compressed public scientific artifacts after checking their public SHA256."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "evidence-archives.json").read_text())
for name, record in manifest.items():
    archive = ROOT / record["archive"]
    packed = archive.read_bytes()
    if hashlib.sha256(packed).hexdigest() != record["archive_sha256"]:
        raise SystemExit(f"Archive SHA mismatch: {name}")
    raw = gzip.decompress(packed)
    if len(raw) != record["size"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise SystemExit(f"Evidence SHA mismatch: {name}")
    target = ROOT / name
    if target.exists() and target.read_bytes() != raw:
        raise SystemExit(f"Refusing to replace different artifact: {name}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
print(f"Verified/restored {len(manifest)} public evidence files")
