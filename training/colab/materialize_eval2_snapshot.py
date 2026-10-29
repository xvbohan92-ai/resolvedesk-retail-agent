"""Project a verified immutable snapshot into current read-model paths without changing history."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from training.common import ROOT
from training.cuda_eval2.evaluation import RUN_ID

DYNAMIC = {
    "evaluation-partial.json", "evaluation.json", "call-journal.json", "evaluation-console.log",
    "protocol-audit.json", "process-deadline-stop.json",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(root=ROOT):
    root = Path(root).resolve()
    folder = root / "reports/cuda-eval2"
    latest = read(folder / "cloud-evidence/latest.json")
    assert latest["run_id"] == RUN_ID
    expected = "reports/cuda-eval2/cloud-evidence/snapshots/" + latest["archive_sha256"]
    assert latest["snapshot_directory"] == expected
    snapshot = root / expected
    assert not snapshot.is_symlink()
    assert sha(snapshot / "import-audit.json") == latest["audit_sha256"]
    audit = read(snapshot / "import-audit.json")
    assert audit["archive_sha256"] == latest["archive_sha256"]
    assert audit["importer_source_sha256"] == sha(root / "training/colab/import_eval2_snapshot.py")
    receipt = read(snapshot / "snapshot-receipt.json")
    current = folder / "current-snapshot-materialization.json"
    old = read(current) if current.exists() else None
    if old:
        assert old["archive_sha256"] in (audit["previous_archive_sha256"], audit["archive_sha256"])
    files = []
    for name, wanted in receipt["files"].items():
        source = snapshot / name
        assert sha(source) == wanted and not source.is_symlink()
        if not name.startswith("reports/cuda-eval2/"):
            continue
        assert Path(name).parent == Path("reports/cuda-eval2")
        target = root / name
        assert not target.is_symlink()
        raw = source.read_bytes()
        if target.exists() and target.read_bytes() != raw:
            assert target.name in DYNAMIC and old
            previous = root / old["immutable_snapshot_directory"] / name
            assert target.read_bytes() == previous.read_bytes(), "Untracked changes to current raw evidence"
        if not target.exists() or target.read_bytes() != raw:
            fd, temporary = tempfile.mkstemp(prefix=".projection-", dir=folder)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
            os.replace(temporary, target)
        files.append(name)
    result = {
        "run_id": RUN_ID, "archive_sha256": latest["archive_sha256"],
        "import_audit_sha256": latest["audit_sha256"], "immutable_snapshot_directory": expected,
        "current_raw_files": sorted(files), "no_historical_snapshot_modified": True,
        "completed_records": audit["completed_records_retained"],
        "actual_complete_record_calls": audit["completed_record_calls_checked"],
        "retained_journal_entries": audit["journal_entries_retained"],
        "retained_inflight_started_entries": audit["started_journal_entries"],
        "extra_tail_entries_retained": audit["extra_inflight_tail_entries_retained"],
        "quality_results_published": False, "materialized_at_utc": datetime.now(UTC).isoformat(),
        "materializer_source_sha256": sha(Path(__file__)),
    }
    current.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(materialize(), indent=2))
