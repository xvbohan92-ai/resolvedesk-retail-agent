"""Validate and retain immutable eval2 cloud snapshots; never run a model or repair evidence."""

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from training.common import ROOT
from training.cuda_eval.provider import token_hash
from training.cuda_eval2.evaluation import ARMS, RUN_ID, value_hash
from training.cuda_eval2.verify_results import verify_record

PREFIX = "reports/cuda-eval2/"
RECEIPT = "snapshot-receipt.json"
IMMUTABLE = tuple(
    PREFIX + n
    for n in (
        "analysis-plan.json",
        "artifact-identity.json",
        "execution-addendum.json",
        "human-compute-authorization.json",
    )
)
REQUIRED = (
    set(IMMUTABLE)
    | {
        PREFIX + n
        for n in (
            "evaluation-launch.json",
            "budgeted-run-start.json",
            "evaluation-console.log",
            "prelaunch-cpu-console.log",
            "evaluation-partial.json",
            "call-journal.json",
        )
    }
    | {
        "colab-source-manifest.json",
        "colab-execution/resource-guard.json",
        "colab-execution/execution-overlay-manifest.json",
    }
)
ALLOWED = REQUIRED | {
    PREFIX + n
    for n in (
        "evaluation.json",
        "protocol-audit.json",
        "process-deadline-stop.json",
    )
}
MAX_ZIP, MAX_FILE, MAX_TOTAL = 128 * 1024**2, 128 * 1024**2, 256 * 1024**2
RESPONSE_KEYS = {"raw", "timing", "completed_at_utc"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(data):
    return json.loads(data, object_pairs_hook=pairs_unique)


def load_archive(archive, expected_sha256, root=ROOT):
    archive, root = Path(archive).resolve(), Path(root).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected lowercase SHA256 is required")
    if archive.is_relative_to(root):
        raise ValueError("Keep every original ZIP outside the checkout")
    if archive.stat().st_size > MAX_ZIP:
        raise ValueError("Snapshot ZIP exceeds bounded size")
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            raise ValueError("Snapshot archive SHA256 mismatch")
    with ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or len(names) > len(ALLOWED) + 1:
            raise ValueError("Duplicate or excessive ZIP members")
        if set(names) - (ALLOWED | {RECEIPT}) or not REQUIRED <= set(names) or RECEIPT not in names:
            raise ValueError("Missing or unapproved snapshot member")
        if sum(e.file_size for e in entries) > MAX_TOTAL:
            raise ValueError("Snapshot exceeds total uncompressed limit")
        for entry in entries:
            p = PurePosixPath(entry.filename)
            if (
                str(p) != entry.filename
                or p.is_absolute()
                or ".." in p.parts
                or "\\" in entry.filename
                or entry.is_dir()
                or stat.S_IFMT(entry.external_attr >> 16) not in (0, stat.S_IFREG)
                or entry.file_size > MAX_FILE
            ):
                raise ValueError("Unsafe ZIP member or size")
        files = {n: bundle.read(n) for n in names if n != RECEIPT}
        receipt_bytes = bundle.read(RECEIPT)
    receipt = read_json(receipt_bytes)
    if receipt.get("run_id") != RUN_ID or set(receipt["files"]) != set(files):
        raise ValueError("Snapshot receipt must enumerate every and only archived member")
    if any(digest(data) != receipt["files"][n] for n, data in files.items()):
        raise ValueError("Snapshot member SHA256 differs from receipt")
    for name in IMMUTABLE:
        if files[name] != (root / name).read_bytes():
            raise ValueError(f"Immutable local identity differs: {name}")
    return files, receipt, receipt_bytes


def validate_payload(files, receipt, root=ROOT):
    """CPU trace/action replay plus token-array hashes; no tokenizer/model inference."""
    plan, identity, addendum = (
        read_json(files[PREFIX + n])
        for n in (
            "analysis-plan.json",
            "artifact-identity.json",
            "execution-addendum.json",
        )
    )
    addendum_sha = digest(files[PREFIX + "execution-addendum.json"])
    assert receipt["analysis_plan_sha256"] == digest(files[PREFIX + "analysis-plan.json"])
    assert receipt["execution_addendum_sha256"] == addendum_sha
    assert addendum["analysis_plan_sha256"] == receipt["analysis_plan_sha256"]
    assert addendum["artifact_identity_sha256"] == digest(files[PREFIX + "artifact-identity.json"])
    manifest = read_json(files["colab-source-manifest.json"])
    for name, wanted in plan["source_inputs_sha256"].items():
        assert manifest["files"][name] == wanted, name
    overlay = read_json(files["colab-execution/execution-overlay-manifest.json"])
    assert overlay["run_id"] == RUN_ID and overlay["execution_addendum_sha256"] == addendum_sha
    assert overlay["analysis_plan_sha256"] == receipt["analysis_plan_sha256"]
    for name, wanted in overlay["files"].items():
        p = PurePosixPath(name)
        assert not p.is_absolute() and ".." not in p.parts and "\\" not in name
        assert digest((Path(root) / name).read_bytes()) == wanted, name
    guard = read_json(files["colab-execution/resource-guard.json"])
    assert guard["run_id"] == RUN_ID and guard["max_existing_compute_units"] == 10
    assert not guard["new_purchases_allowed"] and not guard["renewal_allowed"]
    assert guard["process_deadline_utc"] == addendum["accounting"]["process_deadline_utc"]
    assert guard["runtime_stop_utc"] == addendum["accounting"]["runtime_disconnect_deadline_utc"]
    partial = read_json(files[PREFIX + "evaluation-partial.json"])
    final = read_json(files[PREFIX + "evaluation.json"]) if PREFIX + "evaluation.json" in files else None
    report = final or partial
    assert report["run_id"] == RUN_ID and report["provider"] == "torch_cuda"
    assert report["analysis_plan_sha256"] == receipt["analysis_plan_sha256"]
    assert report["artifact_identity_sha256"] == addendum["artifact_identity_sha256"]
    assert report["execution_addendum_sha256"] == addendum_sha
    assert report["source_inputs_sha256"] == plan["source_inputs_sha256"]
    assert report["cases_sha256"] == plan["cases_sha256"]
    assert report["previous_interrupted_run"] == plan["previous_interrupted_run"]
    assert report["old_outputs_pooled_or_replaced"] is False
    assert report["training_repeated_or_modified"] is False
    records, journal = report["records"], read_json(files[PREFIX + "call-journal.json"])
    order = [(a, c) for a in ARMS for c in plan["case_ids"]]
    actual = [(r["variant"], r["case_id"]) for r in records]
    assert len(records) <= 96 and actual == order[: len(actual)]
    assert partial["records"] == records[: len(partial["records"])]
    if final:
        assert final["status"] == "complete" and len(records) == 96 and not final.get("pending_case")
    else:
        assert partial["status"] == "partial"
    pending = report.get("pending_case")
    if pending:
        assert len(records) < 96 and (pending["variant"], pending["case_id"]) == order[len(records)]
    assert receipt["completed_records"] in {len(partial["records"]), len(records)}
    assert receipt["completed_by_arm"] == dict(
        Counter(
            r["variant"]
            for r in (
                partial["records"] if receipt["completed_records"] == len(partial["records"]) else records
            )
        )
    )
    cases = {
        c["id"]: c for c in read_json((Path(root) / "training/cuda_eval2/datasets/cases.json").read_bytes())
    }
    expected = []
    for record in records:
        assert 0 < record["model_calls"] == len(record["trace"]) <= 12
        verify_record(
            record,
            cases[record["case_id"]],
            identity,
            plan["generation"],
            model_kwargs=plan["inference_model_kwargs"],
        )
        expected.extend(
            (record["variant"], record["case_id"], i, observation)
            for i, observation in enumerate(record["trace"], 1)
        )
    assert len(journal) >= len(expected)
    last_pair, index, tail_position = None, 0, len(records) - 1
    for position, entry in enumerate(journal):
        pair = (entry["variant"], entry["case_id"])
        if position == len(expected):
            assert len(records) < 96 and pair == order[len(records)]
        if pair != last_pair:
            index, last_pair = 0, pair
            if position >= len(expected):
                tail_position += 1
                assert pair == order[tail_position]
        index += 1
        assert entry["call_index"] == index <= 12
        assert entry["run_id"] == RUN_ID and entry["execution_addendum_sha256"] == addendum_sha
        assert entry["request_sha256"] == value_hash(entry["request_messages"])
        assert addendum["created_at_utc"] < entry["started_at_utc"]
        if position < len(expected):
            arm, cid, call, observation = expected[position]
            assert (entry["variant"], entry["case_id"], entry["call_index"]) == (arm, cid, call)
            assert entry["status"] == "completed"
            assert entry["raw"] == observation["raw"] and entry["timing"] == observation["timing"]
        if entry["status"] == "started":
            assert position == len(journal) - 1 and not RESPONSE_KEYS & set(entry)
        else:
            assert entry["status"] == "completed" and isinstance(entry["raw"], str)
            assert entry["started_at_utc"] <= entry["completed_at_utc"]
            timing = entry["timing"]
            assert timing["execution_addendum_sha256"] == addendum_sha
            assert timing["execution_authorization_mode"] == "metered_existing_units"
            assert timing["variant"] == entry["variant"] and timing["provider"] == "torch_cuda"
            assert timing["generation_settings"] == plan["generation"]
            for key, count in (("input_token_ids", "input_tokens"), ("generated_token_ids", "output_tokens")):
                ids = timing[key]
                assert all(type(v) is int and v >= 0 for v in ids)
                assert len(ids) == timing[count] and token_hash(ids) == timing[key + "_sha256"]
            assert timing["input_tokens"] > 0 and 0 <= timing["output_tokens"] <= 512
    if final:
        assert len(journal) == len(expected) and all(e["status"] == "completed" for e in journal)
        assert final["call_journal_sha256"] == digest(files[PREFIX + "call-journal.json"])
    return {
        "records": records,
        "journal": journal,
        "console": files[PREFIX + "evaluation-console.log"],
        "snapshot_at_utc": receipt["snapshot_at_utc"],
        "audit": {
            "completed_records_retained": len(records),
            "completed_by_arm": dict(Counter(r["variant"] for r in records)),
            "completed_record_calls_checked": len(expected),
            "journal_entries_retained": len(journal),
            "completed_journal_entries": sum(e["status"] == "completed" for e in journal),
            "started_journal_entries": sum(e["status"] == "started" for e in journal),
            "extra_inflight_tail_entries_retained": len(journal) - len(expected),
            "complete_96_record_artifact_present": final is not None,
            "token_arrays_and_hashes_checked": True,
            "tokenizer_redecoded": False,
            "quality_result_inferred": False,
            "model_or_cloud_calls": False,
        },
    }


def monotonic(previous, current):
    if current["snapshot_at_utc"] < previous["snapshot_at_utc"]:
        raise ValueError("Older snapshots never replace the latest evidence pointer")
    if current["records"][: len(previous["records"])] != previous["records"]:
        raise ValueError("Completed records must be an immutable prefix")
    old, new = previous["journal"], current["journal"]
    if len(new) < len(old):
        raise ValueError("Never drop issued journal calls, including a started tail")
    transitioned = 0
    for before, after in zip(old, new, strict=False):
        if before["status"] == "completed" or after["status"] == "started":
            if before != after:
                raise ValueError("Completed/started journal metadata cannot be rewritten")
        else:
            assert before["status"] == "started" and after["status"] == "completed"
            if set(after) != set(before) | RESPONSE_KEYS or any(
                after[k] != v for k, v in before.items() if k != "status"
            ):
                raise ValueError("Started-to-completed transition must preserve every request field")
            transitioned += 1
    if not current["console"].startswith(previous["console"]):
        raise ValueError("Console evidence must append without truncation or rewriting")
    return {
        "prior_completed_calls_unchanged": sum(e["status"] == "completed" for e in old),
        "started_to_completed_transitions": transitioned,
    }


def safe_output(root):
    root = Path(root).resolve()
    path = root
    for part in ("reports", "cuda-eval2", "cloud-evidence"):
        path /= part
        if path.is_symlink():
            raise ValueError("Evidence output may not traverse symlinks")
    if path.exists() and any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Evidence output contains a symlink")
    return path


def import_snapshot(archive, expected_sha256, *, root=ROOT):
    files, receipt, receipt_bytes = load_archive(archive, expected_sha256, root)
    current = validate_payload(files, receipt, root)
    evidence = safe_output(root)
    evidence.mkdir(parents=True, exist_ok=True)
    lock = evidence / ".import.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    staging = None
    try:
        destination = evidence / "snapshots" / expected_sha256
        if destination.exists():
            saved = read_json((destination / "import-audit.json").read_bytes())
            assert saved["archive_sha256"] == expected_sha256
            assert saved["snapshot_receipt_sha256"] == digest(receipt_bytes)
            for name, data in {**files, RECEIPT: receipt_bytes}.items():
                if (destination / name).read_bytes() != data:
                    raise ValueError("Previously retained raw snapshot bytes changed")
            return {**saved, "already_imported": True}
        latest = evidence / "latest.json"
        prefix_audit = {"prior_completed_calls_unchanged": 0, "started_to_completed_transitions": 0}
        previous_sha = None
        if latest.exists():
            pointer = read_json(latest.read_bytes())
            previous_sha = pointer["archive_sha256"]
            assert re.fullmatch(r"[0-9a-f]{64}", previous_sha)
            previous_dir = evidence / "snapshots" / previous_sha
            previous_audit_bytes = (previous_dir / "import-audit.json").read_bytes()
            assert digest(previous_audit_bytes) == pointer["audit_sha256"]
            previous_audit = read_json(previous_audit_bytes)
            previous_receipt_bytes = (previous_dir / RECEIPT).read_bytes()
            assert digest(previous_receipt_bytes) == previous_audit["snapshot_receipt_sha256"]
            previous_receipt = read_json(previous_receipt_bytes)
            assert REQUIRED <= set(previous_receipt["files"]) <= ALLOWED
            previous_files = {n: (previous_dir / n).read_bytes() for n in previous_receipt["files"]}
            assert all(digest(b) == previous_receipt["files"][n] for n, b in previous_files.items())
            previous = validate_payload(previous_files, previous_receipt, root)
            prefix_audit = monotonic(previous, current)
            for name in files.keys() & previous_files.keys():
                if (
                    name
                    not in {
                        PREFIX + n
                        for n in (
                            "evaluation-partial.json",
                            "evaluation.json",
                            "call-journal.json",
                            "evaluation-console.log",
                        )
                    }
                    and files[name] != previous_files[name]
                ):
                    raise ValueError(f"Static source/guard/launch metadata changed: {name}")
        audit = {
            "run_id": RUN_ID,
            "imported_at_utc": datetime.now(UTC).isoformat(),
            "archive_path": str(Path(archive).resolve()),
            "archive_sha256": expected_sha256,
            "snapshot_at_utc": receipt["snapshot_at_utc"],
            "previous_archive_sha256": previous_sha,
            "snapshot_member_hashes_verified": len(files),
            "snapshot_receipt_sha256": digest(receipt_bytes),
            "process_return_code_recorded": receipt.get("process_return_code"),
            "importer_source_sha256": digest(Path(__file__).read_bytes()),
            "raw_snapshots_overwritten": False,
            "original_zip_kept_outside_checkout": True,
            "scope": (
                "CPU trace/prefix import only; guard declarations checked; billing and liveness not queried"
            ),
            **current["audit"],
            **prefix_audit,
        }
        (evidence / "snapshots").mkdir(exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=evidence / "snapshots"))
        for name, data in {**files, RECEIPT: receipt_bytes}.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (staging / "import-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        staging.rename(destination)
        staging = None
        pointer = {
            "run_id": RUN_ID,
            "archive_sha256": expected_sha256,
            "snapshot_directory": str(destination.relative_to(Path(root).resolve())),
            "audit_sha256": digest((destination / "import-audit.json").read_bytes()),
        }
        temporary = evidence / "latest.json.tmp"
        with temporary.open("x") as stream:
            json.dump(pointer, stream, indent=2)
        temporary.replace(latest)
        return audit
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(import_snapshot(args.archive, args.expected_sha256), indent=2))


if __name__ == "__main__":
    main()
