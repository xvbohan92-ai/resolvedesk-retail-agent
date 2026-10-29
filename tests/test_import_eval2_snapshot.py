"""CPU archive/storage corruption contracts; generated fixtures never represent model results."""

import copy
import json
import stat
import warnings
from zipfile import ZipFile, ZipInfo

import pytest

from training.colab import import_eval2_snapshot as importer
from training.common import ROOT


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    files = {name: b"{}" for name in importer.REQUIRED}
    for name in importer.IMMUTABLE:
        files[name] = (ROOT / name).read_bytes()
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[name])
    return root, files


def archive(path, files, *, extra=(), receipt_changes=None, special=None):
    receipt = {
        "run_id": importer.RUN_ID,
        "snapshot_at_utc": "2026-10-29T01:00:00+00:00",
        "files": {name: importer.digest(data) for name, data in files.items()},
    }
    if receipt_changes:
        receipt.update(receipt_changes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(path, "w") as z:
            for name, data in files.items():
                z.writestr((special or {}).get(name, name), data)
            z.writestr(importer.RECEIPT, json.dumps(receipt))
            for name, data in extra:
                z.writestr(name, data)
    return importer.digest(path.read_bytes())


def test_trusted_sha_and_exact_receipt_required_before_any_destination_write(source, tmp_path):
    root, files = source
    path = tmp_path / "snapshot.zip"
    expected = archive(path, files)
    payload, receipt, _ = importer.load_archive(path, expected, root)
    assert payload == files and set(receipt["files"]) == set(files)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        importer.load_archive(path, "0" * 64, root)
    wrong = archive(path, files, receipt_changes={"files": {}})
    with pytest.raises(ValueError, match="every and only"):
        importer.load_archive(path, wrong, root)
    assert not (root / "reports/cuda-eval2/cloud-evidence").exists()


@pytest.mark.parametrize(
    "kind", ["traversal", "duplicate", "symlink", "unlisted", "receipt_hash", "local_identity"]
)
def test_archive_corruption_is_rejected_without_extracting(source, tmp_path, kind):
    root, files = source
    path = tmp_path / "snapshot.zip"
    extra, special = [], {}
    if kind == "traversal":
        extra = [("../outside-secret", b"blocked")]
    elif kind == "duplicate":
        extra = [(next(iter(files)), b"duplicate")]
    elif kind == "symlink":
        name = importer.PREFIX + "evaluation-console.log"
        info = ZipInfo(name)
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        files[name] = b"/outside"
        special[name] = info
    elif kind == "unlisted":
        extra = [("reports/cuda-eval2/leaked.env", b"blocked")]
    elif kind == "local_identity":
        files[importer.IMMUTABLE[0]] = b'{"changed":true}'
    expected = archive(path, files, extra=extra, special=special)
    if kind == "receipt_hash":
        with ZipFile(path) as z:
            receipt = json.loads(z.read(importer.RECEIPT))
        receipt["files"][next(iter(files))] = "0" * 64
        expected = archive(path, files, receipt_changes=receipt)
    with pytest.raises((ValueError, AssertionError)):
        importer.load_archive(path, expected, root)
    assert not (tmp_path / "outside-secret").exists()


def test_size_limits_and_zip_in_checkout_are_rejected(source, tmp_path, monkeypatch):
    root, files = source
    path = tmp_path / "snapshot.zip"
    expected = archive(path, files)
    monkeypatch.setattr(importer, "MAX_ZIP", 1)
    with pytest.raises(ValueError, match="bounded size"):
        importer.load_archive(path, expected, root)
    monkeypatch.setattr(importer, "MAX_ZIP", 128 * 1024**2)
    inside = root / "snapshot.zip"
    inside.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="outside"):
        importer.load_archive(inside, expected, root)


def entry(status="started"):
    record = {
        "status": "started",
        "variant": importer.ARMS[0],
        "case_id": "cpu-fixture",
        "call_index": 1,
        "request_messages": [{"role": "user", "content": "CPU storage fixture"}],
        "request_sha256": "unchanged-request",
        "started_at_utc": "2026-10-29T01:00:00+00:00",
        "run_id": importer.RUN_ID,
        "execution_addendum_sha256": "b" * 64,
    }
    if status == "completed":
        record.update(
            status=status,
            raw="raw invalid enum retained",
            timing={"fixture": True},
            completed_at_utc="2026-10-29T01:00:01+00:00",
        )
    return record


def state(*, journal=None, records=None, console=b"first\n", time="2026-10-29T01:00:00+00:00"):
    return {"journal": journal or [], "records": records or [], "console": console, "snapshot_at_utc": time}


def test_started_to_completed_preserves_every_request_field_and_allows_only_response_addition():
    before = state(journal=[entry()])
    after = state(journal=[entry("completed")], console=b"first\nsecond\n")
    assert importer.monotonic(before, after) == {
        "prior_completed_calls_unchanged": 0,
        "started_to_completed_transitions": 1,
    }
    damaged = copy.deepcopy(after)
    damaged["journal"][0]["request_messages"][0]["content"] = "replaced"
    with pytest.raises(ValueError, match="every request field"):
        importer.monotonic(before, damaged)
    damaged = copy.deepcopy(after)
    damaged["journal"][0]["hidden_repair"] = "not allowed"
    with pytest.raises(ValueError, match="every request field"):
        importer.monotonic(before, damaged)


@pytest.mark.parametrize(
    "change", ["drop_tail", "rewrite_completed", "change_records", "truncate_console", "old_time"]
)
def test_all_issued_calls_completed_rows_and_console_are_monotonic(change):
    before = state(journal=[entry("completed"), {**entry(), "call_index": 2}], records=[{"case_id": "done"}])
    after = copy.deepcopy(before)
    if change == "drop_tail":
        after["journal"].pop()
    elif change == "rewrite_completed":
        after["journal"][0]["raw"] = "corrected output"
    elif change == "change_records":
        after["records"][0]["case_id"] = "substituted"
    elif change == "truncate_console":
        after["console"] = b""
    else:
        after["snapshot_at_utc"] = "2026-10-29T00:59:00+00:00"
    with pytest.raises(ValueError):
        importer.monotonic(before, after)


def test_duplicate_json_keys_do_not_hide_rewritten_identity():
    with pytest.raises(ValueError, match="Duplicate JSON"):
        importer.read_json('{"run_id":"old","run_id":"new"}')


def test_output_ancestor_or_leaf_symlinks_are_rejected(source, tmp_path):
    root, _ = source
    evidence = root / "reports/cuda-eval2/cloud-evidence"
    evidence.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        importer.safe_output(root)
    evidence.unlink()
    evidence.mkdir()
    (evidence / "latest.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="symlink"):
        importer.safe_output(root)


def test_import_stores_immutable_snapshots_and_advances_only_derived_pointer(source, tmp_path, monkeypatch):
    root, files = source

    # Storage fixtures have zero neural outputs. The real payload validator is tested by actual import.
    def validate(payload, receipt, root):
        return {
            **state(
                console=payload[importer.PREFIX + "evaluation-console.log"], time=receipt["snapshot_at_utc"]
            ),
            "audit": {"completed_records_retained": 0, "quality_result_inferred": False},
        }

    monkeypatch.setattr(importer, "validate_payload", validate)
    first = tmp_path / "first.zip"
    sha = archive(first, files)
    audit = importer.import_snapshot(first, sha, root=root)
    evidence = root / "reports/cuda-eval2/cloud-evidence"
    saved = evidence / "snapshots" / sha
    assert audit["original_zip_kept_outside_checkout"] and first.exists()
    before = {str(p.relative_to(saved)): p.read_bytes() for p in saved.rglob("*") if p.is_file()}
    assert importer.import_snapshot(first, sha, root=root)["already_imported"]
    files[importer.PREFIX + "evaluation-console.log"] += b"\nappend only"
    second = tmp_path / "second.zip"
    sha2 = archive(second, files, receipt_changes={"snapshot_at_utc": "2026-10-29T01:01:00+00:00"})
    importer.import_snapshot(second, sha2, root=root)
    assert json.loads((evidence / "latest.json").read_text())["archive_sha256"] == sha2
    assert before == {str(p.relative_to(saved)): p.read_bytes() for p in saved.rglob("*") if p.is_file()}
    assert not (root / "reports/cuda-eval2/evaluation-partial.json").exists()
