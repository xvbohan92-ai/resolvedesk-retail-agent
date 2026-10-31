"""Public-only CPU consistency contracts; these fixtures are not model evidence."""

import csv
import hashlib
import importlib.util
import io
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_partial40_reconcile", ROOT / "scripts/reconcile_partial40_public_results.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FIG = Path(MODULE.FIG)
SUMMARY = Path(MODULE.SUMMARY)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def public_only(tmp_path):
    """Only public aggregates/images/48 authored cases; no weights or raw records."""
    root = tmp_path / "public-only"
    inputs = [FIG / name for name in MODULE.ASSETS] + [FIG / "values.json", SUMMARY, Path(MODULE.CASES)]
    for relative in inputs:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return root


def rebind(root, changed_csv=None):
    """Refresh outer public digests so mutation tests reach semantic consistency."""
    values = load(root / FIG / "values.json")
    summary = load(root / SUMMARY)
    if changed_csv:
        h = sha(root / FIG / changed_csv)
        values["output_files_sha256"][changed_csv] = h
        summary["public_files_sha256"][str(FIG / changed_csv)] = h
    dump(root / FIG / "values.json", values)
    summary["public_values_sha256"] = sha(root / FIG / "values.json")
    dump(root / SUMMARY, summary)


def rewrite_csv(root, name, mutate):
    path = root / FIG / name
    reader = csv.DictReader(io.StringIO(path.read_text()))
    fields, records = reader.fieldnames, list(reader)
    mutate(records)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    path.write_text(stream.getvalue(), newline="")
    rebind(root, name)


def test_public_only_inputs_suffice_without_raw_weights_or_private_gate(public_only):
    result = MODULE.reconcile(public_only)
    assert result["business_outcomes"] == [30, 30, 33]
    assert (result["cases"], result["paired_rows"], result["paired_calls"]) == (40, 120, 657)
    assert result["failed_rows"] == 27
    assert result["raw_record_or_tokens_verified"] is False
    assert result["private_full_source_seal_verified"] is False
    assert not list(public_only.rglob("*.safetensors"))
    assert not list(public_only.rglob("*journal*"))
    assert not list(public_only.rglob("*evaluation-partial*"))


def test_csv_call_tamper_fails_even_with_rebound_file_hashes(public_only):
    rewrite_csv(public_only, "per-case.csv", lambda records: records[0].update(model_calls="5"))
    with pytest.raises(MODULE.ReconciliationError, match="model_calls"):
        MODULE.reconcile(public_only)


def test_claim_of_complete_original144_is_rejected(public_only):
    values = load(public_only / FIG / "values.json")
    values["original144_gate_complete"] = True
    dump(public_only / FIG / "values.json", values)
    rebind(public_only)
    with pytest.raises(MODULE.ReconciliationError, match="incomplete original"):
        MODULE.reconcile(public_only)


def test_omitted_failure_is_detected_against_all_csv_rows(public_only):
    for path in [public_only / FIG / "values.json", public_only / SUMMARY]:
        value = load(path)
        value["failed_records"].pop()
        dump(path, value)
    rebind(public_only)
    with pytest.raises(MODULE.ReconciliationError, match="failed arm/case"):
        MODULE.reconcile(public_only)


def test_wrong_authorization_denominator_fails_against_authored_cases(public_only):
    for path in [public_only / FIG / "values.json", public_only / SUMMARY]:
        value = load(path)
        value["selection_denominators"]["initial_permissions"]["yes"] = 9
        dump(path, value)
    rebind(public_only)
    with pytest.raises(MODULE.ReconciliationError, match="case-derived denominators"):
        MODULE.reconcile(public_only)


def test_original_index_must_skip_unselected_old_arm_tail(public_only):
    rewrite_csv(public_only, "per-case.csv", lambda records: records[40].update(original_record_index="40"))
    with pytest.raises(MODULE.ReconciliationError, match="original retained-record index"):
        MODULE.reconcile(public_only)


def test_opaque_record_hash_is_not_accepted_without_sha256_shape(public_only):
    rewrite_csv(
        public_only, "per-case.csv", lambda records: records[0].update(source_record_sha256="not-a-digest")
    )
    with pytest.raises(MODULE.ReconciliationError, match="opaque record digest"):
        MODULE.reconcile(public_only)


def test_asset_traversal_is_rejected_before_an_unlisted_file_read(public_only):
    values = load(public_only / FIG / "values.json")
    values["output_files_sha256"]["../outside.json"] = "0" * 64
    dump(public_only / FIG / "values.json", values)
    rebind(public_only)
    with pytest.raises(MODULE.ReconciliationError, match="allowlisted asset inventory"):
        MODULE.reconcile(public_only)
    with pytest.raises(MODULE.ReconciliationError, match="Unsafe relative"):
        MODULE.read_bytes(public_only, "../outside.json")


def test_symlink_case_input_rejected(public_only, tmp_path):
    path = public_only / MODULE.CASES
    outside = tmp_path / "outside-cases.json"
    shutil.copyfile(path, outside)
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(MODULE.ReconciliationError, match="Symlink"):
        MODULE.reconcile(public_only)


def test_training_curve_truncation_fails_even_with_rebound_hash(public_only):
    rewrite_csv(public_only, "training.csv", lambda records: records.pop())
    with pytest.raises(MODULE.ReconciliationError, match="historical curve length"):
        MODULE.reconcile(public_only)


def test_parse_failure_cannot_borrow_later_correct_fields(public_only):
    rewrite_csv(
        public_only, "per-case.csv", lambda records: records[0].update(initial_raw_parse_failure="True")
    )
    with pytest.raises(MODULE.ReconciliationError, match="Parse failure borrowed repaired"):
        MODULE.reconcile(public_only)


def test_optional_receipt_is_exclusive_and_outside_public_tree(public_only, tmp_path, capsys):
    receipt = tmp_path / "public-reconciliation.json"
    assert MODULE.main(["--root", str(public_only), "--output", str(receipt)]) == 0
    assert load(receipt)["raw_record_or_tokens_verified"] is False
    assert MODULE.main(["--root", str(public_only), "--output", str(receipt)]) == 1
    assert MODULE.main(["--root", str(public_only), "--output", str(public_only / "receipt.json")]) == 1
    assert "Receipt output must be outside" in capsys.readouterr().err
