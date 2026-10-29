"""Independent-run CPU contracts; no model output on the new test population."""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from training.common import ROOT
from training.cuda_eval import evaluation as frozen
from training.cuda_eval.provider import RecordingProvider
from training.cuda_eval2 import evaluation, fixtures, run


def test_import_and_cli_help_do_not_import_a_gpu_or_array_runtime():
    code = """
import sys
from training.cuda_eval2 import evaluation, run
assert not any(n in sys.modules for n in ('torch','mlx','numpy','transformers','safetensors'))
assert evaluation.CallJournal.__module__ == 'training.cuda_eval.evaluation'
assert run.TorchProvider.__module__ == 'training.cuda_eval.provider'
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-B", "-m", "training.cuda_eval2.run", "--help"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0 and "--free-gpu-confirmed" in result.stdout


def test_cpu_preflight_reconciles_delivered_cohort_and_proof_without_ignored_model_files(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 151645, "pad_token_id": 151643}))
    code = """
import json, sys
from training.cuda_eval2 import evaluation
names = ['analysis-plan.json','artifact-identity.json','evaluation-partial.json','call-journal.json']
before = {n:(evaluation.REPORTS/n).read_bytes() if (evaluation.REPORTS/n).exists() else None for n in names}
result = evaluation.cpu_preflight(base_path=sys.argv[1])
assert result['run_id'] == 'resolvedesk-cuda-eval-002'
assert len(result['case_ids']) == len(set(result['case_ids'])) == 48
assert all(c.startswith('cuda-fresh2-test-') for c in result['case_ids'])
assert result['generation']['max_new_tokens'] == 512 and not result['generation']['do_sample']
assert not result['model_imported_or_called'] and not result['files_written']
assert not any(n in sys.modules for n in ('torch','mlx','numpy','transformers','safetensors'))
after = {n:(evaluation.REPORTS/n).read_bytes() if (evaluation.REPORTS/n).exists() else None for n in names}
assert before == after
print(json.dumps({'cases':len(result['case_ids']),'sources':len(result['source_inputs_sha256'])}))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"cases": 48, "sources": len(evaluation.source_names())}


def test_output_isolation_rejects_old_run_and_ancestor_or_leaf_links(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    own = tmp_path / "reports/cuda-eval2"
    assert evaluation.report_directory(own / "separate") == own / "separate"
    for path in ("reports/cuda-eval", "reports/v7", "reports/cuda-torch", "training/cuda_eval2"):
        with pytest.raises(ValueError, match="reports/cuda-eval2"):
            evaluation.report_directory(tmp_path / path)
    own.parent.mkdir()
    own.symlink_to(tmp_path / "reports/cuda-eval", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinks"):
        evaluation.report_directory(own)
    own.unlink()
    own.mkdir()
    (own / "call-journal.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="Symlinks"):
        evaluation.report_directory(own)


def test_previous_interruption_keeps_unresolved_tail_and_never_claims_a_final_result():
    prior = evaluation.previous_interrupted_run()
    assert prior["archive_sha256"] == evaluation.PREVIOUS_ZIP_SHA256
    assert prior["last_verified_completed_records"] == 84
    assert prior["last_verified_completed_record_calls"] == 744
    assert prior["last_verified_journal_entries"] == 755
    assert prior["last_verified_completed_journal_entries"] == 754
    assert prior["last_verified_started_journal_entries"] == 1
    assert prior["unresolved_call"]["call_index"] == 11
    assert prior["unresolved_call"]["status"] == "started"
    assert not prior["automatic_resume_allowed"] and not prior["old_outputs_pooled_or_replaced"]


def test_changed_history_cannot_silently_authorize_replay(tmp_path, monkeypatch):
    for name in evaluation.HISTORY:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    path = tmp_path / "reports/cuda-eval/interruption.json"
    changed = json.loads(path.read_text())
    changed["automatic_model_replay_allowed"] = True
    path.write_text(json.dumps(changed))
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    with pytest.raises(AssertionError):
        evaluation.previous_interrupted_run()


def test_preflight_rejects_changed_new_cases_with_stale_data_and_reference_receipts(tmp_path, monkeypatch):
    for name in evaluation.source_names():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"eos_token_id": 151645, "pad_token_id": 151643}))
    path = tmp_path / "training/cuda_eval2/datasets/cases.json"
    cases = json.loads(path.read_text())
    cases[0]["initial_message"] += " Additional wording authored only in this corruption fixture."
    path.write_text(json.dumps(cases))
    original_receipt = (tmp_path / "reports/cuda-eval2/data-summary.json").read_bytes()
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(fixtures, "DATA", path.parent)
    with pytest.raises(AssertionError):
        evaluation.cpu_preflight(base, tmp_path / "reports/cuda-eval2")
    assert (tmp_path / "reports/cuda-eval2/data-summary.json").read_bytes() == original_receipt
    assert not (tmp_path / "reports/cuda-eval2/analysis-plan.json").exists()


def test_source_allowlist_binds_both_provenance_and_new_data_without_changing_old_helpers():
    names = evaluation.source_names()
    assert set(frozen.source_names()) <= set(names)
    assert set(evaluation.HISTORY) <= set(names)
    assert {"training/cuda_eval2/run.py", "training/cuda_eval2/evaluation.py"} <= set(names)
    assert "training/cuda_eval2/datasets/cases.json" in names
    assert f"{evaluation.TRAINING_PROOF}/verification.json" in names
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
    assert not any(name.endswith((".safetensors", ".sqlite")) for name in names)


def test_plan_rejects_existing_issued_journal_before_reading_or_overwriting_any_input(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    destination = tmp_path / "reports/cuda-eval2"
    destination.mkdir(parents=True)
    path = destination / "call-journal.json"
    original = b'[{"status":"started"}]\n'
    path.write_bytes(original)
    monkeypatch.setattr(evaluation, "cpu_preflight", lambda *a: pytest.fail("Must reject before preflight"))
    with pytest.raises(FileExistsError, match="issued call"):
        evaluation.create_plan(destination)
    assert path.read_bytes() == original and not (destination / "analysis-plan.json").exists()


def checkpoint_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    destination = tmp_path / "reports/cuda-eval2"
    plan = {
        "run_id": evaluation.RUN_ID,
        "created_at_utc": "2026-10-29T00:00:00+00:00",
        "previous_interrupted_run": {"archive_sha256": evaluation.PREVIOUS_ZIP_SHA256},
    }
    identity = {"run_id": evaluation.RUN_ID, "sealed_at_utc": "2026-10-29T00:00:01+00:00"}
    evaluation.write_atomic(destination / "analysis-plan.json", plan)
    evaluation.write_atomic(destination / "artifact-identity.json", identity)
    report = {
        "run_id": evaluation.RUN_ID,
        "previous_interrupted_run": plan["previous_interrupted_run"],
        "analysis_plan_sha256": evaluation.sha(destination / "analysis-plan.json"),
        "artifact_identity_sha256": evaluation.sha(destination / "artifact-identity.json"),
        "records": [],
    }
    return destination, plan, identity, report


def test_pending_case_or_original_run_cannot_be_replayed_even_without_observed_output(tmp_path, monkeypatch):
    destination, plan, identity, report = checkpoint_fixture(tmp_path, monkeypatch)
    report["pending_case"] = {"variant": evaluation.ARMS[1], "case_id": "new-fixture", "call_index": 11}
    path = destination / "evaluation-partial.json"
    evaluation.write_atomic(path, report)
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="refusing to repeat"):
        run.checkpoint_state(destination, plan, identity, resume=True)
    assert path.read_bytes() == before
    report["run_id"] = "resolvedesk-cuda-eval-001"
    evaluation.write_atomic(path, report)
    with pytest.raises(RuntimeError, match="another run"):
        run.checkpoint_state(destination, plan, identity, resume=True)


def test_started_and_completed_orphan_calls_and_reordering_cannot_resume(tmp_path):
    plan = {"case_ids": ["fixture-a", "fixture-b"]}
    journal = evaluation.CallJournal(tmp_path / "calls.json")
    report = {"records": []}
    entry = journal.start(evaluation.ARMS[0], "fixture-a", 1, [{"role": "user", "content": "fixture"}])
    with pytest.raises(RuntimeError, match="ambiguous"):
        run.validate_completed_prefix(plan, report, journal)
    journal.finish(entry, "invalid raw", {"provider": "cpu_fixture"})
    with pytest.raises(RuntimeError, match="Orphan"):
        run.validate_completed_prefix(plan, report, journal)
    observation = {"raw": "invalid raw", "timing": {"provider": "cpu_fixture"}}
    report["records"] = [{"variant": evaluation.ARMS[0], "case_id": "fixture-a", "trace": [observation]}]
    order, actual = run.validate_completed_prefix(plan, report, journal)
    assert actual == order[:1]
    report["records"][0]["case_id"] = "fixture-b"
    with pytest.raises(RuntimeError, match="order prefix"):
        run.validate_completed_prefix(plan, report, journal)


def test_interrupted_model_call_is_retained_and_budget_has_no_implicit_retries(tmp_path):
    class Interrupted:
        def generate(self, prompt):
            raise RuntimeError("fixture interruption after durable start")

    journal = evaluation.CallJournal(tmp_path / "calls.json")
    provider = RecordingProvider(Interrupted(), journal, evaluation.ARMS[0], "new-fixture")
    with pytest.raises(RuntimeError, match="fixture interruption"):
        provider.generate([{"role": "user", "content": "fixture"}])
    assert provider.calls == 1 and len(journal.entries) == 1
    assert journal.entries[0]["status"] == "started" and "raw" not in journal.entries[0]
    assert json.loads(journal.path.read_text()) == journal.entries
    provider.calls = 12
    with pytest.raises(RuntimeError, match="budget"):
        provider.generate([])
    assert len(journal.entries) == 1


def test_run_requires_explicit_free_runtime_attestation_before_any_reads_or_imports(tmp_path):
    with pytest.raises(PermissionError, match="free-gpu-confirmed"):
        run.evaluate(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)
    assert not list(tmp_path.iterdir())


def test_default_cli_is_cpu_preflight_and_plan_and_seal_do_not_fall_through_to_run(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "evaluate", lambda *a: pytest.fail("No implicit model run"))
    monkeypatch.setattr(run, "cpu_preflight", lambda *a: calls.append("preflight") or {})
    monkeypatch.setattr(
        run, "create_plan", lambda *a: calls.append("plan") or {"scope": "CPU fixture plan"},
    )
    monkeypatch.setattr(run, "seal_artifacts", lambda *a: calls.append("seal") or {})
    for arguments in ([], ["plan"], ["seal"]):
        monkeypatch.setattr(sys, "argv", ["fixture", *arguments])
        run.main()
    assert calls == ["preflight", "plan", "seal"]


def test_journal_rejects_changed_raw_output_without_repairing_the_original_evidence(tmp_path):
    journal = evaluation.CallJournal(tmp_path / "calls.json")
    entry = journal.start(evaluation.ARMS[0], "fixture", 1, [{"role": "user", "content": "fixture"}])
    raw = '{"tool":"record_request","arguments":{"reason":"not an enum"}}'
    journal.finish(entry, raw, {"provider": "cpu_fixture"})
    row = {
        "variant": evaluation.ARMS[0], "case_id": "fixture",
        "trace": [{"raw": raw, "timing": entry["timing"]}],
    }
    evaluation.reconcile_journal(journal.entries, [row])
    damaged = copy.deepcopy(row)
    damaged["trace"][0]["raw"] = raw.replace("not an enum", "unknown")
    with pytest.raises(RuntimeError, match="differs"):
        evaluation.reconcile_journal(journal.entries, [damaged])
    assert json.loads(journal.path.read_text())[0]["raw"] == raw
