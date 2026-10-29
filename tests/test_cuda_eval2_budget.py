"""CPU-only authorization overlay tests; no runtime, model, account or cloud calls."""

import ast
import copy
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from training.common import ROOT
from training.cuda_eval2 import budgeted_run as budget
from training.cuda_eval2 import run as frozen


@pytest.fixture
def authorization(tmp_path, monkeypatch):
    for name in ("analysis-plan.json", "artifact-identity.json"):
        (tmp_path / name).write_bytes((ROOT / "reports/cuda-eval2" / name).read_bytes())
    plan = json.loads((tmp_path / "analysis-plan.json").read_text())
    monkeypatch.setattr(budget, "report_directory", lambda p: Path(p))
    monkeypatch.setattr(budget, "verify_plan", lambda p: plan)
    now = datetime.now(UTC)
    options = {
        "existing_units_confirmed": True,
        "authorization_text": "CPU fixture only: scope at most10 existing units, no purchase or renewal",
        "runtime_allocated_at_utc": (now - timedelta(minutes=5)).isoformat(),
        "process_deadline_utc": (now + timedelta(minutes=50)).isoformat(),
        "runtime_disconnect_deadline_utc": (now + timedelta(minutes=60)).isoformat(),
        "observed_units_per_hour": 1.07,
    }
    return tmp_path, options


def test_import_and_cli_help_load_no_gpu_framework():
    code = """
import sys
from training.cuda_eval2 import budgeted_run
assert not any(n in sys.modules for n in ('torch','mlx','numpy','transformers','safetensors'))
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-B", "-m", "training.cuda_eval2.budgeted_run", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "--process-deadline-utc" in result.stdout


def test_authorization_binds_immutable_plan_identity_and_two_overlay_sources(authorization):
    destination, options = authorization
    original = {n: (destination / n).read_bytes() for n in ("analysis-plan.json", "artifact-identity.json")}
    record = budget.authorize(destination, **options)
    assert record["authorization_mode"] == "metered_existing_units"
    assert record["human_authorization"]["cap_compute_units"] == 10
    assert record["human_authorization"]["stop_balance_units"] == 72
    assert not record["human_authorization"]["free_allocation_claimed"]
    assert not record["human_authorization"]["purchase_allowed"]
    assert not record["human_authorization"]["renewal_allowed"]
    assert record["supersedes_only"]["field"] == "compute_authorization.free_gpu_only"
    assert record["supersedes_only"]["original_value"] is True
    assert len(record["source_inputs_sha256"]) == 53
    assert set(record["supplementary_sources_sha256"]) == set(budget.SUPPLEMENTARY_SOURCES)
    assert budget.verify_addendum(destination) == record
    assert original == {n: (destination / n).read_bytes() for n in original}
    assert not any((destination / n).exists() for n in ("evaluation-partial.json", "call-journal.json"))


@pytest.mark.parametrize(
    "change", ["no_human", "empty_text", "no_timezone", "too_much_projection", "expired"]
)
def test_invalid_authorization_never_seals_or_calls_a_model(authorization, change):
    destination, options = authorization
    if change == "no_human":
        options["existing_units_confirmed"] = False
    elif change == "empty_text":
        options["authorization_text"] = " "
    elif change == "no_timezone":
        options["process_deadline_utc"] = "2099-01-01T03:20:00"
    elif change == "too_much_projection":
        options["observed_units_per_hour"] = 100
    else:
        options["process_deadline_utc"] = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    with pytest.raises((PermissionError, ValueError, AssertionError)):
        budget.authorize(destination, **options)
    assert not (destination / budget.ADDENDUM).exists()


def test_existing_pending_or_started_evidence_cannot_gain_a_new_budget(authorization):
    destination, options = authorization
    (destination / "call-journal.json").write_text('[{"status":"started"}]')
    with pytest.raises(FileExistsError, match="issued/attempted"):
        budget.authorize(destination, **options)
    assert not (destination / budget.ADDENDUM).exists()


@pytest.mark.parametrize("change", ["free", "purchase", "reserve", "source", "checkpoint"])
def test_addendum_rejects_drift_without_repair_or_refreeze(authorization, change):
    destination, options = authorization
    record = budget.authorize(destination, **options)
    if change == "free":
        record["human_authorization"]["free_allocation_claimed"] = True
    elif change == "purchase":
        record["human_authorization"]["purchase_allowed"] = True
    elif change == "reserve":
        record["accounting"]["operator_stop_balance"] = 70
    elif change == "source":
        record["source_inputs_sha256"].pop(next(iter(record["source_inputs_sha256"])))
    else:
        record["after_adapter_sha256"] = "0" * 64
    (destination / budget.ADDENDUM).write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        budget.verify_addendum(destination)


def test_t4_probe_is_bounded_and_never_accepts_a_bigger_paid_accelerator(monkeypatch):
    calls = []

    def fake(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="Tesla T4\n")

    monkeypatch.setattr(budget.subprocess, "run", fake)
    assert budget.probe_t4()["name"] == "Tesla T4"
    assert calls[0][1]["timeout"] == 10 and calls[0][1]["capture_output"]
    monkeypatch.setattr(
        budget.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="NVIDIA A100\n"),
    )
    with pytest.raises(RuntimeError, match="T4-only"):
        budget.probe_t4()


def test_raw_rejected_output_and_started_calls_keep_authorization_hash(tmp_path):
    class Queue:
        calls = 0

        def generate(self, prompt):
            self.calls += 1
            return '{"reason":"invalid enum"}', {"provider": "cpu_fixture"}

    queue = Queue()
    sha = "b" * 64
    future = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
    provider = budget.BudgetProvider(queue, sha, future)
    journal = budget.BudgetJournal(tmp_path / "calls.json", sha)
    recorded = budget.BudgetRecordingProvider(provider, journal, budget.ARMS[0], "cpu-fixture")
    raw, timing = recorded.generate([{"role": "user", "content": "CPU fixture only"}])
    assert raw == '{"reason":"invalid enum"}' and queue.calls == recorded.calls == 1
    entry = journal.entries[0]
    assert entry["run_id"] == budget.RUN_ID and entry["execution_addendum_sha256"] == sha
    assert (
        timing["execution_addendum_sha256"] == sha and timing["execution_authorization_mode"] == budget.MODE
    )
    assert entry["status"] == "completed" and entry["raw"] == raw
    with pytest.raises(RuntimeError, match="another authorization"):
        budget.BudgetJournal(journal.path, "c" * 64)


def test_deadline_denies_new_calls_before_journal_start_or_retry(tmp_path):
    class Never:
        def generate(self, prompt):
            pytest.fail("Expired execution must never call a provider")

    deadline = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    provider = budget.BudgetProvider(Never(), "b" * 64, deadline)
    journal = budget.BudgetJournal(tmp_path / "calls.json", "b" * 64)
    recorded = budget.BudgetRecordingProvider(provider, journal, budget.ARMS[0], "cpu-fixture")
    with pytest.raises(TimeoutError, match="deadline"):
        recorded.generate([])
    assert recorded.calls == 0 and journal.entries == [] and not journal.path.exists()


def test_process_deadline_receipt_truthfully_does_not_claim_cloud_billing_stopped(tmp_path):
    exits = []
    budget.deadline_callback(tmp_path, "b" * 64, exit_process=exits.append)()
    record = json.loads((tmp_path / "process-deadline-stop.json").read_text())
    assert exits == [124] and record["status"] == "process_deadline_reached"
    assert not record["billing_stopped"] and not record["pending_calls_replay_allowed"]


def test_runtime_explicit_existing_unit_confirmation_is_required_before_hardware_or_model(tmp_path):
    with pytest.raises(PermissionError, match="existing-units-confirmed"):
        budget.execute(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)
    assert not list(tmp_path.iterdir())


def test_core_model_calls_seed_order_and_scoring_match_frozen_runtime_exactly():
    def expressions(path, name):
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        wanted = {"load_model", "install_final_adapter", "TorchProvider", "run_case", "ResolutionStore"}
        result = []
        for call in ast.walk(function):
            if isinstance(call, ast.Call):
                if isinstance(call.func, ast.Name) and call.func.id in wanted:
                    result.append(ast.unparse(call))
                elif isinstance(call.func, ast.Attribute) and call.func.attr == "manual_seed":
                    result.append(ast.unparse(call))
        return sorted(result)

    actual = expressions(Path(budget.__file__), "_evaluate")
    assert actual == expressions(Path(frozen.__file__), "evaluate")
    assert "torch.manual_seed(202610287)" in actual
    assert any("max_steps=12" in value for value in actual)
    assert budget.TorchProvider is frozen.TorchProvider


def test_failed_generation_retains_an_unresolved_call_instead_of_replaying(tmp_path):
    class Failure:
        def generate(self, prompt):
            raise RuntimeError("CPU fixture generation failure")

    journal = budget.BudgetJournal(tmp_path / "calls.json", "b" * 64)
    provider = budget.BudgetProvider(
        Failure(),
        "b" * 64,
        (datetime.now(UTC) + timedelta(minutes=2)).isoformat(),
    )
    wrapped = budget.BudgetRecordingProvider(provider, journal, budget.ARMS[0], "fixture")
    with pytest.raises(RuntimeError, match="generation failure"):
        wrapped.generate([])
    saved = copy.deepcopy(journal.entries)
    assert len(saved) == 1 and saved[0]["status"] == "started" and "raw" not in saved[0]
    with pytest.raises(RuntimeError, match="ambiguous"):
        frozen.validate_completed_prefix({"case_ids": ["fixture"]}, {"records": []}, journal)
    assert json.loads(journal.path.read_text()) == saved
