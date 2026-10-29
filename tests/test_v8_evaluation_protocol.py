"""New evaluation keeps raw errors, issued-call tails and the actual workflow label."""

import json

import pytest

from training.cuda_eval.evaluation import CallJournal
from training.v8 import cuda_evaluate as evaluation


@pytest.fixture(autouse=True)
def receipt_file(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "REPORTS", tmp_path)
    (tmp_path / "resource-authorization.json").write_text('{"fixture_only":true}')


class FakeProvider:
    def generate(self, prompt):
        return "deliberately invalid raw output", {"workflow": "v7", "output_tokens": 4}


def test_records_raw_once_and_labels_new_workflow_without_repair(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "verify_resource_receipt", lambda: {})
    journal = CallJournal(tmp_path / "journal.json")
    provider = evaluation.RecordedProvider(FakeProvider(), journal, "v8_weights_new_flow", "fixture", "v8")
    prompt = [{"role": "user", "content": "test fixture"}]
    raw, timing = provider.generate(prompt)
    assert raw == "deliberately invalid raw output"
    assert timing["workflow"] == "v8"
    entry = journal.entries[0]
    assert entry["status"] == "completed"
    assert entry["raw"] == raw
    assert entry["request_messages"] == prompt
    assert entry["timing"] == timing
    assert len(journal.entries) == provider.calls == 1


def test_interrupted_call_remains_started_and_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "verify_resource_receipt", lambda: {})

    class Broken:
        def generate(self, _):
            raise ConnectionError("test fixture: completion unknown")

    path = tmp_path / "journal.json"
    provider = evaluation.RecordedProvider(
        Broken(), CallJournal(path), "prior_weights_new_flow", "case", "v8"
    )
    with pytest.raises(ConnectionError):
        provider.generate([{"role": "user", "content": "fixture"}])
    saved = json.loads(path.read_text())
    assert len(saved) == 1 and saved[0]["status"] == "started"
    assert "raw" not in saved[0]
    assert provider.calls == 1


def test_no_hidden_thirteenth_call(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "verify_resource_receipt", lambda: {})
    journal = CallJournal(tmp_path / "journal.json")
    provider = evaluation.RecordedProvider(FakeProvider(), journal, "prior_weights_old_flow", "case", "v7")
    for _ in range(12):
        provider.generate([{"role": "user", "content": "fixture"}])
    with pytest.raises(RuntimeError, match="total12"):
        provider.generate([{"role": "user", "content": "fixture"}])
    assert len(journal.entries) == 12


def test_no_resource_approval_means_no_issued_model_call(tmp_path, monkeypatch):
    def refused():
        raise PermissionError("test fixture: no new budget")

    monkeypatch.setattr(evaluation, "verify_resource_receipt", refused)
    path = tmp_path / "journal.json"
    provider = evaluation.RecordedProvider(
        FakeProvider(), CallJournal(path), "v8_weights_new_flow", "case", "v8"
    )
    with pytest.raises(PermissionError):
        provider.generate([{"role": "user", "content": "fixture"}])
    assert not path.exists()


def test_drifted_approval_is_rejected_before_an_issued_call(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "verify_resource_receipt", lambda: {})
    path = tmp_path / "journal.json"
    provider = evaluation.RecordedProvider(
        FakeProvider(), CallJournal(path), "v8_weights_new_flow", "case", "v8"
    )
    (tmp_path / "resource-authorization.json").write_text('{"different_fixture":true}')
    with pytest.raises(PermissionError, match="changed"):
        provider.generate([{"role": "user", "content": "fixture"}])
    assert not path.exists()
