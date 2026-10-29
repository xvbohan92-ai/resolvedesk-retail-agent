"""CPU fake-provider contracts for independent replay; never actual CUDA/model evidence."""

import copy
import hashlib
import json

import pytest

from resolve_desk.resolution_v7 import ResolutionStore, messages
from training.cuda_eval.fixtures import load_cases
from training.cuda_eval.metrics import ARMS, raw_diagnostics
from training.cuda_eval.prepare import ReferenceProvider
from training.cuda_eval.provider import generation_settings
from training.cuda_eval.verify_results import canonical, token_hash, verify_journal, verify_record
from training.v7.evaluation import run_case


class CPUContractProvider:
    """Synthetic timings exercise a schema only; no working report is written."""

    def __init__(self, case):
        self.reference, self.calls = ReferenceProvider(case), 0
        self.generation = generation_settings({"eos_token_id": 2, "pad_token_id": 0, "bos_token_id": 1})

    def generate(self, prompt):
        self.calls += 1
        raw, _ = self.reference.generate(prompt)
        if self.calls == 1:
            action = json.loads(raw)
            action["arguments"]["reason"] = "CPU invalid enum fixture"
            raw = json.dumps(action)
        return raw, {
            "provider": "torch_cuda",
            "variant": ARMS[1],
            "adapter_sha256": "CPU adapter marker",
            "workflow": "v7",
            "cloud_compute_used": True,
            "gradient_checkpointing": False,
            "json_schema_constrained": False,
            "generation_settings": self.generation,
            "model_kwargs": {"logits_to_keep": 1},
            "input_token_ids": [1],
            "input_tokens": 1,
            "input_token_ids_sha256": token_hash([1]),
            "generated_token_ids": [2],
            "output_tokens": 1,
            "generated_token_ids_sha256": token_hash([2]),
            "latency_ms": 0,
        }


def record_fixture(tmp_path):
    case = next(c for c in load_cases("test") if c["id"] == "cuda-fresh-test-exchange-00")
    provider = CPUContractProvider(case)
    store = ResolutionStore(tmp_path / "cpu-contract.sqlite")
    record = {"variant": ARMS[1], **run_case(store, case, provider, messages)}
    identity = {"before_adapter_sha256": "CPU prior marker", "after_adapter_sha256": "CPU adapter marker"}
    return case, record, identity, provider.generation


def test_independent_replay_preserves_first_enum_failure_and_model_repair(tmp_path):
    case, record, identity, generation = record_fixture(tmp_path)
    assert record["task_success"] and record["reason_enum_recovered"]
    assert raw_diagnostics(record, case)["initial_reason_enum_error"]
    original = copy.deepcopy(record)
    prompts = verify_record(record, case, identity, generation)
    assert len(prompts) == record["model_calls"] and record == original
    assert record["intent_initial"]["reason"] == "CPU invalid enum fixture"


def test_tampered_token_count_fingerprint_is_rejected_even_with_correct_business_result(tmp_path):
    case, record, identity, generation = record_fixture(tmp_path)
    record["trace"][0]["timing"]["input_token_ids"] = [99]
    with pytest.raises(AssertionError):
        verify_record(record, case, identity, generation)


def test_journal_rejects_hidden_extra_calls_and_a_different_actual_prompt(tmp_path):
    case, record, identity, generation = record_fixture(tmp_path)
    prompts = verify_record(record, case, identity, generation)
    identity["sealed_at_utc"] = "2026-01-01T00:00:01+00:00"
    plan = {"created_at_utc": "2026-01-01T00:00:00+00:00"}
    journal, expected = [], {}
    for index, (prompt, observation) in enumerate(zip(prompts, record["trace"], strict=True), 1):
        expected[record["variant"], record["case_id"], index] = prompt
        journal.append(
            {
                "status": "completed",
                "variant": record["variant"],
                "case_id": record["case_id"],
                "call_index": index,
                "raw": observation["raw"],
                "timing": observation["timing"],
                "request_messages": prompt,
                "request_sha256": hashlib.sha256(canonical(prompt).encode()).hexdigest(),
                "started_at_utc": "2026-01-01T00:00:02+00:00",
                "completed_at_utc": "2026-01-01T00:00:03+00:00",
            }
        )
    verify_journal([record], journal, plan, identity, expected)
    with pytest.raises(AssertionError):
        verify_journal([record], [*journal, copy.deepcopy(journal[-1])], plan, identity, expected)
    altered = copy.deepcopy(journal)
    altered[0]["request_messages"][1]["content"] = "CPU forged prompt fixture"
    altered[0]["request_sha256"] = hashlib.sha256(
        canonical(altered[0]["request_messages"]).encode()
    ).hexdigest()
    with pytest.raises(AssertionError):
        verify_journal([record], altered, plan, identity, expected)
