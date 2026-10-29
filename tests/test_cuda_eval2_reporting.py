"""CPU-only reporting fixtures; no candidate output, GPU or working-report writes.

Saved authored oracle receipts test the read-model contract. Mutated records
below are explicitly artificial scoring fixtures, never model measurements.
"""

import copy
import hashlib
import json

import pytest

from resolve_desk.evidence_cuda2 import validated_report
from resolve_desk.resolution_v7 import ResolutionStore, messages
from training.common import ROOT
from training.cuda_eval.prepare import ReferenceProvider
from training.cuda_eval.provider import generation_settings
from training.cuda_eval2 import metrics
from training.cuda_eval2.fixtures import load_cases
from training.cuda_eval2.metrics import ARMS, raw_diagnostics
from training.cuda_eval2.verify_results import canonical, token_hash, verify_journal, verify_record
from training.v7.evaluation import run_case
from training.v7.protocol_audit import COUNTS


@pytest.fixture(scope="module")
def cpu_reference():
    cases = {case["id"]: case for case in load_cases("test")}
    receipt = json.loads((ROOT / "reports/cuda-eval2/reference-audit.json").read_text())
    records = [row["reference"] for row in receipt["records"]]
    assert len(cases) == len(records) == 48
    assert all(
        observation["timing"]["provider"] == "cpu_reference_oracle"
        for record in records
        for observation in record["trace"]
    )
    return cases, records, receipt


def test_initial_raw_metrics_publish_the_48_task_ui_denominator(cpu_reference):
    cases, records, _ = cpu_reference
    result = metrics.summarize(records, cases)
    assert result["intent_tasks"] == result["tasks"] == 48
    assert result["intent_joint_correct"] == 48
    assert result["authorized_tasks"] == result["true_permission_yes"] == 12
    assert result["non_authorized_tasks"] == 36 and result["unsupported_permission_yes"] == 0
    assert result["initial_enum_error_tasks"] == result["initial_raw_parse_failures"] == 0


def test_invalid_first_reason_survives_a_later_accepted_repair(cpu_reference):
    cases, records, _ = cpu_reference
    changed = copy.deepcopy(records)
    record = changed[0]
    correct = copy.deepcopy(record["trace"][0])
    first = json.loads(correct["raw"])
    first["arguments"]["reason"] = "my schedule changed, please cancel"
    record["trace"].insert(0, {**correct, "raw": json.dumps(first)})
    record["model_calls"] += 1
    record["enum_error_attempts"] = record["reason_enum_errors"] = 1
    record["enum_recovered"] = record["reason_enum_recovered"] = True
    # Deliberately present a correct accepted projection: raw scoring must ignore it.
    record["intent_initial"] = dict(cases[record["case_id"]]["intent_initial"])
    assert record["task_success"]
    raw = metrics.raw_diagnostics(record, cases[record["case_id"]])
    assert raw["intent_initial"]["reason"] == "my schedule changed, please cancel"
    assert raw["initial_enum_errors"] == {"reason": "my schedule changed, please cancel"}
    assert not raw["initial_raw_parse_failure"] and not raw["intent_field_correct"]["reason"]
    result = metrics.summarize(changed, cases)
    assert result["task_success"] == 48 and result["intent_joint_correct"] == 47
    assert result["initial_reason_enum_error_tasks"] == result["initial_enum_error_tasks"] == 1
    assert result["reason_enum_recovered_tasks"] == result["enum_recovered_tasks"] == 1


@pytest.mark.parametrize("raw", ["not JSON", '{"tool":"record_request","arguments":{}}'])
def test_missing_first_parse_does_not_borrow_a_later_permission_prediction(cpu_reference, raw):
    cases, records, _ = cpu_reference
    changed = copy.deepcopy(records)
    record = next(
        row for row in changed if cases[row["case_id"]]["intent_initial"]["fallback_return"] == "yes"
    )
    record["trace"][0]["raw"] = raw
    diagnostic = metrics.raw_diagnostics(record, cases[record["case_id"]])
    assert diagnostic["permission_authorized"] and not diagnostic["permission_yes"]
    assert diagnostic["initial_raw_parse_failure"] and diagnostic["intent_initial"] is None
    result = metrics.summarize(changed, cases)
    assert result["authorized_tasks"] == 12 and result["true_permission_yes"] == 11
    assert result["non_authorized_tasks"] == 36 and result["initial_raw_parse_failures"] == 1


def test_18_required_tasks_20_fields_and_declared_choices_differ_from_transactions(cpu_reference):
    cases, records, _ = cpu_reference
    result = metrics.summarize(records, cases)
    assert result["required_clarification_tasks"] == result["required_clarification_tasks_obtained"] == 18
    assert result["required_clarification_fields"] == result["required_clarification_fields_obtained"] == 20
    assert result["declared_accept_choice_tasks"] == result["declared_decline_choice_tasks"] == 24
    assert result["accepted_confirmation_tasks"] == result["declined_confirmation_tasks"] == 16
    handoffs = [row for row in records if cases[row["case_id"]]["expected"]["kind"] == "handoff"]
    subset = metrics.summarize(handoffs, cases)
    assert subset["declared_accept_choice_tasks"] == subset["declared_decline_choice_tasks"] == 8
    assert subset["accepted_confirmation_tasks"] == subset["declined_confirmation_tasks"] == 0


def test_paired_failures_expose_case_id_and_boolean_results_for_both_arms(cpu_reference):
    cases, references, receipt = cpu_reference
    records = [{**copy.deepcopy(row), "variant": arm} for arm in metrics.ARMS for row in references]
    ids = list(cases)
    for row in records:
        if (
            row["case_id"] == ids[0]
            and row["variant"] == metrics.ARMS[1]
            or row["case_id"] == ids[1]
            and row["variant"] == metrics.ARMS[0]
            or row["case_id"] == ids[2]
        ):
            row["task_success"] = False
    protocol = {
        "records": [
            {**copy.deepcopy(row["protocol"]), "variant": arm}
            for arm in metrics.ARMS
            for row in receipt["records"]
        ]
    }
    values = metrics.derived_values({"records": records}, cases, protocol)
    assert values["paired"] == {"both_pass": 45, "both_fail": 1, "improved": 1, "regressed": 1}
    index = {row["case_id"]: row for row in values["failures"]}
    assert set(index) == set(ids[:3])
    assert index[ids[0]] == {"case_id": ids[0], metrics.ARMS[0]: True, metrics.ARMS[1]: False}
    assert index[ids[1]] == {"case_id": ids[1], metrics.ARMS[0]: False, metrics.ARMS[1]: True}
    assert index[ids[2]] == {"case_id": ids[2], metrics.ARMS[0]: False, metrics.ARMS[1]: False}
    assert all(type(row[arm]) is bool for row in values["failures"] for arm in metrics.ARMS)


def test_protocol_blocked_raw_attempt_is_not_a_created_proposal_or_commit():
    def row(arm, created):
        return {
            "variant": arm,
            "case_id": "explicit-cpu-protocol-fixture",
            "audited": True,
            **dict.fromkeys(COUNTS, 0),
            "raw_return_attempts": 1,
            "return_attempts_without_scoped_yes": 1,
            "return_proposals": int(created),
            "return_proposals_without_scoped_yes": int(created),
            "attempts": [
                {
                    "raw_return_attempt": True,
                    "scoped_yes_verified": False,
                    "return_proposal_created": created,
                    "backend_error": None if created else "blocked_cpu_fixture",
                }
            ],
        }

    result = metrics.protocol_summary([row(metrics.ARMS[0], False), row(metrics.ARMS[1], True)])
    before, after = (result[arm] for arm in metrics.ARMS)
    assert before["rejected_return_attempts_without_scoped_yes"] == 1
    assert (
        before["return_proposals_without_scoped_yes"] == before["committed_returns_without_scoped_yes"] == 0
    )
    assert after["rejected_return_attempts_without_scoped_yes"] == 0
    assert after["return_proposals_without_scoped_yes"] == 1
    assert (
        after["committed_returns_without_scoped_yes"] == after["final_returns_without_verified_commit"] == 0
    )


def test_empty_evidence_root_stays_pending_without_creating_reports(tmp_path):
    result = validated_report(tmp_path)
    assert result["available"] is False and result["experiment"] == "cuda_eval2"
    assert "96-record" in result["reason"]
    assert list(tmp_path.iterdir()) == []


def test_pending_writer_cannot_replace_a_document_or_create_figure_evidence(tmp_path, monkeypatch):
    from training.cuda_eval2 import write_summary

    document = tmp_path / "docs/cuda-evaluation2.md"
    document.parent.mkdir()
    document.write_text("Awaiting a complete new run; no measured scores.\n")
    before = document.read_bytes()
    monkeypatch.setattr(write_summary, "ROOT", tmp_path)
    with pytest.raises(AssertionError, match="96-record"):
        write_summary.main()
    assert document.read_bytes() == before
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == [
        "docs",
        "docs/cuda-evaluation2.md",
    ]


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
    case = next(c for c in load_cases("test") if c["id"] == "cuda-fresh2-test-exchange-00")
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


def test_token_replay_keeps_special_text_and_removes_only_the_terminal_eos(tmp_path):
    case, record, identity, generation = record_fixture(tmp_path)
    prompts = verify_record(record, case, identity, generation)
    encoded, decoded = {}, {}
    event_observations = [event["detail"] for event in record["events"] if event["kind"] == "model_tool"]
    for index, (prompt, observation, event) in enumerate(
        zip(prompts, record["trace"], event_observations, strict=True), 3
    ):
        # Unique artificial token IDs exercise decode bookkeeping, not a real tokenizer/model.
        encoded[canonical(prompt)] = [index + 100]
        decoded[index,] = observation["raw"]
        decoded[index, 2] = observation["raw"] + "<eos>"
        timing = {
            **observation["timing"],
            "input_token_ids": [index + 100],
            "input_token_ids_sha256": token_hash([index + 100]),
            "generated_token_ids": [index, 2],
            "generated_token_ids_sha256": token_hash([index, 2]),
            "output_tokens": 2,
            "terminal_eos_removed_for_text": True,
            "finish_reason": "eos",
            "decoded_with_terminal_special_tokens": decoded[index, 2],
        }
        observation["timing"] = event["timing"] = timing
    record["output_tokens"] = 2 * len(record["trace"])

    class CPUBookkeepingTokenizer:
        def apply_chat_template(self, prompt, **kwargs):
            assert kwargs == {
                "enable_thinking": False,
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": False,
            }
            return encoded[canonical(prompt)]

        def decode(self, token_ids, **kwargs):
            assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
            return decoded[tuple(token_ids)]

    tokenizer = CPUBookkeepingTokenizer()
    assert verify_record(record, case, identity, generation, tokenizer, (2,)) == prompts
    record["trace"][0]["timing"]["decoded_with_terminal_special_tokens"] = "silently filtered CPU fixture"
    with pytest.raises(AssertionError):
        verify_record(record, case, identity, generation, tokenizer, (2,))


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


def test_new_reporting_paths_do_not_redirect_the_interrupted_run():
    from training.cuda_eval import metrics as old_metrics
    from training.cuda_eval import verify_results as old_verifier
    from training.cuda_eval2 import verify_results as new_verifier
    from training.cuda_eval2 import visualize as new_visualizer
    from training.cuda_eval2 import write_summary as new_writer

    assert old_metrics.REPORTS == old_verifier.REPORTS == ROOT / "reports/cuda-eval"
    assert old_metrics.FIGURES == ROOT / "docs/figures/cuda-eval"
    assert metrics.REPORTS == new_verifier.REPORTS == new_visualizer.REPORTS == new_writer.REPORTS
    assert metrics.REPORTS == ROOT / "reports/cuda-eval2"
    assert metrics.FIGURES == new_verifier.FIGURES == new_visualizer.FIGURES == new_writer.FIGURES
    assert metrics.FIGURES == ROOT / "docs/figures/cuda-eval2"


def test_publication_guard_rejects_incomplete_duplicate_old_and_out_of_budget_records(cpu_reference):
    from training.cuda_eval2.verify_results import require_complete_records

    cases, references, _ = cpu_reference
    report = {
        "run_id": metrics.RUN_ID,
        "status": "complete",
        "provider": "torch_cuda",
        "records": [{**copy.deepcopy(row), "variant": arm} for arm in metrics.ARMS for row in references],
    }
    require_complete_records(report, list(cases))
    for mutation in ("partial", "missing", "duplicate", "old_run", "old_case", "pending", "extra_call"):
        changed = copy.deepcopy(report)
        if mutation == "partial":
            changed["status"] = "partial"
        elif mutation == "missing":
            changed["records"].pop()
        elif mutation == "duplicate":
            changed["records"][-1] = copy.deepcopy(changed["records"][-2])
        elif mutation == "old_run":
            changed["run_id"] = "resolvedesk-cuda-eval-001"
        elif mutation == "old_case":
            changed["records"][0]["case_id"] = "cuda-fresh-test-cancel-00"
        elif mutation == "pending":
            changed["pending_case"] = {"case_id": list(cases)[-1]}
        else:
            changed["records"][0]["model_calls"] = 13
            changed["records"][0]["trace"] = [None] * 13
        with pytest.raises(AssertionError):
            require_complete_records(changed, list(cases))


def test_run_identity_binds_retained_interruption_and_rejects_pooling():
    from training.cuda_eval2.evaluation import previous_interrupted_run
    from training.cuda_eval2.verify_results import verify_run_identity

    # Actual retained metadata only; this makes no model call and writes nothing.
    previous = previous_interrupted_run()
    record = {
        "run_id": metrics.RUN_ID,
        "previous_interrupted_run": previous,
        "old_outputs_pooled_or_replaced": False,
        "training_repeated_or_modified": False,
        "compute_authorization": {
            "free_runtime_operator_confirmed": True,
            "free_gpu_only": True,
            "paid_fallback_allowed": False,
            "automatic_provisioning": False,
        },
        "after_checkpoint": {
            "adapter_sha256": "33fe5c505ddf08cd8cb4032dbd9d350fa205134417ec0f1f90648f4af658e053",
        },
        "after_adapter_sha256": "33fe5c505ddf08cd8cb4032dbd9d350fa205134417ec0f1f90648f4af658e053",
        "before_initializer_sha256": "CPU-only predecessor marker",
        "before_adapter_sha256": "CPU-only predecessor marker",
    }
    verify_run_identity(record, copy.deepcopy(record), copy.deepcopy(record))
    for key, value in (
        ("last_verified_completed_records", 96),
        ("automatic_resume_allowed", True),
        ("interruption_sha256", "forged CPU hash"),
    ):
        changed = copy.deepcopy(record)
        changed["previous_interrupted_run"][key] = value
        with pytest.raises(AssertionError):
            verify_run_identity(changed, copy.deepcopy(changed), copy.deepcopy(changed))
    changed = copy.deepcopy(record)
    changed["old_outputs_pooled_or_replaced"] = True
    with pytest.raises(AssertionError):
        verify_run_identity(changed, copy.deepcopy(changed), copy.deepcopy(changed))

    changed = copy.deepcopy(record)
    changed["compute_authorization"]["paid_fallback_allowed"] = True
    with pytest.raises(AssertionError):
        verify_run_identity(changed, copy.deepcopy(changed), copy.deepcopy(changed))


@pytest.fixture
def cpu_budget_authorization(tmp_path):
    """Artificial CPU metadata only: no actual billing, authorization or model evidence."""
    from training.common import sha

    folder = tmp_path / "reports/cuda-eval2"
    folder.mkdir(parents=True)
    source_names = [
        "training/cuda_eval2/evaluation.py",
        "training/cuda_eval2/budgeted_run.py",
        "tests/test_cuda_eval2_budget.py",
    ]
    for name in source_names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# CPU metadata fixture, not actual evaluator source\n")
    plan = {
        "run_id": metrics.RUN_ID,
        "compute_authorization": {
            "free_gpu_only": True,
            "paid_fallback_allowed": False,
            "automatic_provisioning": False,
        },
        "source_inputs_sha256": {source_names[0]: sha(tmp_path / source_names[0])},
        "cases_sha256": "CPU cases marker",
        "previous_interrupted_run": {"scope": "CPU retained-provenance contract marker"},
        "generation": {"max_new_tokens": 512, "do_sample": False},
        "inference_model_kwargs": {"logits_to_keep": 1},
        "case_ids": ["CPU case-order marker"],
    }
    identity = {
        "run_id": metrics.RUN_ID,
        "sealed_at_utc": "2026-01-01T00:00:01+00:00",
        "before_adapter_sha256": "CPU before marker",
        "after_adapter_sha256": "CPU after marker",
    }
    (folder / "analysis-plan.json").write_text(json.dumps(plan))
    (folder / "artifact-identity.json").write_text(json.dumps(identity))
    addendum = {
        "schema_version": 1,
        "run_id": metrics.RUN_ID,
        "created_at_utc": "2026-01-01T00:00:02+00:00",
        "authorization_mode": "metered_existing_units",
        "analysis_plan_sha256": sha(folder / "analysis-plan.json"),
        "artifact_identity_sha256": sha(folder / "artifact-identity.json"),
        "before_adapter_sha256": identity["before_adapter_sha256"],
        "after_adapter_sha256": identity["after_adapter_sha256"],
        "cases_sha256": plan["cases_sha256"],
        "source_inputs_sha256": plan["source_inputs_sha256"],
        "supplementary_sources_sha256": {name: sha(tmp_path / name) for name in source_names[1:]},
        "previous_interrupted_run": plan["previous_interrupted_run"],
        "supersedes_only": {
            "field": "compute_authorization.free_gpu_only",
            "original_value": True,
            "effective_authorization": "metered_existing_units",
        },
        "unchanged_protocol": {
            "no_retraining": True,
            "no_checkpoint_selection": True,
            "no_old_run_replay": True,
            "no_old_outputs_pooled": True,
            "T4_only": True,
            "seed": 202610287,
            "total_calls_per_task": 12,
            "generation": copy.deepcopy(plan["generation"]),
            "inference_model_kwargs": copy.deepcopy(plan["inference_model_kwargs"]),
            "case_order": list(plan["case_ids"]),
            "arms": list(metrics.ARMS),
            "controller": "frozen_v7",
            "scoring": "frozen_v7",
        },
        "human_authorization": {
            "explicit": True,
            "cap_compute_units": 10,
            "starting_balance_units": 80,
            "stop_balance_units": 72,
            "hard_floor_balance_units": 70,
            "purchase_allowed": False,
            "renewal_allowed": False,
            "free_allocation_claimed": False,
            "T4_only": True,
            "approval_text": "Synthetic CPU contract fixture, not an actual human approval.",
        },
        "accounting": {
            "automatic_compute_unit_metering": False,
            "billing_stops_when_process_exits": False,
            "owner": "CPU fixture external operator",
            "starting_balance_units": 80,
            "hard_cap_units": 10,
            "hard_cap_balance": 70,
            "operator_stop_balance": 72,
            "reserved_units": 2,
            "observed_units_per_hour": 1.07,
            "projected_units_until_disconnect": 660 / 3600 * 1.07,
            "projection_is_actual_consumption": False,
            "runtime_allocated_at_utc": "2026-01-01T00:00:00+00:00",
            "process_deadline_utc": "2026-01-01T00:10:00+00:00",
            "runtime_disconnect_deadline_utc": "2026-01-01T00:11:00+00:00",
        },
    }
    path = folder / "execution-addendum.json"
    path.write_text(json.dumps(addendum))
    fingerprint = sha(path)
    report = {
        "run_id": metrics.RUN_ID,
        "analysis_plan_sha256": addendum["analysis_plan_sha256"],
        "artifact_identity_sha256": addendum["artifact_identity_sha256"],
        "cases_sha256": plan["cases_sha256"],
        "execution_addendum_sha256": fingerprint,
        "compute_authorization": {
            "authorization_mode": "metered_existing_units",
            "existing_units_operator_confirmed": True,
            "free_runtime_operator_confirmed": False,
            "cap_compute_units": 10,
            "purchases_allowed": False,
            "renewal_allowed": False,
            "execution_addendum_sha256": fingerprint,
            "execution_addendum_path": "reports/cuda-eval2/execution-addendum.json",
        },
        "execution_budget": addendum["accounting"],
        "records": [
            {
                "trace": [
                    {
                        "timing": {
                            "execution_addendum_sha256": fingerprint,
                            "execution_authorization_mode": "metered_existing_units",
                        }
                    }
                ]
            }
        ],
    }
    journal = [
        {
            "run_id": metrics.RUN_ID,
            "execution_addendum_sha256": fingerprint,
            "started_at_utc": "2026-01-01T00:00:03+00:00",
            "completed_at_utc": "2026-01-01T00:00:04+00:00",
        }
    ]
    return tmp_path, report, plan, identity, addendum, journal


def test_metered_authorization_is_additive_truthful_and_read_only(cpu_budget_authorization):
    from training.cuda_eval2.verify_results import verify_execution_authorization

    root, report, plan, identity, _, journal = cpu_budget_authorization
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = verify_execution_authorization(report, plan, identity, root=root, journal=journal)
    assert result["authorization_mode"] == "metered_existing_units"
    assert result["cap_compute_units"] == 10 and result["stop_balance_units"] == 72
    assert result["free_runtime_operator_confirmed"] is result["automatic_metering"] is False
    assert result["purchases_allowed"] is result["renewal_allowed"] is False
    assert "not billing proof" in result["scope"]
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "fault", ["missing", "fake_free", "over_cap", "purchase", "hash", "late", "call_sha"]
)
def test_metered_authorization_rejects_missing_tampered_over_budget_or_late_proof(
    cpu_budget_authorization, fault
):
    from training.cuda_eval2.verify_results import verify_execution_authorization

    root, report, plan, identity, _, journal = cpu_budget_authorization
    if fault == "missing":
        (root / "reports/cuda-eval2/execution-addendum.json").unlink()
    elif fault == "fake_free":
        report["compute_authorization"]["free_runtime_operator_confirmed"] = True
    elif fault == "over_cap":
        report["compute_authorization"]["cap_compute_units"] = 11
    elif fault == "purchase":
        report["compute_authorization"]["purchases_allowed"] = True
    elif fault == "hash":
        (root / "training/cuda_eval2/budgeted_run.py").write_text("# tampered CPU fixture\n")
    elif fault == "late":
        journal[0]["started_at_utc"] = "2026-01-01T00:00:00+00:00"
    else:
        report["records"][0]["trace"][0]["timing"]["execution_addendum_sha256"] = "different call"
    with pytest.raises((AssertionError, FileNotFoundError)):
        verify_execution_authorization(report, plan, identity, root=root, journal=journal)
