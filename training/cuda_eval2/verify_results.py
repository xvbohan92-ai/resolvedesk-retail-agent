"""Independent CPU raw-action, prompt-token and scoped-consent reconciliation; no LLM."""

import argparse
import copy
import csv
import json
import math
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from resolve_desk.resolution_v7 import ResolutionStore, messages
from training.common import ROOT, sha
from training.cuda_eval2.metrics import (
    ARMS,
    FIGURE_NAMES,
    FIGURES,
    REPORTS,
    RUN_ID,
    derived_values,
    protocol_summary,
)
from training.cuda_torch.verify_report import verify as verify_training
from training.v7.protocol_audit import COUNTS
from training.v7.protocol_audit import audit_record as audit_consent
from training.v7.verify_results import audit_record as audit_business


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def token_hash(ids):
    import hashlib

    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def verify_execution_authorization(report, plan, identity, *, root=ROOT, journal=None):
    """Verify truthful resource authorization separately from the immutable scoring plan."""
    declared, original = report["compute_authorization"], plan["compute_authorization"]
    assert original["free_gpu_only"] is True
    assert original["paid_fallback_allowed"] is original["automatic_provisioning"] is False
    mode = declared.get("authorization_mode")
    referenced = report.get("execution_addendum_sha256")
    if mode != "metered_existing_units" and not referenced:
        assert mode in (None, "free_gpu")
        assert declared["free_runtime_operator_confirmed"] is True
        assert declared["paid_fallback_allowed"] is declared["automatic_provisioning"] is False
        return {
            "authorization_mode": "free_gpu",
            "free_runtime_operator_confirmed": True,
            "execution_addendum_required": False,
            "paid_fallback_allowed": False,
        }

    assert mode == "metered_existing_units" and referenced
    assert declared["existing_units_operator_confirmed"] is True
    assert declared["free_runtime_operator_confirmed"] is False
    assert declared["cap_compute_units"] == 10
    assert declared["purchases_allowed"] is declared["renewal_allowed"] is False
    assert declared.get("paid_fallback_allowed", False) is False
    assert declared.get("automatic_provisioning", False) is False
    assert declared["execution_addendum_sha256"] == referenced
    assert declared["execution_addendum_path"] == "reports/cuda-eval2/execution-addendum.json"
    folder = Path(root) / "reports/cuda-eval2"
    addendum_path = folder / "execution-addendum.json"
    assert sha(addendum_path) == referenced
    addendum = json.loads(addendum_path.read_text())
    assert addendum["schema_version"] == 1
    assert addendum["run_id"] == report["run_id"] == plan["run_id"] == identity["run_id"] == RUN_ID
    assert addendum["authorization_mode"] == mode
    assert (
        addendum["analysis_plan_sha256"]
        == report["analysis_plan_sha256"]
        == sha(folder / "analysis-plan.json")
    )
    assert (
        addendum["artifact_identity_sha256"]
        == report["artifact_identity_sha256"]
        == sha(folder / "artifact-identity.json")
    )
    for key in ("before_adapter_sha256", "after_adapter_sha256"):
        assert addendum[key] == identity[key]
    assert addendum["cases_sha256"] == report["cases_sha256"] == plan["cases_sha256"]
    assert addendum["source_inputs_sha256"] == plan["source_inputs_sha256"]
    assert addendum["previous_interrupted_run"] == plan["previous_interrupted_run"]
    for name, wanted in addendum["source_inputs_sha256"].items():
        assert sha(Path(root) / name) == wanted, name
    supplementary = addendum["supplementary_sources_sha256"]
    assert set(supplementary) == {
        "training/cuda_eval2/budgeted_run.py",
        "tests/test_cuda_eval2_budget.py",
    }
    for name, wanted in supplementary.items():
        assert sha(Path(root) / name) == wanted, name
    superseded = addendum["supersedes_only"]
    assert superseded["field"] == "compute_authorization.free_gpu_only"
    assert superseded["original_value"] is True and superseded["effective_authorization"] == mode
    unchanged = addendum["unchanged_protocol"]
    for key in (
        "no_retraining",
        "no_checkpoint_selection",
        "no_old_run_replay",
        "no_old_outputs_pooled",
        "T4_only",
    ):
        assert unchanged[key] is True, key
    assert unchanged["seed"] == 202610287 and unchanged["total_calls_per_task"] == 12
    assert unchanged["generation"] == plan["generation"]
    assert unchanged["inference_model_kwargs"] == plan["inference_model_kwargs"]
    assert unchanged["case_order"] == plan["case_ids"] and tuple(unchanged["arms"]) == ARMS
    assert unchanged["controller"] == unchanged["scoring"] == "frozen_v7"
    human = addendum["human_authorization"]
    assert human["explicit"] is True
    assert human["cap_compute_units"] == 10
    assert human["starting_balance_units"] == 80 and human["stop_balance_units"] == 72
    assert human["hard_floor_balance_units"] == 70
    assert human["purchase_allowed"] is human["renewal_allowed"] is human["free_allocation_claimed"] is False
    assert human["T4_only"] is True
    assert isinstance(human["approval_text"], str) and human["approval_text"].strip()
    accounting = addendum["accounting"]
    assert report["execution_budget"] == accounting
    assert (
        accounting["automatic_compute_unit_metering"]
        is accounting["billing_stops_when_process_exits"]
        is False
    )
    assert isinstance(accounting["owner"], str) and accounting["owner"].strip()
    for key, value in {
        "starting_balance_units": 80,
        "hard_cap_units": 10,
        "hard_cap_balance": 70,
        "operator_stop_balance": 72,
        "reserved_units": 2,
    }.items():
        assert accounting[key] == value, key
    assert math.isfinite(accounting["observed_units_per_hour"]) and accounting["observed_units_per_hour"] > 0
    created = datetime.fromisoformat(addendum["created_at_utc"])
    assert datetime.fromisoformat(identity["sealed_at_utc"]) <= created
    allocated, process_deadline, disconnect_deadline = (
        datetime.fromisoformat(accounting[key])
        for key in ("runtime_allocated_at_utc", "process_deadline_utc", "runtime_disconnect_deadline_utc")
    )
    assert allocated <= created < process_deadline < disconnect_deadline
    projected = (
        (disconnect_deadline - allocated).total_seconds() / 3600 * accounting["observed_units_per_hour"]
    )
    assert accounting["projected_units_until_disconnect"] == projected <= 8
    assert accounting["projection_is_actual_consumption"] is False
    for record in report["records"]:
        for observation in record["trace"]:
            assert observation["timing"]["execution_addendum_sha256"] == referenced
            assert observation["timing"]["execution_authorization_mode"] == mode
    if journal is not None:
        assert journal
        for entry in journal:
            assert entry["run_id"] == RUN_ID and entry["execution_addendum_sha256"] == referenced
            assert created < datetime.fromisoformat(entry["started_at_utc"]) < process_deadline
            assert datetime.fromisoformat(entry["completed_at_utc"]) <= process_deadline
    return {
        "authorization_mode": mode,
        "execution_addendum_required": True,
        "execution_addendum_path": "reports/cuda-eval2/execution-addendum.json",
        "execution_addendum_sha256": referenced,
        "existing_units_operator_confirmed": True,
        "free_runtime_operator_confirmed": False,
        "cap_compute_units": 10,
        "starting_balance_units": 80,
        "stop_balance_units": 72,
        "hard_floor_balance_units": 70,
        "reserved_units": 2,
        "purchases_allowed": False,
        "renewal_allowed": False,
        "automatic_metering": False,
        "billing_stops_when_process_exits": False,
        "process_deadline_utc": accounting["process_deadline_utc"],
        "runtime_disconnect_deadline_utc": accounting["runtime_disconnect_deadline_utc"],
        "scope": (
            "Human-authorized existing-unit ceiling with external operator monitoring; not billing proof."
        ),
    }


def verify_run_identity(report, plan, identity, *, root=ROOT):
    """Independently bind run002 to retained interruption receipts, never old predictions."""
    assert report["run_id"] == plan["run_id"] == identity["run_id"] == RUN_ID
    assert report["old_outputs_pooled_or_replaced"] is plan["old_outputs_pooled_or_replaced"] is False
    assert report["training_repeated_or_modified"] is plan["training_repeated_or_modified"] is False
    verify_execution_authorization(report, plan, identity, root=root)
    assert (
        plan["after_checkpoint"]["adapter_sha256"]
        == identity["after_adapter_sha256"]
        == "33fe5c505ddf08cd8cb4032dbd9d350fa205134417ec0f1f90648f4af658e053"
    )
    assert identity["before_adapter_sha256"] == plan["before_initializer_sha256"]
    previous = plan["previous_interrupted_run"]
    assert previous == report["previous_interrupted_run"] == identity["previous_interrupted_run"]
    assert previous["run_id"] == "resolvedesk-cuda-eval-001"
    assert previous["status"] == "interrupted_final_state_unverified"
    assert previous["analysis_plan_sha256"] == (
        "cb08373c73938335659c018b13bd2ca31e76dda43a96b73e34bc3b0e30ce5866"
    )
    assert previous["archive_sha256"] == ("5f94ca79b20997987d09d5849d645b9761fc000462f8de1a9e91801bcd6fc741")
    assert previous["automatic_resume_allowed"] is previous["old_outputs_pooled_or_replaced"] is False
    assert previous["interruption_path"] == "reports/cuda-eval/interruption.json"
    assert previous["snapshot_audit_path"] == "reports/cuda-eval/interrupted-snapshot-audit.json"
    interruption_path, snapshot_path = (
        Path(root) / previous[key] for key in ("interruption_path", "snapshot_audit_path")
    )
    assert sha(interruption_path) == previous["interruption_sha256"]
    assert sha(snapshot_path) == previous["snapshot_audit_sha256"]
    interruption, snapshot = (json.loads(p.read_text()) for p in (interruption_path, snapshot_path))
    assert interruption["status"] == "interrupted" and not interruption["later_actual_progress_known"]
    assert not interruption["automatic_model_replay_allowed"] and not interruption["model_replayed"]
    assert not interruption["complete_96_record_evidence_available"]
    assert interruption["analysis_plan_sha256"] == previous["analysis_plan_sha256"]
    assert interruption["raw_snapshot"]["path"] == previous["archive_path"]
    assert interruption["raw_snapshot"]["sha256"] == previous["archive_sha256"]
    for source, retained, expected in [
        ("last_verified_completed_cases", "last_verified_completed_records", 84),
        ("complete_record_calls", "last_verified_completed_record_calls", 744),
        ("retained_journal_entries", "last_verified_journal_entries", 755),
        ("retained_completed_journal_entries", "last_verified_completed_journal_entries", 754),
        ("retained_started_journal_entries", "last_verified_started_journal_entries", 1),
    ]:
        assert interruption[source] == previous[retained] == expected
    assert previous["unresolved_call"] == interruption["unresolved_call"]
    assert previous["unresolved_call"]["status"] == "started"
    assert snapshot["all_executed_checks_passed"] and snapshot["latest_completed_arm_task_records"] == 84
    assert snapshot["latest_journal_counts"] == {"all_entries": 755, "completed": 754, "started": 1}


def require_complete_records(report, case_ids):
    """Reject partial, mixed-run, duplicate or reordered evidence before publishing anything."""
    assert report["status"] == "complete" and report["provider"] == "torch_cuda"
    assert report["run_id"] == RUN_ID and not report.get("pending_case")
    assert len(case_ids) == len(set(case_ids)) == 48
    assert all(cid.startswith("cuda-fresh2-test-") for cid in case_ids)
    assert [(r["variant"], r["case_id"]) for r in report["records"]] == [
        (arm, cid) for arm in ARMS for cid in case_ids
    ]
    assert all(0 < r["model_calls"] == len(r["trace"]) <= 12 for r in report["records"])


def replay_prompt_tokens(record, case, tokenizer, eos_ids):
    """Recreate every actual prompt and decode every stored output, preserving model mistakes."""
    with tempfile.TemporaryDirectory(prefix="resolvedesk-cuda-token-replay-") as tmp:
        store = ResolutionStore(Path(tmp) / "replay.sqlite")
        state = store.create(case)
        prompts, observed_prefix = [], []
        for event in record["events"]:
            if event["kind"] == "customer_message":
                state = store.customer_message(state["id"], event["detail"]["content"], state["revision"])
            elif event["kind"] == "model_tool":
                observation = event["detail"]
                timing = observation["timing"]
                # Backend semantics are independently replayed by audit_business.
                # Preserve actual diagnostic prose for model-input reconstruction:
                # local/cloud Pydantic patch versions can spell the same error differently.
                state["observations"] = copy.deepcopy(observed_prefix)
                prompt = messages(state)
                prompts.append(prompt)
                if tokenizer is None:
                    state = store.apply(state["id"], observation["raw"], {}, state["revision"])
                    observed_prefix.append(observation)
                    continue
                wanted = tokenizer.apply_chat_template(
                    prompt,
                    enable_thinking=False,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                )
                assert wanted == timing["input_token_ids"], (record["case_id"], "actual model input differs")
                generated = timing["generated_token_ids"]
                terminal = bool(generated and generated[-1] in eos_ids)
                assert terminal == timing["terminal_eos_removed_for_text"]
                assert timing["finish_reason"] == ("eos" if terminal else "length")
                text_ids = generated[:-1] if terminal else generated
                assert (
                    tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    == observation["raw"]
                )
                assert (
                    tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    == timing["decoded_with_terminal_special_tokens"]
                )
                state = store.apply(state["id"], observation["raw"], {}, state["revision"])
                observed_prefix.append(observation)
    return prompts


def verify_record(record, case, identity, generation, tokenizer=None, eos_ids=(), model_kwargs=None):
    audit_business(record, case, expected_provider="torch_cuda")
    wanted_sha = (
        identity["before_adapter_sha256"]
        if record["variant"] == ARMS[0]
        else identity["after_adapter_sha256"]
    )
    actions = Counter(canonical(o["action"]) for o in record["trace"] if o["action"])
    assert record["identical_action_repeats"] == sum(n - 1 for n in actions.values())
    assert math.isfinite(record["latency_ms"]) and record["latency_ms"] >= 0
    for observation in record["trace"]:
        timing = observation["timing"]
        assert timing["variant"] == record["variant"] and timing["adapter_sha256"] == wanted_sha
        assert timing["workflow"] == "v7" and timing["cloud_compute_used"] is True
        assert timing["gradient_checkpointing"] is False and timing["json_schema_constrained"] is False
        assert timing["generation_settings"] == generation
        assert timing["model_kwargs"] == (model_kwargs if model_kwargs is not None else {"logits_to_keep": 1})
        for ids_key, count_key in [
            ("input_token_ids", "input_tokens"),
            ("generated_token_ids", "output_tokens"),
        ]:
            ids = timing[ids_key]
            assert isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids)
            assert timing[count_key] == len(ids) and timing[ids_key + "_sha256"] == token_hash(ids)
        assert 0 <= timing["output_tokens"] <= 512 and timing["input_tokens"] > 0
        assert math.isfinite(timing["latency_ms"]) and timing["latency_ms"] >= 0
    return replay_prompt_tokens(record, case, tokenizer, eos_ids)


def verify_journal(records, journal, plan, identity, expected_prompts=None):
    assert all(row["status"] == "completed" for row in journal), (
        "Incomplete/ambiguous calls are not complete evidence"
    )
    expected = [
        (r["variant"], r["case_id"], i, observation)
        for r in records
        for i, observation in enumerate(r["trace"], 1)
    ]
    assert len(journal) == len(expected) == sum(r["model_calls"] for r in records)
    assert datetime.fromisoformat(plan["created_at_utc"]) <= datetime.fromisoformat(identity["sealed_at_utc"])
    for actual, (variant, cid, call_index, observation) in zip(journal, expected, strict=True):
        assert (actual["variant"], actual["case_id"], actual["call_index"]) == (variant, cid, call_index)
        assert actual["raw"] == observation["raw"] and actual["timing"] == observation["timing"]
        if expected_prompts is not None:
            assert actual["request_messages"] == expected_prompts[variant, cid, call_index]
        import hashlib

        assert (
            actual["request_sha256"]
            == hashlib.sha256(canonical(actual["request_messages"]).encode()).hexdigest()
        )
        assert datetime.fromisoformat(identity["sealed_at_utc"]) < datetime.fromisoformat(
            actual["started_at_utc"]
        )
        assert datetime.fromisoformat(actual["started_at_utc"]) <= datetime.fromisoformat(
            actual["completed_at_utc"]
        )


def training_proof(training_path, identity):
    report = json.loads(training_path.read_text())
    assert report["status"] == "completed" and report["kind"] == "train"
    assert report["completed_true_optimizer_updates"] == 1344
    assert sha(training_path) == identity["training_report_sha256"]
    assert sha(training_path.with_name("plan.json")) == identity["training_plan_sha256"]
    assert report["final_adapter"]["sha256"] == identity["after_adapter_sha256"]
    verification_path = training_path.with_name("verification.json")
    persisted = json.loads(verification_path.read_text())
    assert (
        persisted["report_sha256"] == sha(training_path) and persisted["plan_sha256"] == report["plan_sha256"]
    )
    assert persisted["completed_true_optimizer_updates"] == 1344
    assert (
        persisted["tensor_audit"]["available"]
        and persisted["tensor_audit"]["numeric_values_independently_checked"]
    )
    assert persisted["tensor_audit"]["files_sha256"]["final_adapter"] == identity["after_adapter_sha256"]
    assert (
        persisted["tensor_audit"]["tensor_count"] == 112
        and persisted["tensor_audit"]["parameter_count"] == 3670016
    )
    # Rerun the CPU metadata/source/schedule checks; no Torch or model is imported.
    recomputed = verify_training(training_path, require_tensors=False)
    for key in [
        "completed_true_optimizer_updates",
        "actual_sampling",
        "recorded_initializer_exact_equality",
        "recorded_initializer_tensor_count",
        "recorded_trainable_parameters",
        "timing",
        "memory",
    ]:
        assert persisted[key] == recomputed[key], key
    return {
        "report_path": str(training_path.relative_to(ROOT)),
        "report_sha256": sha(training_path),
        "plan_sha256": sha(training_path.with_name("plan.json")),
        "verification_path": str(verification_path.relative_to(ROOT)),
        "verification_sha256": sha(verification_path),
        "completed_updates": 1344,
        "parameters": 3670016,
        "tensor_count": 112,
        "before_adapter_sha256": identity["before_adapter_sha256"],
        "after_adapter_sha256": identity["after_adapter_sha256"],
        "adapter_l2_change": report["adapter_l2_change"],
        "elapsed_seconds": report["elapsed_loop_gpu_sync_seconds"],
        "peak_allocated_bytes": report["peak_cuda_allocated_bytes"],
        "peak_reserved_bytes": report["peak_cuda_reserved_bytes"],
        "full_physical_rows_used": report["sampling"]["all_physical_rows_used"],
        "final_fixed_before_test": True,
        "numeric_initialization_and_final_delta_verified": True,
        "precision_scope": (
            "Decoded-4bit-derived FP16 frozen base, FP32 LoRA masters; "
            "not NF4 QLoRA or MLX bitwise equivalence."
        ),
    }


def verify(training_path, *, tokenizer_root=None, check_figures=False):
    from training.cuda_eval2.evaluation import verify_plan
    from training.cuda_eval2.fixtures import load_cases
    from training.cuda_eval2.prepare import denominators

    report_path, plan_path, identity_path = (
        REPORTS / name for name in ["evaluation.json", "analysis-plan.json", "artifact-identity.json"]
    )
    report, plan, identity = (
        json.loads(path.read_text()) for path in [report_path, plan_path, identity_path]
    )
    assert plan == verify_plan(REPORTS)
    assert sha(plan_path) == "a547fd0790ad2003fcd3dc0a990b581484bf5decd30222c87499237fd236f704"
    assert sha(identity_path) == "f83a97af4775ac2c01bf82f5ff188de1743c010ec143c554b74a8309f7840d1e"
    verify_run_identity(report, plan, identity)
    require_complete_records(report, plan["case_ids"])
    assert (
        report["checkpoint_selection_performed"] is False
        and report["training_or_test_tuning_performed"] is False
    )
    assert report["analysis_plan_sha256"] == identity["analysis_plan_sha256"] == sha(plan_path)
    assert report["artifact_identity_sha256"] == sha(identity_path)
    assert identity["training_verification_sha256"] == sha(training_path.with_name("verification.json"))
    assert identity["base_config_sha256"] == plan["base_config_sha256"]
    assert report["source_inputs_sha256"] == plan["source_inputs_sha256"]
    for name, wanted in plan["source_inputs_sha256"].items():
        assert sha(ROOT / name) == wanted, name
    assert not plan["test_inference_started"] and tuple(plan["arms"]) == ARMS
    assert (
        plan["after_checkpoint"]["completed_updates"] == identity["completed_true_optimizer_updates"] == 1344
    )
    assert (
        identity["selection_performed"] is False
        and identity["tensor_count"] == 112
        and identity["parameter_count"] == 3670016
    )
    assert plan["total_calls_per_task"] == 12 and plan["generation"]["max_new_tokens"] == 512
    assert plan["generation"]["do_sample"] is False and plan["generation"]["num_beams"] == 1
    assert plan["generation"]["repetition_penalty"] == 1 and plan["generation"]["temperature"] is None
    assert plan["inference_model_kwargs"] == {"logits_to_keep": 1}
    assert (
        report["cases_sha256"]
        == plan["cases_sha256"]
        == sha(ROOT / "training/cuda_eval2/datasets/cases.json")
    )
    cases = {case["id"]: case for case in load_cases("test")}
    assert len(cases) == 48 and list(cases) == plan["case_ids"]
    quality = json.loads((REPORTS / "data-summary.json").read_text())
    assert quality["denominators"] == denominators(list(cases.values()))
    assert (
        quality["quality"]["unique_interactions"]
        == quality["quality"]["unique_order_worlds"]
        == quality["quality"]["unique_initial_model_prompts"]
        == 48
    )
    for name, wanted in quality["quality"]["historical_datasets_sha256"].items():
        assert sha(ROOT / name) == wanted, name
    model_audits = report["model_audits"]
    base_spec = json.loads((ROOT / "training/base-model-spec.json").read_text())
    base_weight_sha = next(
        item["sha256"] for item in base_spec["files"] if item["path"] == "model.safetensors"
    )
    assert model_audits[ARMS[0]]["base_model_sha256"] == base_weight_sha
    assert model_audits[ARMS[0]]["base_repo"] == base_spec["repo"]
    assert model_audits[ARMS[0]]["base_revision"] == base_spec["revision"]
    assert (
        model_audits[ARMS[0]]["exact_tensor_equality"]
        and model_audits[ARMS[0]]["initializer_sha256"] == identity["before_adapter_sha256"]
    )
    assert model_audits[ARMS[0]]["source_tensor_count"] == model_audits[ARMS[1]]["source_tensor_count"] == 112
    assert (
        model_audits[ARMS[0]]["source_parameter_count"]
        == model_audits[ARMS[1]]["source_parameter_count"]
        == 3670016
    )
    assert (
        model_audits[ARMS[1]]["exact_tensor_equality"]
        and model_audits[ARMS[1]]["source_sha256"] == identity["after_adapter_sha256"]
    )
    assert (
        model_audits[ARMS[1]]["base_reloaded"] is False
        and model_audits[ARMS[1]]["only_lora_tensors_overwritten"] is True
    )
    assert report["model_load_events"]
    for loaded in report["model_load_events"]:
        assert loaded["audit"]["initializer_sha256"] == identity["before_adapter_sha256"]
        assert loaded["audit"]["base_model_sha256"] == base_weight_sha
        assert datetime.fromisoformat(loaded["loaded_at_utc"]) > datetime.fromisoformat(
            identity["sealed_at_utc"]
        )
    tokenizer, eos_ids = None, ()
    if tokenizer_root is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_root, local_files_only=True, trust_remote_code=False
        )
        assert sha(tokenizer_root / "config.json") == plan["base_config_sha256"]
        config = json.loads((tokenizer_root / "config.json").read_text())
        eos_ids = (
            [config["eos_token_id"]] if isinstance(config["eos_token_id"], int) else config["eos_token_id"]
        )
    replayed_prompts = {}
    for record in report["records"]:
        prompts = verify_record(
            record,
            cases[record["case_id"]],
            identity,
            plan["generation"],
            tokenizer,
            eos_ids,
            plan["inference_model_kwargs"],
        )
        replayed_prompts.update(
            {(record["variant"], record["case_id"], index): prompt for index, prompt in enumerate(prompts, 1)}
        )
    journal_path = REPORTS / "call-journal.json"
    assert report["call_journal_sha256"] == sha(journal_path)
    journal = json.loads(journal_path.read_text())
    verify_journal(report["records"], journal, plan, identity, replayed_prompts)
    execution_authorization = verify_execution_authorization(report, plan, identity, journal=journal)
    if execution_authorization["execution_addendum_required"]:
        assert execution_authorization["execution_addendum_sha256"] == (
            "e6e72a162af8c140a83ef9cf4c13102c24afb7428905ade797e98463d14c0e3e"
        )
    protocol_path = REPORTS / "protocol-audit.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["run_id"] == RUN_ID
    if report.get("execution_addendum_sha256"):
        assert protocol["execution_addendum_sha256"] == report["execution_addendum_sha256"]
    assert report["protocol_reference"] == {"path": "protocol-audit.json", "sha256": sha(protocol_path)}
    assert protocol["analysis_plan_sha256"] == sha(plan_path) and protocol["artifact_identity_sha256"] == sha(
        identity_path
    )
    assert protocol["cases_sha256"] == plan["cases_sha256"]
    import hashlib

    assert (
        protocol["evaluation_records_sha256"]
        == hashlib.sha256(canonical(report["records"]).encode()).hexdigest()
    )
    assert protocol["source_sha256"] == sha(ROOT / "training/v7/protocol_audit.py")
    assert (
        protocol["business_scores_or_release_gate_modified"] is False
        and protocol["selection_rule_modified"] is False
    )
    actual_protocol = [audit_consent(record, cases[record["case_id"]]) for record in report["records"]]
    assert (
        protocol["records"] == actual_protocol and protocol["records_audited"] == len(actual_protocol) == 96
    )
    computed_protocol = protocol_summary(actual_protocol)
    for arm in ARMS:
        assert protocol["summary"][arm] == {
            key: value
            for key, value in computed_protocol[arm].items()
            if key in {"tasks", "gold_exchange_tasks", *COUNTS}
        }
    values = derived_values(report, cases, protocol)
    for arm in ARMS:
        for key, value in values["summary"][arm].items():
            if key in report["summary"][arm]:
                assert report["summary"][arm][key] == value, (arm, key)
    proof = training_proof(training_path, identity)
    if check_figures:
        exported = json.loads((FIGURES / "values.json").read_text())
        for key, value in values.items():
            assert exported[key] == value, key
        assert exported["evaluation_sha256"] == sha(report_path)
        assert exported["protocol_audit_sha256"] == sha(protocol_path)
        assert exported["analysis_plan_sha256"] == sha(plan_path)
        assert exported["training_report_sha256"] == proof["report_sha256"]
        assert exported["training_proof_sha256"] == sha(REPORTS / "training-proof.json")
        assert exported["artifact_identity_sha256"] == sha(identity_path)
        assert exported["analysis_source_sha256"] == sha(ROOT / "training/cuda_eval2/visualize.py")
        assert exported["metrics_source_sha256"] == sha(ROOT / "training/cuda_eval2/metrics.py")
        assert exported["run_id"] == RUN_ID
        assert exported["previous_interrupted_run"] == plan["previous_interrupted_run"]
        assert exported["verification_sha256"] == sha(REPORTS / "verification.json")
        assert exported["csv_sha256"] == sha(FIGURES / "per-case.csv")
        with (FIGURES / "per-case.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        assert Counter((r["variant"], r["case_id"]) for r in csv_rows) == Counter(
            (r["variant"], r["case_id"]) for r in report["records"]
        )
        source_rows = {(r["variant"], r["case_id"]): r for r in report["records"]}
        assert {"intent_initial", "initial_enum_errors", "model_calls", "task_success"} <= set(csv_rows[0])
        for row in csv_rows:
            actual = source_rows[row["variant"], row["case_id"]]
            for key, value in row.items():
                if key in {"intent_initial", "initial_enum_errors"}:
                    assert json.loads(value) == actual[key]
                else:
                    assert value == str(actual[key])
        assert set(exported["figure_sha256"]) == {
            f"{name}.{extension}" for name in FIGURE_NAMES for extension in ("png", "svg")
        }
        for name, wanted in exported["figure_sha256"].items():
            assert sha(FIGURES / name) == wanted, name
    write_path = REPORTS / ("figure-verification.json" if check_figures else "verification.json")
    (REPORTS / "training-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    result = {
        "run_id": RUN_ID,
        "previous_interrupted_run": plan["previous_interrupted_run"],
        "execution_authorization": execution_authorization,
        "records_reconciled": 96,
        "fresh_test_tasks": 48,
        "max_total_calls_per_task": 12,
        "raw_actions_and_confirmations_replayed": True,
        "raw_first_output_metrics_reconciled": True,
        "scoped_consent_rows_recomputed": 96,
        "journal_calls_reconciled": sum(r["model_calls"] for r in report["records"]),
        "prompt_and_output_tokens_replayed": tokenizer is not None,
        "actual_diagnostic_prose_preserved_for_prompt_replay": True,
        "evaluation_sha256": sha(report_path),
        "protocol_audit_sha256": sha(protocol_path),
        "analysis_plan_sha256": sha(plan_path),
        "artifact_identity_sha256": sha(identity_path),
        "training_proof_sha256": sha(REPORTS / "training-proof.json"),
        "source_sha256": sha(__file__),
        "figures_reconciled": check_figures,
        "model_load_events": len(report["model_load_events"]),
        "completed_case_boundary_resumes": report["resume_count"],
        "scope": (
            "Independent CPU code/trace/token reconciliation, "
            "not independent human annotation or production certification."
        ),
    }
    if check_figures:
        result.update(
            values_sha256=sha(FIGURES / "values.json"),
            csv_sha256=sha(FIGURES / "per-case.csv"),
            figure_hashes_verified=exported["figure_sha256"],
        )
    write_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-report", type=Path, default=ROOT / "reports/cuda-torch/probe-003/train-1344/report.json"
    )
    parser.add_argument("--tokenizer-root", type=Path, default=ROOT / "data/models/qwen3-4b-mlx")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.training_report,
                tokenizer_root=None if args.skip_tokenizer else args.tokenizer_root,
                check_figures=args.figures,
            )
        )
    )


if __name__ == "__main__":
    main()
