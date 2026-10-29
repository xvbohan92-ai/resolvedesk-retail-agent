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
from training.cuda_eval.metrics import ARMS, FIGURES, REPORTS, derived_values, protocol_summary
from training.cuda_torch.verify_report import verify as verify_training
from training.v7.protocol_audit import COUNTS
from training.v7.protocol_audit import audit_record as audit_consent
from training.v7.verify_results import audit_record as audit_business


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def token_hash(ids):
    import hashlib

    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


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
    from training.cuda_eval.fixtures import load_cases
    from training.cuda_eval.prepare import denominators

    report_path, plan_path, identity_path = (
        REPORTS / name for name in ["evaluation.json", "analysis-plan.json", "artifact-identity.json"]
    )
    report, plan, identity = (
        json.loads(path.read_text()) for path in [report_path, plan_path, identity_path]
    )
    assert report["status"] == "complete" and report["provider"] == "torch_cuda"
    assert (
        report["checkpoint_selection_performed"] is False
        and report["training_or_test_tuning_performed"] is False
    )
    assert report["analysis_plan_sha256"] == identity["analysis_plan_sha256"] == sha(plan_path)
    assert report["artifact_identity_sha256"] == sha(identity_path)
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
        report["cases_sha256"] == plan["cases_sha256"] == sha(ROOT / "training/cuda_eval/datasets/cases.json")
    )
    cases = {case["id"]: case for case in load_cases("test")}
    assert len(cases) == 48 and list(cases) == plan["case_ids"]
    assert [(r["variant"], r["case_id"]) for r in report["records"]] == [
        (arm, cid) for arm in ARMS for cid in cases
    ]
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
    verify_journal(report["records"], json.loads(journal_path.read_text()), plan, identity, replayed_prompts)
    protocol_path = REPORTS / "protocol-audit.json"
    protocol = json.loads(protocol_path.read_text())
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
        assert exported["analysis_source_sha256"] == sha(ROOT / "training/cuda_eval/visualize.py")
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
        for name, wanted in exported["figure_sha256"].items():
            assert sha(FIGURES / name) == wanted, name
    write_path = REPORTS / ("figure-verification.json" if check_figures else "verification.json")
    (REPORTS / "training-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    result = {
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
