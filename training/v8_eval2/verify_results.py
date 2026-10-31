"""Independent CPU replay of the fresh saved-weight-only144-task evaluation.

This mutable delivery verifier does not change the frozen controller, grader, data,
decoding or failed scores. A verification receipt is issued only after all144
fresh records, saved adapter identities and actual input/output tokens reconcile.
"""

import argparse
import copy
import json
import math
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from resolve_desk.resolution_v7 import ResolutionStore as OldStore
from resolve_desk.resolution_v7 import messages as old_messages
from resolve_desk.resolution_v8 import ResolutionStore as NewStore
from resolve_desk.resolution_v8 import messages as new_messages
from training.common import ROOT, sha
from training.cuda_eval.evaluation import value_hash
from training.cuda_eval.provider import MODEL_KWARGS, generation_settings, token_hash
from training.v8.evaluation import ARMS, MAX_CALLS, run_case, summarize
from training.v8.protocol_audit import audit_record as audit_consent
from training.v8.protocol_audit import summarize as consent_summary

REPORTS = ROOT / "reports/v8-eval2"
RUN_ID = "resolvedesk-v8-eval-002"
BEFORE_SHA = "33fe5c505ddf08cd8cb4032dbd9d350fa205134417ec0f1f90648f4af658e053"
AFTER_SHA = "39dabb9b73ae6cb6230fc8a0bca9c914f6ba658c3c54a17f9775e3fcef60970f"
LEGACY_SHA = "041cde85d1593f27f154da5b4afbec2f0635fb2bcfe4a34940339e092925ce08"
PHYSICAL_PROOF_SHA = "51902c343ea1dd1a352a46a26dfa6f2c2f82fee81b527bb4f6c88a5385b79418"


def load(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular evidence file: {path}")
    return json.loads(path.read_text())


def finite(value):
    assert type(value) in (int, float) and math.isfinite(value) and value >= 0


def utc(value):
    result = datetime.fromisoformat(value)
    assert result.tzinfo is not None and result.utcoffset().total_seconds() == 0
    return result


def require_complete(report, cases, plan):
    """No prefix, old rows, pending case, additional training or selection is accepted."""
    assert report["run_id"] == RUN_ID and report["round"] == "v8-eval2" and report["split"] == "test"
    assert (
        report["status"] == "complete" and not report.get("pending_case") and not report.get("interruption")
    )
    assert report["resume_count"] == 0 and report["checkpoint_selection_performed"] is False
    assert report["additional_training_performed"] is False and report["validation_model_calls"] == 0
    assert report["max_total_calls_per_task"] == MAX_CALLS == 12
    assert report["maximum_total_issued_calls"] == 1728 and report["model_loads_this_process"] == 1
    assert tuple(report["arms"]) == tuple(ARMS) and len(cases) == len({c["id"] for c in cases}) == 48
    assert [c["id"] for c in cases] == plan["case_order"]
    assert all(c["id"].startswith("v8-eval2-") for c in cases)
    assert report["cases_sha256"] == plan["dataset"]["canonical_cases_sha256"] == value_hash(cases)
    assert [(r["variant"], r["case_id"]) for r in report["records"]] == [
        (arm, c["id"]) for arm in ARMS for c in cases
    ]
    assert all(0 < r["model_calls"] == len(r["trace"]) <= 12 for r in report["records"])
    assert sum(r["model_calls"] for r in report["records"]) <= 1728
    assert utc(report["started_at_utc"]) <= utc(report["completed_at_utc"])


def normalize_result(result, proposal_ids):
    """Normalize opaque proposal IDs only; retain fingerprints and business values."""
    result = copy.deepcopy(result)
    if "proposal" in result:
        proposal = result["proposal"]
        proposal["id"] = proposal_ids[proposal["id"]]
    # Pydantic/Python diagnostic prose can differ across CPU/Colab patch versions.
    # The actual prose remains in the exact prompt-token replay below.
    if result.get("error") == "invalid_action_json":
        result.pop("detail", None)
    return result


def compare_replay(record, replay):
    """Compare every graded field and full transaction/event payload after replay."""
    assert set(record) == set(replay) | {"variant"}
    proposal_ids = {}
    assert len(record["trace"]) == len(replay["trace"])
    for original, actual in zip(record["trace"], replay["trace"], strict=True):
        assert original["raw"] == actual["raw"] and original["action"] == actual["action"]
        assert original["timing"] == actual["timing"]
        want, got = original["result"], actual["result"]
        assert ("proposal" in want) == ("proposal" in got)
        if "proposal" in want:
            old, new = want["proposal"]["id"], got["proposal"]["id"]
            assert old not in proposal_ids or proposal_ids[old] == new
            assert new not in proposal_ids.values() or proposal_ids.get(old) == new
            proposal_ids[old] = new
        got = copy.deepcopy(got)
        if got.get("error") == "invalid_action_json":
            got.pop("detail", None)
        assert normalize_result(want, proposal_ids) == got, "Backend result differs on CPU replay"
    for key in set(replay) - {"trace", "events", "latency_ms"}:
        assert record[key] == replay[key], (record["case_id"], key)
    finite(record["latency_ms"])
    assert len(record["events"]) == len(replay["events"])
    previous_id = -1
    for original, actual in zip(record["events"], replay["events"], strict=True):
        assert type(original["id"]) is int and original["id"] > previous_id
        previous_id = original["id"]
        finite(original["created"])
        assert original["kind"] == actual["kind"], "Event order differs"
        want, got = copy.deepcopy(original["detail"]), copy.deepcopy(actual["detail"])
        if original["kind"] == "model_tool":
            want["result"] = normalize_result(want["result"], proposal_ids)
            if got["result"].get("error") == "invalid_action_json":
                got["result"].pop("detail", None)
        if original["kind"] in {"operation_committed", "confirmation_declined"}:
            want["proposal_id"] = proposal_ids[want["proposal_id"]]
        assert want == got, "Event/confirmation payload differs"


def verify_timing(observation, variant, identity, generation, bindings):
    timing = observation["timing"]
    before = variant != ARMS[-1]
    assert timing["provider"] == "torch_cuda" and timing["variant"] == variant
    assert timing["adapter_sha256"] == identity["before_adapter_sha256" if before else "after_adapter_sha256"]
    assert timing["workflow"] == ("v7" if variant == ARMS[0] else "v8")
    assert timing["round"] == "v8-eval2" and timing["cloud_compute_used"] is True
    assert timing["run_id"] == RUN_ID and all(timing[k] == v for k, v in bindings.items())
    assert timing["gradient_checkpointing"] is False and timing["json_schema_constrained"] is False
    assert timing["autocast"] is False and timing["generation_settings"] == generation
    assert timing["model_kwargs"] == MODEL_KWARGS
    finite(timing["latency_ms"])
    for key, count in [("input_token_ids", "input_tokens"), ("generated_token_ids", "output_tokens")]:
        ids = timing[key]
        assert isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids)
        assert timing[count] == len(ids) and timing[key + "_sha256"] == token_hash(ids)
    assert timing["input_tokens"] > 0 and 0 <= timing["output_tokens"] <= 512
    assert type(timing["peak_cuda_allocated_bytes"]) is int
    assert type(timing["peak_cuda_reserved_bytes"]) is int
    assert 0 < timing["peak_cuda_allocated_bytes"] <= timing["peak_cuda_reserved_bytes"]


def verify_tokens(observation, prompt, tokenizer, eos_ids):
    timing = observation["timing"]
    expected = tokenizer.apply_chat_template(
        prompt, enable_thinking=False, add_generation_prompt=True, tokenize=True, return_dict=False
    )
    assert timing["input_token_ids"] == expected, "Actual prompt token IDs differ"
    generated = timing["generated_token_ids"]
    assert not any(token in eos_ids for token in generated[:-1]), "Generation continued after EOS"
    terminal = bool(generated and generated[-1] in eos_ids)
    assert terminal or len(generated) == 512, "Unexplained early generation stop without terminal EOS"
    assert timing["terminal_eos_removed_for_text"] is terminal
    assert timing["finish_reason"] == ("eos" if terminal else "length")
    assert (
        tokenizer.decode(
            generated[:-1] if terminal else generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        == observation["raw"]
    ), "Generated IDs do not decode to retained raw output"
    assert (
        tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        == timing["decoded_with_terminal_special_tokens"]
    )


def verify_record(record, case, identity, generation, tokenizer, eos_ids, bindings):
    """Re-run frozen grading through the actual Store using only retained raw outputs."""
    assert record["variant"] in ARMS
    assert record["case_id"] == case["id"]
    assert 0 < record["model_calls"] == len(record["trace"]) <= MAX_CALLS
    for observation in record["trace"]:
        verify_timing(observation, record["variant"], identity, generation, bindings)
    prompts = []

    class RawProvider:
        def generate(self, prompt):
            index = len(prompts)
            assert index < len(record["trace"]), "CPU replay requested an unrecorded model call"
            observation = record["trace"][index]
            verify_tokens(observation, prompt, tokenizer, eos_ids)
            prompts.append(copy.deepcopy(prompt))
            return observation["raw"], copy.deepcopy(observation["timing"])

    old = record["variant"] == ARMS[0]
    store_class, prompt_function = (OldStore, old_messages) if old else (NewStore, new_messages)

    def actual_prompt(state):
        # Preserve the actual diagnostic text in model inputs, while the Store
        # independently executes and compares the semantics of every raw action.
        state = copy.deepcopy(state)
        state["observations"] = copy.deepcopy(record["trace"][: len(prompts)])
        return prompt_function(state)

    with tempfile.TemporaryDirectory(prefix="resolvedesk-v8-eval2-cpu-replay-") as directory:
        replay = run_case(store_class(Path(directory) / "replay.sqlite"), case, RawProvider(), actual_prompt)
    assert len(prompts) == record["model_calls"]
    compare_replay(record, replay)
    return prompts


def verify_journal(report, journal, prompts, identity, resource, bindings):
    expected = [
        (row, index, observation)
        for row in report["records"]
        for index, observation in enumerate(row["trace"], 1)
    ]
    assert len(journal) == len(expected) and all(e["status"] == "completed" for e in journal)
    last_completed = utc(report["started_at_utc"])
    for entry, (row, index, observation) in zip(journal, expected, strict=True):
        assert (entry["variant"], entry["case_id"], entry["call_index"]) == (
            row["variant"],
            row["case_id"],
            index,
        )
        assert entry["raw"] == observation["raw"] and entry["timing"] == observation["timing"]
        assert all(entry[k] == v for k, v in bindings.items())
        assert entry["request_messages"] == prompts[row["variant"], row["case_id"], index]
        assert entry["request_sha256"] == value_hash(entry["request_messages"])
        started, completed = utc(entry["started_at_utc"]), utc(entry["completed_at_utc"])
        assert utc(identity["sealed_at_utc"]) < started
        assert (
            utc(resource["allocated_at_utc"]) <= started <= completed < utc(resource["process_deadline_utc"])
        )
        assert last_completed <= started and completed <= utc(report["completed_at_utc"])
        last_completed = completed
    return len(expected)


def regular(root, relative):
    path = Path(root) / relative
    assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
    assert path.resolve().is_relative_to(Path(root).resolve())
    assert not any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root))
    assert path.is_file(), f"Missing regular input: {relative}"
    return path


def read_at(root, relative):
    return load(regular(root, relative))


def verify_saved_adapters(plan, identity, artifacts, root=ROOT):
    """Read only the saved adapter tensors/configs; no model framework or base load."""
    import numpy as np
    from safetensors import safe_open

    expected = {"before": BEFORE_SHA, "after": AFTER_SHA, "legacy": LEGACY_SHA}
    assert set(plan["weights"]) == set(expected)
    for label, spec in plan["weights"].items():
        assert spec["sha256"] == expected[label]
        folder = spec["runtime_directory"]
        assert sha(regular(root, f"{folder}/{spec['filename']}")) == expected[label]
        assert sha(regular(root, f"{folder}/adapter_config.json")) == spec["config_sha256"]
    before_spec, after_spec = (plan["weights"][k] for k in ("before", "after"))
    configs = [
        read_at(root, f"{spec['runtime_directory']}/adapter_config.json")
        for spec in (before_spec, after_spec)
    ]
    for config in configs:
        assert config["r"] == 8 and config["lora_alpha"] == 128 and config["lora_dropout"] == 0
        assert not config.get("use_rslora", False) and config.get("bias", "none") == "none"
    assert set(configs[0]["target_modules"]) == set(configs[1]["target_modules"])
    paths = [
        regular(root, f"{spec['runtime_directory']}/{spec['filename']}") for spec in (before_spec, after_spec)
    ]

    def key(name):
        return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(
            ".lora_B.default.weight", ".lora_B.weight"
        )

    total, squares = 0, 0.0
    with (
        safe_open(str(paths[0]), framework="numpy") as before,
        safe_open(str(paths[1]), framework="numpy") as after,
    ):
        left, right = ({key(k): k for k in handle.keys()} for handle in (before, after))
        assert len(left) == len(right) == len(list(before.keys())) == len(list(after.keys())) == 112
        assert set(left) == set(right)
        for name in left:
            a, b = before.get_tensor(left[name]), after.get_tensor(right[name])
            assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
            assert np.isfinite(a).all() and np.isfinite(b).all()
            total += a.size
            squares += float(np.square(b.astype(np.float64) - a.astype(np.float64)).sum())
    assert total == 3_670_016 and math.isfinite(squares) and squares > 0
    delta = math.sqrt(squares)
    proof = artifacts["final-training-physical-verification.json"]
    recorded = proof["training"]
    assert proof["status"] == "verified_final_training_physical_weights_only" and proof["cpu_only"] is True
    assert recorded["initializer_sha256"] == BEFORE_SHA and recorded["final_adapter_sha256"] == AFTER_SHA
    assert recorded["exact_prior_initialization"] is True and recorded["all_values_finite"] is True
    assert recorded["tensors_checked"] == 112 and recorded["parameters_checked"] == total
    assert recorded["true_optimizer_updates"] == 1344 and recorded["fixed_final_only"] is True
    assert recorded["test_used_for_selection"] is False
    assert math.isclose(delta, recorded["adapter_l2_change"], rel_tol=1e-7, abs_tol=1e-10)
    assert math.isclose(
        delta, identity["independent_cpu_tensor_audit"]["adapter_l2_change"], rel_tol=1e-7, abs_tol=1e-10
    )
    train = artifacts["training-report.json"]
    assert train["status"] == "completed" and train["completed_true_optimizer_updates"] == 1344
    assert train["final_adapter"]["sha256"] == AFTER_SHA and not train.get("attempt_in_progress")
    assert recorded["training_report_sha256"] == plan["artifact_bindings"]["training-report.json"]["sha256"]
    assert recorded["training_plan_sha256"] == plan["artifact_bindings"]["training-plan.json"]["sha256"]
    assert train["plan_sha256"] == recorded["training_plan_sha256"]
    cloud_proof = artifacts["training-proof.json"]
    assert cloud_proof["training_report_sha256"] == recorded["training_report_sha256"]
    assert cloud_proof["training_plan_sha256"] == recorded["training_plan_sha256"]
    assert (
        cloud_proof["final_adapter_sha256"] == AFTER_SHA and cloud_proof["initializer_sha256"] == BEFORE_SHA
    )
    assert cloud_proof["true_optimizer_updates"] == 1344 and cloud_proof["test_used_for_selection"] is False
    assert (
        sha(
            regular(
                root, plan["artifact_bindings"]["final-training-physical-verification.json"]["runtime_path"]
            )
        )
        == PHYSICAL_PROOF_SHA
    )
    return {
        "cpu_only": True,
        "tensors_checked_each": 112,
        "parameters_checked_each": total,
        "all_values_finite": True,
        "rank": 8,
        "adapter_l2_difference": delta,
        "before_sha256": BEFORE_SHA,
        "after_sha256": AFTER_SHA,
        "exact_config_sha256": {k: v["config_sha256"] for k, v in plan["weights"].items()},
        "new_training_performed": False,
        "initialization_proof_reused_by_exact_hash": PHYSICAL_PROOF_SHA,
    }


def verify_protocol_inputs(plan, identity, report, cases, root=ROOT):
    """Bind the new immutable input closure and exclude historical rows, selection and training."""
    assert plan["run_id"] == identity["run_id"] == report["run_id"] == RUN_ID
    assert plan["status"] == "frozen_pre_inference" and plan["model_inference_started"] is False
    assert plan["arms"] == list(ARMS) and plan["total_calls_per_task"] == 12
    assert plan["maximum_total_issued_calls"] == 1728
    assert plan["report_only"] is True and plan["deployment_or_default_provider_change_allowed"] is False
    for key in (
        "additional_training_allowed",
        "smoke_gradient_updates_allowed",
        "validation_inference_allowed",
        "checkpoint_selection_performed",
        "resume_allowed",
        "ambiguous_call_replay_allowed",
    ):
        assert plan[key] is False
    assert identity["no_training_performed_this_run"] is True
    assert identity["checkpoint_selection_performed"] is False
    assert identity["before_adapter_sha256"] == BEFORE_SHA and identity["after_adapter_sha256"] == AFTER_SHA
    assert identity["physical_proof_sha256"] == PHYSICAL_PROOF_SHA
    assert (
        identity["weights"] == plan["weights"] and identity["artifact_bindings"] == plan["artifact_bindings"]
    )
    assert report["old_run_provenance"] == identity["old_run_provenance"] == plan["old_run_provenance"]
    old = plan["old_run_provenance"]
    assert old["archive_sha256"] == "e256d29c0bb85634cc9050fad652813c2eeee38a14eccc19d74dab301a707104"
    assert old["old_completed_records"] == 34 and old["old_issued_calls"] == 223
    assert old["old_unfinished_calls"] == 0 and old["old_completed_calls_outside_completed_task_rows"] == 11
    assert old["old_pending_task_apply_or_finalization_not_reconstructed"] is True
    assert old["old_run_resume_allowed"] is old["old_outputs_pooled"] is old["old_calls_replayed"] is False
    assert old["existing_validation"]["records"] == 48
    assert old["existing_validation"]["rerun_allowed"] is False
    assert old["existing_validation"]["quality_scores_reused_in_new_comparison"] is False
    for name, expected in plan["source_inputs_sha256"].items():
        assert sha(regular(root, name)) == expected, name
    frozen = read_at(root, "training/v8/datasets/manifest.json")
    assert len(frozen["files"]) == 218
    assert sha(regular(root, "training/v8/datasets/manifest.json")) == plan["frozen_v8_manifest_sha256"]
    for name, expected in frozen["files"].items():
        assert sha(regular(root, name)) == expected, name
    assert plan["dataset"]["cases_sha256"] == sha(regular(root, "training/v8_eval2/datasets/cases.json"))
    assert plan["dataset"]["canonical_cases_sha256"] == value_hash(cases)
    assert plan["dataset"]["public_sha256"] == sha(regular(root, "training/v8_eval2/datasets/public.json"))
    assert plan["base_model_spec_sha256"] == sha(regular(root, "training/base-model-spec.json"))
    assert (
        identity["analysis_plan_sha256"]
        == report["analysis_plan_sha256"]
        == sha(regular(root, "reports/v8-eval2/analysis-plan.json"))
    )
    assert report["artifact_identity_sha256"] == sha(regular(root, "reports/v8-eval2/artifact-identity.json"))
    assert utc(plan["created_at_utc"]) <= utc(identity["sealed_at_utc"]) < utc(report["started_at_utc"])
    artifacts = {}
    assert set(plan["artifact_bindings"]) == {
        "training-report.json",
        "training-plan.json",
        "training-proof.json",
        "v8-artifact-identity.json",
        "final-training-physical-verification.json",
    }
    for name, item in plan["artifact_bindings"].items():
        assert sha(regular(root, item["runtime_path"])) == item["sha256"]
        artifacts[name] = read_at(root, item["runtime_path"])
    return artifacts


def verify_historical_boundary(plan, cases, root=ROOT):
    """Read the cached immutable historical metadata; never replay its calls or grade it."""
    old = plan["old_run_provenance"]
    folder = old["snapshot"]
    partial_path = folder + "/reports/v8/evaluation/evaluation-partial.json"
    journal_path = folder + "/reports/v8/evaluation/call-journal.json"
    assert sha(regular(root, partial_path)) == old["old_partial_report_sha256"]
    assert sha(regular(root, journal_path)) == old["old_call_journal_sha256"]
    partial, journal = read_at(root, partial_path), read_at(root, journal_path)
    assert partial["status"] == "partial" and len(partial["records"]) == 34
    assert len(journal) == 223 and all(e["status"] == "completed" for e in journal)
    assert sum(r["model_calls"] for r in partial["records"]) == 212
    assert partial.get("pending_case") == old["old_pending_case"]
    old_ids = {r["case_id"] for r in partial["records"]} | {old["old_pending_case"]["case_id"]}
    assert not old_ids & {c["id"] for c in cases}
    cached_valid = folder + "/reports/v8/validation/"
    for field, filename in (
        ("evaluation_sha256", "evaluation.json"),
        ("journal_sha256", "call-journal.json"),
        ("protocol_sha256", "protocol-audit.json"),
    ):
        assert sha(regular(root, cached_valid + filename)) == old["existing_validation"][field]
    return {
        "old_snapshot_sha256": old["archive_sha256"],
        "old_completed_records": 34,
        "old_journal_completed_calls": 223,
        "old_completed_calls_in_task_records": 212,
        "old_pending_task_completed_responses": 11,
        "no_historical_case_rows_pooled": True,
        "old_model_calls_replayed": False,
        "old_task_or_validation_scores_recomputed": False,
    }


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def verify(*, tokenizer_root=None):
    """Only actual complete144 evidence can issue a new verification receipt."""
    if not __debug__:
        raise RuntimeError("Verification requires Python assertions to be enabled")
    report = load(REPORTS / "evaluation.json")  # Fail before tokenizer/tensor imports on partial work.
    plan, identity = load(REPORTS / "analysis-plan.json"), load(REPORTS / "artifact-identity.json")
    from training.v8_eval2.fixtures import load_cases
    from training.v8_eval2.prepare import denominators
    from training.v8_eval2.resource_guard import verify_resource_receipt

    cases = load_cases("test")
    require_complete(report, cases, plan)
    assert plan["dataset"]["denominators"] == denominators(cases)
    artifacts = verify_protocol_inputs(plan, identity, report, cases)
    history = verify_historical_boundary(plan, cases)
    physical = verify_saved_adapters(plan, identity, artifacts)
    resource = verify_resource_receipt(active=False)
    bindings = {
        "run_id": RUN_ID,
        "analysis_plan_sha256": sha(REPORTS / "analysis-plan.json"),
        "artifact_identity_sha256": sha(REPORTS / "artifact-identity.json"),
        "resource_receipt_sha256": sha(REPORTS / "resource-authorization.json"),
    }
    assert all(report[k] == v for k, v in bindings.items())
    assert utc(report["completed_at_utc"]) < utc(resource["process_deadline_utc"])
    base = Path(tokenizer_root or ROOT / "data/models/qwen3-4b-mlx")
    base_config = load(base / "config.json")
    assert sha(base / "config.json") == plan["base_config_sha256"]
    assert plan["generation"] == generation_settings(base_config) and plan["model_kwargs"] == MODEL_KWARGS
    spec = load(ROOT / "training/base-model-spec.json")
    for item in spec["files"]:
        if not item["path"].endswith(".safetensors"):
            assert sha(base / item["path"]) == item["sha256"]
    # Tokenizer only. Neither inference framework nor any base model is loaded.
    os.environ["USE_TORCH"], os.environ["USE_TF"] = "0", "0"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    eos = plan["generation"]["eos_token_id"]
    eos = [eos] if isinstance(eos, int) else eos
    prompts, consent_rows = {}, []
    case_by_id = {c["id"]: c for c in cases}
    for row in report["records"]:
        actual_prompts = verify_record(
            row, case_by_id[row["case_id"]], identity, plan["generation"], tokenizer, eos, bindings
        )
        for i, prompt in enumerate(actual_prompts, 1):
            prompts[row["variant"], row["case_id"], i] = prompt
        consent_rows.append(audit_consent(row, case_by_id[row["case_id"]]))
    journal = load(REPORTS / "call-journal.json")
    calls = verify_journal(report, journal, prompts, identity, resource, bindings)
    assert calls <= 1728
    assert report["call_journal_sha256"] == sha(REPORTS / "call-journal.json")
    for arm in ARMS:
        assert report["summary"][arm] == summarize([r for r in report["records"] if r["variant"] == arm])
        audit = report["model_audits"][arm]
        assert audit["source_sha256"] == (AFTER_SHA if arm == ARMS[-1] else BEFORE_SHA)
        assert audit["source_tensor_count"] == 112 and audit["source_parameter_count"] == 3_670_016
        assert audit["exact_tensor_equality"] is True and audit["base_reloaded"] is False
        assert audit["only_lora_tensors_overwritten"] is True
    protocol = load(REPORTS / "protocol-audit.json")
    assert all(protocol[k] == v for k, v in bindings.items())
    assert protocol["records"] == consent_rows and protocol["summary"] == consent_summary(consent_rows)
    assert protocol["source_sha256"] == sha(ROOT / "training/v8/protocol_audit.py")
    assert report["protocol_audit_sha256"] == sha(REPORTS / "protocol-audit.json")
    loaded = load(REPORTS / "evaluation-model-load.json")
    assert loaded == report["model_load_receipt"] and loaded["run_id"] == RUN_ID
    assert report["model_load_receipt_sha256"] == sha(REPORTS / "evaluation-model-load.json")
    assert (
        loaded["model_load_count"] == 1
        and loaded["resource_receipt_sha256"] == bindings["resource_receipt_sha256"]
    )
    assert utc(identity["sealed_at_utc"]) < utc(loaded["loaded_at_utc"]) <= utc(report["started_at_utc"])
    loader = loaded["loader_audit"]
    assert loader["initializer_sha256"] == LEGACY_SHA and loader["exact_tensor_equality"] is True
    assert loader["source_tensor_count"] == 112 and loader["source_parameter_count"] == 3_670_016
    assert loader["base_repo"] == spec["repo"] and loader["base_revision"] == spec["revision"]
    assert loader["base_model_sha256"] == next(
        x["sha256"] for x in spec["files"] if x["path"].endswith(".safetensors")
    )
    assert loader["adapter_master_dtype"] == "float32" and loader["base_storage_dtype"] == "float16"
    assert loader["not_qlora_or_nf4"] is True and loader["mlx_compute_bitwise_equivalence_claimed"] is False
    assert loaded["loader_source_sha256"] == sha(ROOT / "training/cuda_torch/model.py")
    launch = load(REPORTS / "evaluation-launch.json")
    assert all(launch[k] == v for k, v in bindings.items()) and launch["no_automatic_retry"] is True
    assert launch["command"] == "python -u -B -m training.v8_eval2.run run --execute"
    names = [
        f"reports/v8-eval2/{n}.json"
        for n in (
            "analysis-plan",
            "artifact-identity",
            "resource-authorization",
            "evaluation",
            "call-journal",
            "protocol-audit",
            "evaluation-launch",
            "evaluation-model-load",
        )
    ]
    names += [v["runtime_path"] for v in plan["artifact_bindings"].values()]
    names += [
        f"training/v8_eval2/datasets/{n}.json" for n in ("cases", "public", "data-summary", "reference-audit")
    ]
    names += ["training/v8_eval2/verify_results.py", "training/v8_eval2/metrics.py"]
    for value in plan["weights"].values():
        names.append(value["runtime_directory"] + "/adapter_config.json")
    receipt = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "verified",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "counts": {"test_records": 144},
        "calls_checked": calls,
        "replay": {
            "business": True,
            "token": True,
            "actions": True,
            "replies": True,
            "confirmation": True,
            "scoped_consent": True,
            "journal": True,
        },
        "physical_adapter_audit": physical,
        "historical_boundary_audit": history,
        "old_run_provenance": plan["old_run_provenance"],
        "source_inputs_sha256": plan["source_inputs_sha256"],
        "files": {n: sha(regular(ROOT, n)) for n in names},
        "verifier_source_sha256": sha(__file__),
        "additional_training_performed": False,
        "report_only": True,
        "deployment_or_default_provider_change_allowed": False,
        "scope": (
            "All144 fresh traces independently replayed. Evidence consistency only; not a model-quality pass."
        ),
    }
    assert not any(
        n == "torch" or n.startswith("torch.") or n == "mlx" or n.startswith("mlx.") for n in sys.modules
    ), "CPU verifier imported an inference framework"
    write_once(REPORTS / "result-verification.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(tokenizer_root=args.tokenizer_root), indent=2))


if __name__ == "__main__":
    main()
