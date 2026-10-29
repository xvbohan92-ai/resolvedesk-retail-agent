"""CPU-only development-data contracts; these fixtures are not model evidence."""

import json
from collections import Counter
from pathlib import Path

import pytest

from training.common import ROOT, sha
from training.v8.common import DATA, INITIAL_ADAPTER_SHA256, read_rows
from training.v8.fixtures import FAMILIES, SPLITS, load_cases, make_case
from training.v8.prepare import (
    HISTORICAL_NAMES,
    PRIVATE_KEYS,
    canonical,
    denominators,
    digest_value,
    instruction_skeleton,
    quality_audit,
    recovery_kinds,
    run_reference,
)


def development_cases():
    return [
        make_case(split, family, n)
        for split, count in SPLITS.items()
        for family in FAMILIES
        for n in range(count)
    ]


def test_development_splits_and_business_denominators_are_fixed():
    cases = development_cases()
    assert Counter(case["split"] for case in cases) == {"v8-train": 96, "v8-valid": 48}
    for split, count in SPLITS.items():
        subset = [case for case in cases if case["split"] == split]
        d = denominators(subset)
        assert d["families"] == dict.fromkeys(FAMILIES, count)
        assert d["initial_permissions"] == {"unknown": 12 * count, "yes": 6 * count, "no": 6 * count}
        assert d["expected_outcomes"] == {
            "cancel": 2 * count,
            "return": 7 * count,
            "exchange": 7 * count,
            "handoff": 8 * count,
        }
        assert d["tasks_requiring_clarification"] == 9 * count
        assert d["required_answer_occurrences"] == 10 * count
        assert all(not recovery_kinds(case) for case in subset) if split == "v8-valid" else True


def test_payment_order_and_option_ambiguity_labels_are_not_conflated():
    cases = development_cases()
    for case in cases:
        if "payment_method" in case["required_answers"]:
            assert case["intent_initial"]["payment_method"] == "unknown"
            assert (
                "original order" in case["instruction_text"]
                or "original purchase" in case["instruction_text"]
                or "original-order" in case["instruction_text"]
                or "original hoodie purchase" in case["instruction_text"]
                or "original hoodie" in case["instruction_text"]
                or "original pullover" in case["instruction_text"]
            )
            sequence = case["reply_sequences"]["payment_method"]
            assert sequence[-1]["value"] == "original"
            if len(sequence) > 1:
                assert sequence[0]["value"] == "unknown"
        if case["family"] == "exchange_missing_option":
            assert case["intent_initial"]["requested_option"] == "unknown"
            assert (
                case["reply_sequences"]["new_option"][-1]["value"]
                == case["intent_complete"]["requested_option"]
            )
        if case["intent_initial"]["operation"] != "cancel":
            assert case["intent_initial"]["reason"] == "unknown"
        assert case["evidence_initial"] in case["initial_message"]


def test_fresh_language_worlds_and_inputs_do_not_read_heldout_test(monkeypatch):
    read = Path.read_text
    accessed = []

    def tracked(path, *args, **kwargs):
        accessed.append(str(path))
        assert path.name not in {"test_dataset.json", "eval_fixtures.py"}
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked)
    cases = development_cases()
    result = quality_audit(cases, {"v8-train": []})
    assert result["historical_cases_checked"] == 1323
    assert result["historical_datasets_checked"] == 10
    assert result["all_prior_and_cross_split_exact_overlaps_empty"]
    assert len(set(map(instruction_skeleton, cases))) == 144
    assert set(result["historical_inputs_sha256"]) == set(HISTORICAL_NAMES)
    assert all("/training/v8/" not in path for path in accessed)


def test_reference_targets_pass_real_flow_and_faults_remain_input_only(tmp_path):
    from resolve_desk.resolution_v8 import ResolutionStore

    store = ResolutionStore(tmp_path / "reference.sqlite")
    # All business families include every hard-negative and free-form reply type.
    for case in development_cases():
        rows, audit = run_reference(case, store)
        assert audit["task_success"] and audit["model_calls"] <= 12
        assert all(field in audit["questions"] for field in case["required_answers"])
        assert all(reply["after_actual_question"] for reply in audit["actual_replies"])
        for row in rows:
            visible = json.loads(row["messages"][1]["content"])
            assert set(visible).isdisjoint(PRIVATE_KEYS)
            target = json.loads(row["messages"][-1]["content"])
            if target["tool"] == "record_request":
                assert target["arguments"]["evidence"] in visible["messages"][-1]["content"]
    examples = {
        "nonliteral_quote": ("exchange_fallback", 0),
        "option_product_suffix": ("exchange", 0),
        "order_is_not_payment": ("missing_payment", 0),
        "reason_enum": ("cancel", 0),
        "unsupported_grant": ("exchange_unavailable", 0),
        "stale_quote": ("missing_reason", 1),
        "repeated_usable_question": ("exchange_missing_option", 0),
        "operation_change": ("exchange_missing_option", 0),
    }
    for recovery, (family, n) in examples.items():
        rows, audit = run_reference(make_case("v8-train", family, n), store, recovery)
        assert audit["input_faults"] and audit["task_success"]
        faults = {
            (fault["input_prompt_sha256"], canonical(fault["action"])) for fault in audit["input_faults"]
        }
        assert not faults & {
            (digest_value(row["messages"][:-1]), canonical(json.loads(row["messages"][-1]["content"])))
            for row in rows
        }
        contexts = [json.loads(row["messages"][1]["content"]) for row in rows]
        for fault in audit["input_faults"]:
            assert any(
                (
                    fault["result"].get("resolution_request") is not None
                    and fault["result"]["resolution_request"]
                    in [visible["current_request"], visible["previous_request"]]
                )
                or (
                    visible["validation_feedback"]
                    and visible["validation_feedback"].get("error") == fault["result"].get("error")
                )
                or any(event["action"] == fault["action"] for event in visible["recent_events"])
                for visible in contexts
            ), recovery


def test_saved_development_rows_are_unique_and_fixed_initializer_is_retained():
    cases = load_cases()
    assert cases == development_cases()
    rows = read_rows("train")
    assert len(rows) == len({canonical(row["messages"]) for row in rows})
    assert {row["case_id"] for row in rows} == {case["id"] for case in load_cases("train")}
    summary = json.loads((DATA / "data-summary.json").read_text())
    assert summary["initial_adapter_sha256"] == INITIAL_ADAPTER_SHA256
    assert summary["cases_sha256"] == sha(DATA / "cases.json")
    assert summary["reference_audit_sha256"] == sha(DATA / "reference-audit.json")
    assert summary["all_reference_tasks_pass"]
    assert not summary["model_inference_performed"]
    assert {"train", "valid"} == {split.removeprefix("v8-") for split in summary["rows"]}
    config = json.loads((ROOT / "training/v8/config.json").read_text())
    assert config["initialization_sha256"] == INITIAL_ADAPTER_SHA256


def test_saved_full_token_and_sampling_audit_discloses_partial_row_coverage():
    from training.v7.train import balanced_sample_indices

    audit = json.loads((DATA / "token-audit.json").read_text())
    summary = json.loads((DATA / "data-summary.json").read_text())
    assert summary["token_audit_sha256"] == sha(DATA / "token-audit.json")
    assert audit["config_sha256"] == sha(ROOT / "training/v8/config.json")
    assert audit["max_seq_length"] == 3200 and audit["projection_window"] == 192
    assert not audit["model_or_test_inference_performed"]
    assert audit["splits"]["train"]["longest_sequence"] == 3143
    assert len(audit["splits"]["train"]["original_3072_limit_exceeded_rows"]) == 2
    assert audit["splits"]["train"]["longest_completion"] == 112
    for split in ("train", "valid"):
        actual = audit["splits"][split]
        assert actual["data_sha256"] == sha(DATA / f"v8-{split}.jsonl")
        assert actual["all_inputs_and_supervised_targets_retained"]
        assert actual["maximum_tail_including_padding"] <= 192
    rows = read_rows("train")
    schedule = balanced_sample_indices(rows, 202610288, 1344)
    seen = {index for _, index in schedule}
    assert len(seen) == audit["sampling"]["unique_rows_sampled"] == 888
    assert len({rows[index]["case_id"] for index in seen}) == 96
    assert audit["sampling"]["physical_rows"] == 1100
    assert not audit["sampling"]["all_physical_rows_used"]
    assert audit["sampling"]["schedule_sha256"] == digest_value(schedule)


@pytest.mark.parametrize("split", ["test", "v8-test", "../test_dataset"])
def test_development_loaders_refuse_heldout_test(split):
    with pytest.raises(ValueError):
        load_cases(split)
    with pytest.raises(ValueError):
        read_rows(split)
