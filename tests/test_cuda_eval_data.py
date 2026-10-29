"""CPU-only fresh-cohort contracts; oracle outputs are not model evidence."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from resolve_desk.resolution_v7 import ResolutionStore
from training.cuda_eval import fixtures, prepare


@pytest.fixture(scope="module")
def cases():
    return [fixtures.make_case(family, n) for family in fixtures.FAMILIES for n in range(2)]


@pytest.fixture(scope="module")
def reference(cases):
    return prepare.reference_audit(cases)


def test_48_fresh_cases_have_predeclared_business_and_permission_denominators(cases):
    actual = prepare.denominators(cases)
    assert actual["tasks"] == len({case["id"] for case in cases}) == 48
    assert actual["families"] == dict.fromkeys(fixtures.FAMILIES, 2)
    assert actual["initial_permissions"] == {"unknown": 24, "yes": 12, "no": 12}
    assert actual["expected_outcomes"] == {"cancel": 4, "return": 14, "exchange": 14, "handoff": 16}
    assert actual["non_authorized_tasks"] == 36 and actual["exchange_origin_tasks"] == 30
    assert actual["available_exchange_tasks"] == 14 and actual["authorized_stockout_return_tasks"] == 4
    assert actual["required_handoff_tasks"] == 16 and actual["missing_email_tasks"] == 8
    assert actual["tasks_requiring_clarification"] == 18
    assert actual["required_answer_occurrences"] == 20
    assert actual["confirmation_choices_all_tasks"] == {"true": 24, "false": 24}
    assert actual["confirmation_choices_non_handoff"] == {"true": 16, "false": 16}


def test_world_generation_is_deterministic_and_not_a_previous_case_alias(cases):
    assert cases == [fixtures.make_case(family, n) for family in fixtures.FAMILIES for n in range(2)]
    quality = prepare.quality_audit(cases)
    assert quality["unique_interactions"] == quality["unique_order_worlds"] == 48
    assert quality["unique_initial_model_prompts"] == quality["unique_instruction_skeletons"] == 48
    assert quality["unique_entity_ids"] == 434
    assert len(quality["historical_datasets_sha256"]) >= 8
    assert all(not value for key, value in quality.items() if key.endswith("_overlap"))
    assert quality["duplicate_initial_prompt_groups"] == []


def test_public_projection_omits_all_gold_answers_and_family_context(tmp_path, cases):
    poisoned = copy.deepcopy(cases[0])
    for key in prepare.FORBIDDEN_MODEL_KEYS:
        poisoned[key] = "SECRET_" + key
    store = ResolutionStore(tmp_path / "public.sqlite")
    prompt = prepare.initial_prompt(poisoned, store)
    encoded = json.dumps(prompt)
    assert all("SECRET_" + key not in encoded for key in prepare.FORBIDDEN_MODEL_KEYS)
    assert set(prepare.public_case(cases[0])) == set(prepare.PUBLIC_FIELDS)
    assert [message["role"] for message in prompt] == ["system", "user"]


def test_reference_all48_pass_unchanged_business_grader_and_12_total_calls(reference):
    assert reference["tasks"] == len(reference["records"]) == 48
    assert reference["all_business_references_pass"]
    assert all(row["reference"]["task_success"] for row in reference["records"])
    assert all(row["reference"]["model_calls"] <= 12 for row in reference["records"])
    assert all(row["reference"]["repeated_question_attempts"] == 0 for row in reference["records"])
    assert all(not row["reference"]["initial_enum_errors"] for row in reference["records"])


def test_reference_protocol_covers_real_scoped_yes_commits_and_declines(reference):
    assert reference["protocol_exchange_tasks"] == 30
    summary = reference["protocol_summary"]
    assert summary["raw_return_attempts"] == summary["return_proposals"] == 4
    assert summary["committed_returns"] == summary["final_return_orders"] == 2
    assert summary["declined_return_proposals"] == 2
    for key in [
        "return_attempts_without_scoped_yes",
        "return_proposals_without_scoped_yes",
        "committed_returns_without_scoped_yes",
        "final_returns_without_verified_commit",
    ]:
        assert summary[key] == 0


def test_reference_quotes_only_latest_actual_customer_message_and_replies_follow_asks(cases, reference):
    index = {case["id"]: case for case in cases}
    for row in reference["records"]:
        case = index[row["case_id"]]
        latest, pending = case["initial_message"], None
        actual_questions = []
        for event in row["reference"]["events"]:
            if event["kind"] == "customer_message":
                assert pending is not None
                assert event["detail"]["content"] == case["answers"][pending]
                latest = event["detail"]["content"]
                actual_questions.append(pending)
                pending = None
            elif event["kind"] == "model_tool":
                detail = event["detail"]
                action, result = detail["action"], detail["result"]
                if action and action["tool"] == "record_request":
                    assert action["arguments"]["evidence"] in latest
                if "question" in result and "error" not in result:
                    pending = result["field"]
        assert actual_questions == row["reference"]["questions"]
        assert all(field in actual_questions for field in case["required_answers"])


def test_payment_permission_does_not_become_return_permission_and_missing_payment_stays_unknown(cases):
    for case in cases:
        initial = case["intent_initial"]
        if case["family"] == "exchange_gift":
            assert initial["payment_method"] == "gift_card" and initial["fallback_return"] == "unknown"
            assert case["answers"]["payment_method"] == case["customer"]["gift_card"]["id"]
        if "payment_method" in case["required_answers"]:
            assert initial["payment_method"] == "unknown"
        if initial["operation"] != "cancel":
            assert initial["reason"] == "unknown"
    assert Counter(case["intent_initial"]["reason"] for case in cases) == {
        "unknown": 44,
        "no longer needed": 2,
        "ordered by mistake": 2,
    }


def test_evaluation_only_loader_has_no_training_split(tmp_path, monkeypatch, cases):
    monkeypatch.setattr(fixtures, "DATA", Path(tmp_path))
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    assert fixtures.load_cases("test") == fixtures.load_cases(None) == cases
    with pytest.raises(ValueError, match="evaluation-only"):
        fixtures.load_cases("train")
