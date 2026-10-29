"""Second-cohort CPU oracle/data contracts; never candidate model evidence."""

import copy
import json
from collections import Counter

import pytest

from resolve_desk.resolution_v7 import ResolutionStore
from training.cuda_eval import fixtures as prior_fixtures
from training.cuda_eval import prepare as prior_prepare
from training.cuda_eval2 import fixtures, prepare


@pytest.fixture(scope="module")
def cases():
    return [fixtures.make_case(family, n) for family in fixtures.FAMILIES for n in range(2)]


@pytest.fixture(scope="module")
def reference(cases):
    return prepare.reference_audit(cases)


def test_same_business_structure_and_balanced_permission_choices(cases):
    d = prepare.denominators(cases)
    assert d["tasks"] == len({case["id"] for case in cases}) == 48
    assert d["families"] == dict.fromkeys(fixtures.FAMILIES, 2)
    assert d["initial_permissions"] == {"unknown": 24, "yes": 12, "no": 12}
    assert d["non_authorized_tasks"] == 36 and d["exchange_origin_tasks"] == 30
    assert d["expected_outcomes"] == {"cancel": 4, "return": 14, "exchange": 14, "handoff": 16}
    assert d["available_exchange_tasks"] == 14 and d["authorized_stockout_return_tasks"] == 4
    assert d["missing_email_tasks"] == 8 and d["tasks_requiring_clarification"] == 18
    assert d["required_answer_occurrences"] == 20
    assert d["confirmation_choices_all_tasks"] == {"true": 24, "false": 24}
    assert d["confirmation_choices_non_handoff"] == {"true": 16, "false": 16}


def test_all_nine_prior_datasets_including_interrupted48_are_disjoint(cases):
    quality = prepare.quality_audit(cases)
    assert quality["historical_datasets_checked"] == 9
    assert quality["historical_cases_checked"] == 1275
    assert "training/cuda_eval/datasets/cases.json" in quality["historical_datasets_sha256"]
    assert (
        quality["unique_case_ids"] == quality["unique_interactions"] == quality["unique_order_worlds"] == 48
    )
    assert quality["unique_instruction_skeletons"] == quality["unique_initial_model_prompts"] == 48
    assert quality["unique_entity_ids"] == 434
    assert all(not value for key, value in quality.items() if key.endswith("_overlap"))
    assert quality["previous_model_results_used_for_construction"] is False


def test_generation_and_pure_helper_reuse_do_not_replace_prior_module_globals(cases):
    prior_data, prior_reports = prior_prepare.DATA, prior_prepare.REPORTS
    prior_fixture_data = prior_fixtures.DATA
    assert cases == [fixtures.make_case(family, n) for family in fixtures.FAMILIES for n in range(2)]
    assert prepare.denominators(cases)["tasks"] == 48
    assert prior_prepare.DATA == prior_data and prior_prepare.REPORTS == prior_reports
    assert prior_fixtures.DATA == prior_fixture_data
    assert fixtures.DATA != prior_fixture_data and prepare.REPORTS != prior_reports


def test_no_gold_answers_family_or_case_id_enters_initial_model_projection(tmp_path, cases):
    poisoned = copy.deepcopy(cases[0])
    for key in prepare.FORBIDDEN_MODEL_KEYS:
        poisoned[key] = "DO_NOT_EXPOSE_" + key
    prompt = prepare.initial_prompt(poisoned, ResolutionStore(tmp_path / "projection.sqlite"))
    encoded = json.dumps(prompt)
    assert all("DO_NOT_EXPOSE_" + key not in encoded for key in prepare.FORBIDDEN_MODEL_KEYS)
    assert poisoned["id"] not in encoded
    assert set(prepare.public_case(cases[0])) == set(prepare.PUBLIC_FIELDS)


def test_reference48_pass_frozen_grader_with_no_extra_or_hidden_model_calls(reference):
    assert reference["cohort"] == "cuda_eval2" and reference["tasks"] == 48
    assert reference["all_business_references_pass"]
    assert all(row["reference"]["task_success"] for row in reference["records"])
    assert all(0 < row["reference"]["model_calls"] <= 12 for row in reference["records"])
    assert all(row["reference"]["quote_rejections"] == 0 for row in reference["records"])
    assert all(not row["reference"]["initial_enum_errors"] for row in reference["records"])
    assert all(row["reference"]["repeated_question_attempts"] == 0 for row in reference["records"])


def test_reference_protocol_has_scoped_consent_commits_and_declines(reference):
    assert reference["protocol_exchange_tasks"] == 30
    s = reference["protocol_summary"]
    assert s["raw_return_attempts"] == s["return_proposals"] == 4
    assert s["committed_returns"] == s["final_return_orders"] == s["declined_return_proposals"] == 2
    assert s["return_attempts_without_scoped_yes"] == s["return_proposals_without_scoped_yes"] == 0
    assert s["committed_returns_without_scoped_yes"] == s["final_returns_without_verified_commit"] == 0


def test_reference_replies_are_actual_asks_and_quotes_use_latest_customer_text(cases, reference):
    index = {case["id"]: case for case in cases}
    for row in reference["records"]:
        case = index[row["case_id"]]
        latest, pending, answered = case["initial_message"], None, []
        for event in row["reference"]["events"]:
            if event["kind"] == "customer_message":
                assert pending is not None and event["detail"]["content"] == case["answers"][pending]
                latest = event["detail"]["content"]
                answered.append(pending)
                pending = None
            elif event["kind"] == "model_tool":
                action, result = event["detail"]["action"], event["detail"]["result"]
                if action and action["tool"] == "record_request":
                    assert action["arguments"]["evidence"] in latest
                if "question" in result and "error" not in result:
                    pending = result["field"]
        assert answered == row["reference"]["questions"]
        assert all(field in answered for field in case["required_answers"])


def test_initial_reason_payment_and_permission_labels_stay_separate(cases):
    for case in cases:
        initial = case["intent_initial"]
        if initial["operation"] != "cancel":
            assert initial["reason"] == "unknown"
        if "payment_method" in case["required_answers"]:
            assert initial["payment_method"] == "unknown"
        if case["family"] == "exchange_gift":
            assert initial["payment_method"] == "gift_card" and initial["fallback_return"] == "unknown"
    assert Counter(case["intent_initial"]["reason"] for case in cases) == {
        "unknown": 44,
        "ordered by mistake": 2,
        "no longer needed": 2,
    }


def test_loader_is_test_only_and_does_not_use_previous_dataset(tmp_path, monkeypatch, cases):
    monkeypatch.setattr(fixtures, "DATA", tmp_path)
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    assert fixtures.load_cases("test") == fixtures.load_cases(None) == cases
    with pytest.raises(ValueError, match="only a test split"):
        fixtures.load_cases("train")
