"""Independent held-out data/CPU-oracle checks, never candidate model measurements."""

import copy
import json
from collections import Counter

import pytest

from resolve_desk.resolution_v7 import ResolutionStore
from training.cuda_eval import fixtures as old_fixtures
from training.cuda_eval import prepare as old_prepare
from training.v8 import eval_fixtures as fresh


@pytest.fixture(scope="module")
def cases():
    return [fresh.make_case(family, n) for family in fresh.FAMILIES for n in (0, 1)]


@pytest.fixture(scope="module")
def reference(cases):
    return fresh.reference_audit(cases)


def test_48_worlds_24_families_and_declared_business_denominators(cases):
    assert len(cases) == len({c["id"] for c in cases}) == 48
    assert Counter(c["family"] for c in cases) == dict.fromkeys(fresh.FAMILIES, 2)
    d = fresh.denominators(cases)
    assert d["initial_permissions"] == {"unknown": 24, "yes": 12, "no": 12}
    assert d["non_authorized_tasks"] == 36 and d["exchange_origin_tasks"] == 30
    assert d["expected_outcomes"] == {"cancel": 4, "return": 14, "exchange": 14, "handoff": 16}
    assert (
        d["available_exchange_tasks"],
        d["authorized_stockout_return_tasks"],
        d["required_handoff_tasks"],
        d["missing_email_tasks"],
    ) == (14, 4, 16, 8)
    assert d["tasks_requiring_clarification"] == 18 and d["required_answer_occurrences"] == 20
    assert d["confirmation_choices_all_tasks"] == {"true": 24, "false": 24}
    assert d["confirmation_choices_non_handoff"] == {"true": 16, "false": 16}
    assert all(len(c["orders"]) == len(c["products"]) == 1 for c in cases)
    assert len({p["name"] for c in cases for p in c["products"].values()}) == 6
    assert all(c["answers"]["operation"] == c["intent_initial"]["operation"] for c in cases)
    fallback = [
        c
        for c in cases
        if c["intent_initial"]["operation"] == "exchange" and c["expected"]["kind"] == "return"
    ]
    assert len(fallback) == 4 and all(c["answers"]["operation"] == "exchange" for c in fallback)


def test_history_isolation_uses_only_explicit_1323_case_allowlist(cases, monkeypatch):
    from pathlib import Path

    reads = []
    original = Path.read_text

    def track(path, *args, **kwargs):
        reads.append(str(path))
        assert not ("/training/v8/" in str(path) and any(t in str(path).lower() for t in ("train", "valid")))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", track)
    quality = fresh.quality_audit(cases)
    assert quality["historical_cases_checked"] == 1323 and quality["historical_datasets_checked"] == 10
    assert (
        quality["unique_worlds"]
        == quality["unique_initial_messages"]
        == quality["unique_instruction_skeletons"]
        == quality["unique_v7_projection_prompts"]
        == 48
    )
    assert quality["unique_entity_ids"] == 434
    assert all(not values for values in quality["historical_feature_overlaps"].values())
    assert (
        not quality["v8_train_or_validation_content_read"] and not quality["v8_candidate_inference_performed"]
    )
    assert all("/training/v8/" not in path for path in reads)


def test_original_order_is_not_a_payment_choice_and_ambiguity_requires_actual_literal_reply(cases):
    payment = [c for c in cases if "payment_method" in c["required_answers"]]
    assert len(payment) == 8
    for case in payment:
        assert case["intent_initial"]["payment_method"] == "unknown"
        assert "original" in case["initial_message"].lower()
    ambiguous = [c for c in cases if "genuine_two_variant_ambiguity_requires_ask" in c["coverage_tags"]]
    assert len(ambiguous) == 2
    for case in ambiguous:
        assert case["intent_initial"]["requested_option"] == "unknown"
        assert "new_option" in case["required_answers"]
        literal = case["answers"]["new_option"]
        options = {v["option"] for p in case["products"].values() for v in p["variants"]}
        assert literal in options and literal == case["intent_complete"]["requested_option"]
        assert sum(option in case["initial_message"] for option in options) == 2


def test_fidelity_and_suffix_coverage_preserve_raw_targets_without_forcing_model_failures(cases):
    assert all(case["evidence_initial"] in case["initial_message"] for case in cases)
    assert sum("latest_reply_quote" in c["coverage_tags"] for c in cases) == 18
    suffix = [c for c in cases if "variant_literal_with_product_noun_context" in c["coverage_tags"]]
    assert len(suffix) == 28
    for case in suffix:
        literal = case["intent_initial"]["requested_option"]
        assert literal != "unknown" and literal in case["initial_message"]
        assert any(v["option"] == literal for p in case["products"].values() for v in p["variants"])
    assert all("forced_model_output" not in c and "injected_error" not in c for c in cases)


def test_gold_answers_tags_and_case_ids_never_enter_initial_candidate_projection(cases, tmp_path):
    case = copy.deepcopy(cases[0])
    for key in (*old_prepare.FORBIDDEN_MODEL_KEYS, "coverage_tags"):
        case[key] = "GOLD_POISON_" + key
    prompt = fresh.initial_prompt(case, ResolutionStore(tmp_path / "projection.sqlite"))
    text = json.dumps(prompt)
    assert "GOLD_POISON_" not in text and case["id"] not in text
    assert set(fresh.public_case(cases[0])) == set(old_prepare.PUBLIC_FIELDS)


def test_cpu_oracle_confirms_policy_feasibility_with_actual_asks_and_12_total_actions(reference, cases):
    assert reference["tasks"] == 48 and reference["all_business_references_pass"]
    assert reference["all_references_within_12_total_calls"]
    assert all(
        row["reference"]["task_success"] and row["reference"]["model_calls"] <= 12
        for row in reference["records"]
    )
    assert all(
        row["reference"]["quote_rejections"] == row["reference"]["repeated_question_attempts"] == 0
        for row in reference["records"]
    )
    assert all(
        o["timing"]["provider"] == "cpu_reference_oracle"
        for row in reference["records"]
        for o in row["reference"]["trace"]
    )
    by_id = {c["id"]: c for c in cases}
    assert all(
        set(by_id[row["case_id"]]["required_answers"]) <= set(row["reference"]["questions"])
        for row in reference["records"]
    )
    summary = reference["protocol_summary"]
    assert (
        summary["return_proposals"] == 4
        and summary["committed_returns"] == summary["declined_return_proposals"] == 2
    )
    assert (
        summary["return_proposals_without_scoped_yes"]
        == summary["committed_returns_without_scoped_yes"]
        == summary["final_returns_without_verified_commit"]
        == 0
    )


def test_test_only_loader_and_generation_do_not_mutate_historical_module_globals(
    cases, tmp_path, monkeypatch
):
    historical = (old_fixtures.DATA, old_prepare.DATA, old_prepare.REPORTS)
    path = tmp_path / "test_dataset.json"
    path.write_text(json.dumps(cases))
    monkeypatch.setattr(fresh, "TEST_PATH", path)
    monkeypatch.setattr(fresh, "PLAN_PATH", tmp_path / "draft-plan.json")
    assert fresh.load_cases() == fresh.load_cases("test") == cases
    with pytest.raises(ValueError, match="test-only"):
        fresh.load_cases("train")
    with pytest.raises(FileExistsError):
        fresh.generate()
    assert json.loads(path.read_text()) == cases
    assert (old_fixtures.DATA, old_prepare.DATA, old_prepare.REPORTS) == historical


def test_persisted_authoring_is_exactly_the_saved_test_cohort(cases):
    assert fresh.load_cases() == cases


def test_cross_development_isolation_reads_hash_metadata_only(cases, monkeypatch):
    from pathlib import Path

    original = Path.read_text
    permitted = fresh.ROOT / "training/v8/datasets/data-summary.json"

    def guard(path, *args, **kwargs):
        if "/training/v8/" in str(path):
            assert path == permitted, "Test author must not read V8 development examples"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guard)
    result = fresh.compare_development_fingerprints(cases)
    assert result["status"] == "passed" and not result["development_content_read"]
    assert result["only_metadata_fingerprints_read"]
    assert all(
        not hashes for split in result["comparisons"].values() for hashes in split["overlaps"].values()
    )
