"""CPU-only fresh-cohort construction and result-blind business/protocol audits.

Reference trajectories are authored oracle decisions, not model measurements or
SFT data. They must pass the unchanged independent business grader before the
evaluation plan is frozen. No benchmark inference is performed here.
"""

import argparse
import copy
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from resolve_desk.resolution_v7 import ResolutionStore, messages
from training.common import ROOT, sha
from training.cuda_eval.fixtures import DATA, FAMILIES, make_case
from training.v7.common import verify_frozen
from training.v7.evaluation import run_case
from training.v7.protocol_audit import COUNTS, audit_record

REPORTS = ROOT / "reports/cuda-eval"
PUBLIC_FIELDS = ("id", "initial_message", "customer", "orders", "products")
INTERACTION_FIELDS = (
    "initial_message",
    "customer",
    "orders",
    "products",
    "answers",
    "confirmation",
    "required_answers",
)
FORBIDDEN_MODEL_KEYS = {
    "expected",
    "answers",
    "intent_initial",
    "intent_complete",
    "family",
    "required_answers",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest_value(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def public_case(case):
    return {key: copy.deepcopy(case[key]) for key in PUBLIC_FIELDS}


def entities(case):
    result = {case["customer"]["id"], case["customer"]["email"]}
    if "gift_card" in case["customer"]:
        result.add(case["customer"]["gift_card"]["id"])
    for product_id, product in case["products"].items():
        result.add(product_id)
        result.update(v["item_id"] for v in product["variants"])
    for order in case["orders"]:
        result.update(order[key] for key in ("id", "user_id", "original_payment"))
        result.update(item[key] for item in order["items"] for key in ("item_id", "product_id"))
    return result


def catalog_labels(case):
    return {
        label
        for product in case["products"].values()
        for label in [product["name"], *[v["option"] for v in product["variants"]]]
    }


def instruction_skeleton(case):
    text = case.get("evidence_initial", case["initial_message"])
    for token in sorted(entities(case) | catalog_labels(case), key=len, reverse=True):
        text = re.sub(re.escape(token), "<entity>", text, flags=re.IGNORECASE)
    return " ".join(text.casefold().split())


def initial_prompt(case, store):
    state = store.create(public_case(case))
    prompt = messages(state)
    visible = json.loads(prompt[1]["content"])
    assert set(visible).isdisjoint(FORBIDDEN_MODEL_KEYS)
    assert visible["messages"] == [{"role": "customer", "content": case["initial_message"]}]
    assert visible["completed_reads"] == [] and visible["current_request"] is None
    return prompt


class ReferenceProvider:
    """A declared CPU oracle for feasibility, never used by a candidate evaluator."""

    def __init__(self, case):
        self.case = case
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(copy.deepcopy(prompt))
        visible = json.loads(prompt[1]["content"])
        assert set(visible).isdisjoint(FORBIDDEN_MODEL_KEYS)
        action = copy.deepcopy(visible["decision_gate"]["next_action"])
        if action["tool"] == "record_request":
            intent = dict(self.case["intent_initial"])
            for actual in visible["clarification_memory"]:
                key = "requested_option" if actual["field"] == "new_option" else actual["field"]
                if key != "fallback_return":
                    intent[key] = self.case["intent_complete"][key]
            action = {
                "tool": "record_request",
                "arguments": {
                    **intent,
                    "evidence": self.case["evidence_initial"]
                    if len(visible["messages"]) == 1
                    else visible["messages"][-1]["content"],
                },
            }
        return json.dumps(action), {
            "provider": "cpu_reference_oracle",
            "variant": "reference",
            "latency_ms": 0,
        }


def reference_audit(cases):
    records = []
    with tempfile.TemporaryDirectory(prefix="resolvedesk-cuda-fresh-reference-") as tmp:
        store = ResolutionStore(Path(tmp) / "reference.sqlite")
        for case in cases:
            provider = ReferenceProvider(case)
            record = {"variant": "reference", **run_case(store, case, provider, messages, max_steps=12)}
            assert record["task_success"], (case["id"], record)
            assert record["initial_enum_errors"] == {} and not record["initial_raw_parse_failure"]
            assert record["quote_rejections"] == record["repeated_question_attempts"] == 0
            assert all(field in record["questions"] for field in case["required_answers"])
            protocol = audit_record(record, case)
            assert protocol["return_proposals_without_scoped_yes"] == 0
            assert protocol["committed_returns_without_scoped_yes"] == 0
            assert protocol["final_returns_without_verified_commit"] == 0
            records.append({"case_id": case["id"], "reference": record, "protocol": protocol})
    protocol_summary = {key: sum(row["protocol"][key] for row in records) for key in COUNTS}
    return {
        "scope": (
            "Deterministic authored CPU oracle feasibility only; not LLM accuracy, training "
            "targets or customer evidence."
        ),
        "frozen_controller_sha256": sha(ROOT / "resolve_desk/resolution_v7.py"),
        "frozen_evaluator_sha256": sha(ROOT / "training/v7/evaluation.py"),
        "frozen_business_grader_sha256": sha(ROOT / "training/v6/scoring.py"),
        "frozen_protocol_auditor_sha256": sha(ROOT / "training/v7/protocol_audit.py"),
        "tasks": len(cases),
        "all_business_references_pass": True,
        "all_references_within_12_total_calls": True,
        "reference_total_calls": sum(row["reference"]["model_calls"] for row in records),
        "required_replies_supplied_only_after_actual_asks": True,
        "protocol_exchange_tasks": sum(row["protocol"]["audited"] for row in records),
        "protocol_summary": protocol_summary,
        "records": records,
    }


def quality_audit(cases):
    historical_paths = sorted(
        path for path in (ROOT / "training").glob("**/datasets/cases.json") if path.parent != DATA
    )
    assert historical_paths
    historical = [case for path in historical_paths for case in json.loads(path.read_text())]
    current_entities = [entities(case) for case in cases]
    all_current_entities = set.union(*current_entities)
    all_historical_entities = set.union(*(entities(case) for case in historical))
    current_worlds = {order["id"] for case in cases for order in case["orders"]}
    old_worlds = {order["id"] for case in historical for order in case["orders"]}
    instructions = [instruction_skeleton(case) for case in cases]
    old_instructions = {instruction_skeleton(case) for case in historical}
    assert not all_current_entities & all_historical_entities
    assert sum(map(len, current_entities)) == len(all_current_entities), (
        "Fresh cases must share no entity IDs"
    )
    assert len(current_worlds) == len(cases) and not current_worlds & old_worlds
    assert len(set(instructions)) == len(cases) and not set(instructions) & old_instructions
    current_initial = {case["initial_message"] for case in cases}
    assert len(current_initial) == len(cases)
    assert not current_initial & {case["initial_message"] for case in historical}
    new_catalog = set.union(*(catalog_labels(case) for case in cases))
    old_catalog = set.union(*(catalog_labels(case) for case in historical))
    assert not new_catalog & old_catalog
    with tempfile.TemporaryDirectory(prefix="resolvedesk-cuda-fresh-inputs-") as tmp:
        store = ResolutionStore(Path(tmp) / "prompts.sqlite")
        current_prompt_hashes = {case["id"]: digest_value(initial_prompt(case, store)) for case in cases}
        historical_prompt_hashes = {digest_value(initial_prompt(case, store)) for case in historical}
    assert len(set(current_prompt_hashes.values())) == len(cases)
    assert not set(current_prompt_hashes.values()) & historical_prompt_hashes
    interactions = {canonical({key: case[key] for key in INTERACTION_FIELDS}) for case in cases}
    assert len(interactions) == len(cases)
    return {
        "historical_cases_checked": len(historical),
        "historical_datasets_sha256": {str(path.relative_to(ROOT)): sha(path) for path in historical_paths},
        "unique_interactions": len(interactions),
        "unique_order_worlds": len(current_worlds),
        "unique_initial_model_prompts": len(set(current_prompt_hashes.values())),
        "unique_entity_ids": len(all_current_entities),
        "unique_instruction_skeletons": len(set(instructions)),
        "initial_prompt_sha256": current_prompt_hashes,
        "duplicate_interaction_groups": [],
        "duplicate_initial_prompt_groups": [],
        "duplicate_instruction_skeleton_groups": [],
        "prior_world_overlap": [],
        "prior_entity_overlap": [],
        "prior_initial_message_overlap": [],
        "prior_instruction_skeleton_overlap": [],
        "prior_catalog_label_overlap": [],
        "prior_v7_projection_prompt_overlap": [],
        "historical_prompt_comparison_scope": (
            "All historical case inputs conservatively projected with unchanged V7 public "
            "messages; historical workflow versions are not reclassified as V7 evidence."
        ),
        "authorship_scope": (
            "New authored wording and independent entities/worlds; policy and scenario "
            "families are inherited, so this is not an independent human benchmark."
        ),
        "gold_and_scripted_answers_excluded_from_model_projection": True,
    }


def denominators(cases):
    return {
        "tasks": len(cases),
        "families": dict(Counter(case["family"] for case in cases)),
        "initial_permissions": dict(Counter(case["intent_initial"]["fallback_return"] for case in cases)),
        "non_authorized_tasks": sum(case["intent_initial"]["fallback_return"] != "yes" for case in cases),
        "expected_outcomes": dict(Counter(case["expected"]["kind"] for case in cases)),
        "available_exchange_tasks": sum(case["expected"]["kind"] == "exchange" for case in cases),
        "authorized_stockout_return_tasks": sum(
            case["family"].startswith("exchange") and case["expected"]["kind"] == "return" for case in cases
        ),
        "required_handoff_tasks": sum(case["expected"]["kind"] == "handoff" for case in cases),
        "missing_email_tasks": sum("email" in case["required_answers"] for case in cases),
        "tasks_requiring_clarification": sum(bool(case["required_answers"]) for case in cases),
        "required_answer_occurrences": sum(len(case["required_answers"]) for case in cases),
        "required_answer_fields": dict(
            Counter(field for case in cases for field in case["required_answers"])
        ),
        "exchange_origin_tasks": sum(case["intent_initial"]["operation"] == "exchange" for case in cases),
        "confirmation_choices_all_tasks": {
            str(choice).lower(): sum(case["confirmation"] == choice for case in cases)
            for choice in [True, False]
        },
        "confirmation_choices_non_handoff": {
            str(choice).lower(): sum(
                case["confirmation"] == choice and case["expected"]["kind"] != "handoff" for case in cases
            )
            for choice in [True, False]
        },
    }


def generate():
    verify_frozen()
    assert not (REPORTS / "analysis-plan.json").exists(), "Never modify a planned evaluation cohort"
    assert not (REPORTS / "evaluation.json").exists() and not (REPORTS / "evaluation-partial.json").exists()
    cases = [make_case(family, n) for family in FAMILIES for n in range(2)]
    assert len(cases) == len({case["id"] for case in cases}) == 48
    quality = quality_audit(cases)
    reference = reference_audit(cases)
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (DATA / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (DATA / "public.json").write_text(json.dumps([public_case(case) for case in cases], indent=2) + "\n")
    reference["cases_sha256"] = sha(DATA / "cases.json")
    (REPORTS / "reference-audit.json").write_text(json.dumps(reference, indent=2) + "\n")
    summary = {
        "scope": (
            "Evaluation-only fresh authored cohort fixed before model inference; no SFT data, "
            "no checkpoint selection, no reused historical case aliases."
        ),
        "model_inference_started": False,
        "cases_sha256": sha(DATA / "cases.json"),
        "public_sha256": sha(DATA / "public.json"),
        "reference_audit_sha256": sha(REPORTS / "reference-audit.json"),
        "source_inputs_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for path in [
                Path(__file__),
                ROOT / "training/cuda_eval/fixtures.py",
                ROOT / "tests/test_cuda_eval_data.py",
            ]
        },
        "frozen_v7_manifest_sha256": sha(ROOT / "training/v7/datasets/manifest.json"),
        "denominators": denominators(cases),
        "quality": quality,
        "all_48_cpu_reference_business_tasks_pass": True,
        "all_reference_scoped_consent_audits_pass": True,
    }
    (REPORTS / "data-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "quality"}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate"])
    parser.parse_args()
    generate()
