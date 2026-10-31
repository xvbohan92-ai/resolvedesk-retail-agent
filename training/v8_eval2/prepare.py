"""Model-free construction, isolation checks and two-flow CPU feasibility.

Only explicit historical case files are read. Saved model outputs, the old
interrupted 34 records and candidate quality results are not construction inputs.
The oracle uses labels solely to check feasibility; it is never an evaluator arm.
"""

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path

from training.common import ROOT, sha
from training.cuda_eval.prepare import (
    FORBIDDEN_MODEL_KEYS,
    INTERACTION_FIELDS,
    PUBLIC_FIELDS,
    catalog_labels,
    denominators,
    digest_value,
    entities,
    public_case,
)
from training.v7.evaluation import run_case
from training.v8.common import verify_frozen
from training.v8.prepare import instruction_skeleton
from training.v8.protocol_audit import COUNTS, audit_record
from training.v8_eval2.fixtures import DATA, FAMILIES, make_case

REPORTS = ROOT / "reports/v8-eval2"
HISTORICAL_NAMES = (
    "training/datasets/cases.json",
    "training/v2/datasets/cases.json",
    "training/v3/datasets/cases.json",
    "training/v4/datasets/cases.json",
    "training/v5/datasets/cases.json",
    "training/v6/datasets/cases.json",
    "training/v6_1/datasets/cases.json",
    "training/v7/datasets/cases.json",
    "training/cuda_eval/datasets/cases.json",
    "training/cuda_eval2/datasets/cases.json",
    "training/v8/datasets/cases.json",
    "training/v8/test_dataset.json",
)
PRIVATE_MODEL_KEYS = FORBIDDEN_MODEL_KEYS | {
    "reply_sequences",
    "instruction_text",
    "development_profile",
    "split",
    "evidence_initial",
    "coverage_tags",
    "confirmation",
    "required_clarification",
}
SOURCES = (
    "training/v8_eval2/__init__.py",
    "training/v8_eval2/fixtures.py",
    "training/v8_eval2/prepare.py",
    "tests/test_v8_eval2_data.py",
    "training/cuda_eval/fixtures.py",
    "training/cuda_eval/prepare.py",
    "training/v8/prepare.py",
    "training/v8/protocol_audit.py",
    "training/v7/evaluation.py",
    "training/v6/scoring.py",
    "resolve_desk/resolution_v7.py",
    "resolve_desk/resolution_v8.py",
)

__all__ = [
    "PUBLIC_FIELDS",
    "FORBIDDEN_MODEL_KEYS",
    "public_case",
    "denominators",
    "generate",
    "initial_prompt",
    "quality_audit",
    "reference_audit",
]

AUTHORING_TRACE = [
    {
        "stage": "CPU authoring checks before saving cohort or running any candidate",
        "change": "Historical fingerprint optional fields use get(None)",
        "reason": (
            "Early case schemas have no required_answers field; retain absent as null in the comparison."
        ),
        "historical_inputs_modified": False,
    },
    {
        "stage": "CPU authoring checks before saving cohort or running any candidate",
        "case": "v8-eval2-test-return-01",
        "observed_issue": (
            "Frozen V8 did not ground the explicit initial phrase 'reimbursing original payment'; "
            "the preliminary free-form payment reply was also unusable to its field parser."
        ),
        "preliminary_payment_reply": "Use original payment for the amount involved.",
        "final_payment_reply": "original",
        "change": "Use canonical original only after an actual payment question, for every applicable case",
        "initial_request_or_gold_changed": False,
        "case_removed": False,
        "controller_or_scoring_changed": False,
        "model_results_read": False,
        "interpretation": (
            "Do not discard the initial semantic challenge. Record any extra question in the CPU "
            "receipt and future actual trace; business completion alone does not imply zero needless asks."
        ),
    },
]


def serialized(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def runtime(flow):
    if flow == "v7":
        from resolve_desk.resolution_v7 import ResolutionStore, messages
    elif flow == "v8":
        from resolve_desk.resolution_v8 import ResolutionStore, messages
    else:
        raise ValueError("Expected frozen v7 or v8 flow")
    return ResolutionStore, messages


def initial_prompt(case, store, message_function):
    prompt = message_function(store.create(public_case(case)))
    visible = json.loads(prompt[1]["content"])
    assert set(visible).isdisjoint(PRIVATE_MODEL_KEYS)
    assert visible["messages"] == [{"role": "customer", "content": case["initial_message"]}]
    assert visible["completed_reads"] == [] and visible["current_request"] is None
    assert case["id"] not in json.dumps(prompt)
    return prompt


def fingerprints(cases):
    return {
        "case_ids": sorted(digest_value(c["id"]) for c in cases),
        "world_ids": sorted(digest_value(o["id"]) for c in cases for o in c["orders"]),
        "entities": sorted(digest_value(v) for v in set.union(*(entities(c) for c in cases))),
        "catalog_labels": sorted(digest_value(v) for v in set.union(*(catalog_labels(c) for c in cases))),
        "initial_messages": sorted(digest_value(c["initial_message"]) for c in cases),
        "instruction_skeletons": sorted(digest_value(instruction_skeleton(c)) for c in cases),
        "public_inputs": sorted(digest_value(public_case(c)) for c in cases),
        "whole_interactions": sorted(digest_value({k: c.get(k) for k in INTERACTION_FIELDS}) for c in cases),
    }


def quality_audit(cases):
    historical = [c for name in HISTORICAL_NAMES for c in json.loads((ROOT / name).read_text())]
    assert len(historical) == 1515 and len(cases) == 48
    current, previous = fingerprints(cases), fingerprints(historical)
    overlap = {key: sorted(set(values) & set(previous[key])) for key, values in current.items()}
    assert all(not values for values in overlap.values()), overlap
    for key in (
        "case_ids",
        "world_ids",
        "initial_messages",
        "instruction_skeletons",
        "public_inputs",
        "whole_interactions",
    ):
        assert len(set(current[key])) == 48, (key, "duplicate new cases")
    assert sum(len(entities(c)) for c in cases) == len(current["entities"])
    assert sum(len(catalog_labels(c)) for c in cases) == len(current["catalog_labels"])
    projections = {}
    with tempfile.TemporaryDirectory(prefix="resolvedesk-v8-eval2-isolation-") as temp:
        for flow in ("v7", "v8"):
            Store, messages = runtime(flow)
            store = Store(Path(temp) / f"{flow}.sqlite")
            new = {c["id"]: digest_value(initial_prompt(c, store, messages)) for c in cases}
            old = {digest_value(initial_prompt(c, store, messages)) for c in historical}
            assert len(set(new.values())) == 48 and not set(new.values()) & old
            projections[flow] = {"unique": 48, "historical_overlap": [], "new_prompt_sha256": new}
    return {
        "historical_datasets_checked": len(HISTORICAL_NAMES),
        "historical_cases_checked": len(historical),
        "historical_datasets_sha256": {name: sha(ROOT / name) for name in HISTORICAL_NAMES},
        "unique_case_ids": 48,
        "unique_interactions": 48,
        "unique_order_worlds": 48,
        "unique_initial_messages": 48,
        "unique_instruction_skeletons": 48,
        "unique_entity_ids": len(current["entities"]),
        "unique_catalog_labels": len(current["catalog_labels"]),
        "historical_feature_overlaps": overlap,
        "initial_model_projections": projections,
        "fingerprints": current,
        "fingerprint_algorithm": "SHA256 of sorted compact canonical JSON",
        "historical_scope": (
            "Ten earlier case datasets plus V8 development144 and original heldout48; no result files read."
        ),
        "previous_model_results_used_for_construction": False,
        "old_partial_calls_replayed_or_pooled": False,
        "authorship_scope": (
            "New wording and independent retail entities/catalog facts under the same authored policy "
            "and24 families; not an independent human, official or production benchmark."
        ),
    }


class ReferenceProvider:
    """A label-aware CPU feasibility oracle, never candidate-model evidence."""

    def __init__(self, case):
        self.case = case

    def generate(self, prompt):
        visible = json.loads(prompt[1]["content"])
        assert set(visible).isdisjoint(PRIVATE_MODEL_KEYS)
        action = copy.deepcopy(visible["decision_gate"]["next_action"])
        if action["tool"] == "record_request":
            intent = dict(self.case["intent_initial"])
            for reply in visible["clarification_memory"]:
                field = reply["field"]
                assert reply["reply"] == self.case["answers"][field], (self.case["id"], reply)
                if field != "fallback_return":
                    key = "requested_option" if field == "new_option" else field
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
        return json.dumps(action), {"provider": "cpu_reference_oracle", "latency_ms": 0}


def reference_audit(cases):
    records, by_flow = [], {}
    with tempfile.TemporaryDirectory(prefix="resolvedesk-v8-eval2-oracle-") as temp:
        for flow, variant in (("v7", "prior_weights_old_flow"), ("v8", "v8_weights_new_flow")):
            Store, messages = runtime(flow)
            store = Store(Path(temp) / f"{flow}.sqlite")
            rows = []
            for case in cases:
                record = {
                    "variant": variant,
                    **run_case(store, case, ReferenceProvider(case), messages, max_steps=12),
                }
                assert record["task_success"], (flow, case["id"], record)
                assert record["initial_enum_errors"] == {} and not record["initial_raw_parse_failure"]
                assert record["quote_rejections"] == record["repeated_question_attempts"] == 0
                assert all(field in record["questions"] for field in case["required_answers"])
                protocol = audit_record(record, case)
                for key in (
                    "return_proposals_without_scoped_yes",
                    "committed_returns_without_scoped_yes",
                    "final_returns_without_verified_commit",
                ):
                    assert protocol[key] == 0, (flow, case["id"], protocol)
                pending, latest, actual_replies, actions = None, case["initial_message"], [], []
                for event in record["events"]:
                    if event["kind"] == "customer_message":
                        text = event["detail"]["content"]
                        assert pending and text == case["answers"][pending]
                        actual_replies.append({"field": pending, "text": text, "after_actual_question": True})
                        pending, latest = None, text
                    elif event["kind"] == "model_tool":
                        action, result = event["detail"]["action"], event["detail"]["result"]
                        actions.append(action)
                        if action and action["tool"] == "record_request":
                            assert action["arguments"]["evidence"] in latest
                        if "question" in result and "error" not in result:
                            pending = result["field"]
                row = {
                    "case_id": case["id"],
                    "flow": flow,
                    "provider": "cpu_reference_oracle",
                    "task_success": record["task_success"],
                    "model_calls": record["model_calls"],
                    "questions": record["questions"],
                    "actual_replies": actual_replies,
                    "additional_question_fields": [
                        field
                        for field in record["questions"]
                        if field not in case["required_answers"] and field != "fallback_return"
                    ],
                    "actions": actions,
                    "first_raw_intent": record["intent_initial"],
                    "protocol": {key: protocol[key] for key in COUNTS},
                }
                rows.append(row)
            by_flow[flow] = {
                "tasks": len(rows),
                "all_business_references_pass": True,
                "reference_total_calls": sum(r["model_calls"] for r in rows),
                "max_reference_calls": max(r["model_calls"] for r in rows),
                "additional_question_tasks": sum(bool(r["additional_question_fields"]) for r in rows),
                "additional_question_occurrences": sum(len(r["additional_question_fields"]) for r in rows),
                "additional_questions_by_case": {
                    r["case_id"]: r["additional_question_fields"]
                    for r in rows
                    if r["additional_question_fields"]
                },
                "protocol_summary": {key: sum(r["protocol"][key] for r in rows) for key in COUNTS},
            }
            records.extend(rows)
    return {
        "cohort": "v8_eval2",
        "tasks": len(cases),
        "reference_trajectories": len(records),
        "cases_sha256": hashlib.sha256(serialized(cases)).hexdigest(),
        "public_sha256": hashlib.sha256(serialized([public_case(c) for c in cases])).hexdigest(),
        "source_inputs_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "frozen_v8_manifest_sha256": sha(ROOT / "training/v8/datasets/manifest.json"),
        "by_flow": by_flow,
        "records": records,
        "required_replies_supplied_only_after_actual_asks": True,
        "authoring_trace": copy.deepcopy(AUTHORING_TRACE),
        "business_task_success_does_not_assert_zero_extra_questions": True,
        "additional_question_definition": (
            "Actual oracle asks outside required_answers, excluding separate scoped fallback consent. "
            "Initial wording and gold stay intact even where a frozen flow asks for a known payment slot."
        ),
        "scope": (
            "96 deterministic CPU oracle trajectories (48cases in each of two frozen flows), not LLM "
            "outputs, model quality, SFT data, training or extra evaluator calls. No injected errors; "
            "actual quote provenance and scoped YES reconstructed from real oracle events."
        ),
    }


def generate():
    verify_frozen()
    assert not (REPORTS / "analysis-plan.json").exists(), "Never rewrite a planned evaluation cohort"
    assert not any((REPORTS / name).exists() for name in ("evaluation.json", "evaluation-partial.json"))
    assert not (DATA / "cases.json").exists(), "Saved cohort is immutable; do not regenerate it"
    cases = [make_case(family, n) for family in FAMILIES for n in range(2)]
    quality, reference = quality_audit(cases), reference_audit(cases)
    DATA.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "cases.json": cases,
        "public.json": [public_case(c) for c in cases],
        "reference-audit.json": reference,
    }
    for name, value in artifacts.items():
        with (DATA / name).open("xb") as handle:
            handle.write(serialized(value))
    summary = {
        "cohort": "v8_eval2",
        "evaluation_only": True,
        "model_inference_started": False,
        "training_or_validation_rerun": False,
        "previous_model_results_used_for_construction": False,
        "authoring_trace": copy.deepcopy(AUTHORING_TRACE),
        "cases_sha256": sha(DATA / "cases.json"),
        "public_sha256": sha(DATA / "public.json"),
        "reference_audit_sha256": sha(DATA / "reference-audit.json"),
        "source_inputs_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "frozen_v8_manifest_sha256": sha(ROOT / "training/v8/datasets/manifest.json"),
        "denominators": denominators(cases),
        "quality": quality,
        "cpu_reference_by_flow": reference["by_flow"],
        "scope": (
            "Fresh48 eval-only synthetic tasks, same policy/families and fixed saved weights. No old "
            "partial output reuse, retraining, checkpoint selection or model inference. Dataset hashes "
            "are frozen inputs for the parent's separate pre-inference plan, not a claim that the "
            "whole new execution is sealed here."
        ),
    }
    with (DATA / "data-summary.json").open("xb") as handle:
        handle.write(serialized(summary))
    verify_frozen()
    print(json.dumps({key: value for key, value in summary.items() if key != "quality"}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate",))
    parser.parse_args()
    generate()
