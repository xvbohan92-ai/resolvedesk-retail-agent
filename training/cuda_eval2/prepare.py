"""Construct the separate second cohort and audit it without model inference.

Only pure policy/schema helpers are reused. No old DATA/REPORTS global is
modified and no earlier artifact is written. Prior candidate predictions,
success rates and interrupted model traces are not construction inputs.
"""

import argparse
import json
import tempfile
from pathlib import Path

from resolve_desk.resolution_v7 import ResolutionStore
from training.common import ROOT, sha
from training.cuda_eval.prepare import (
    FORBIDDEN_MODEL_KEYS,
    INTERACTION_FIELDS,
    PUBLIC_FIELDS,
    canonical,
    catalog_labels,
    denominators,
    digest_value,
    entities,
    initial_prompt,
    instruction_skeleton,
    public_case,
)
from training.cuda_eval.prepare import (
    reference_audit as policy_reference_audit,
)
from training.cuda_eval2.fixtures import DATA, FAMILIES, make_case
from training.v7.common import verify_frozen

REPORTS = ROOT / "reports/cuda-eval2"
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
)

__all__ = [
    "FORBIDDEN_MODEL_KEYS",
    "PUBLIC_FIELDS",
    "denominators",
    "generate",
    "initial_prompt",
    "public_case",
    "quality_audit",
    "reference_audit",
]


def quality_audit(cases):
    paths = [ROOT / name for name in HISTORICAL_NAMES]
    historical = [case for path in paths for case in json.loads(path.read_text())]
    assert len(historical) == 1275, "Expected the eight historical datasets plus the first CUDA48"
    assert sha(paths[-1]) == "7d8c7987881da5ec36db32f66a038a22aabba146c20fae3f6151a213a6f20eb0"
    case_ids = [case["id"] for case in cases]
    old_ids = {case["id"] for case in historical}
    entity_sets = [entities(case) for case in cases]
    new_entities = set.union(*entity_sets)
    prior_entities = set.union(*(entities(case) for case in historical))
    worlds = {order["id"] for case in cases for order in case["orders"]}
    prior_worlds = {order["id"] for case in historical for order in case["orders"]}
    instructions = [instruction_skeleton(case) for case in cases]
    prior_instructions = {instruction_skeleton(case) for case in historical}
    initial_messages = {case["initial_message"] for case in cases}
    prior_messages = {case["initial_message"] for case in historical}
    catalogs = set.union(*(catalog_labels(case) for case in cases))
    prior_catalogs = set.union(*(catalog_labels(case) for case in historical))
    assert len(case_ids) == len(set(case_ids)) == len(worlds) == len(cases) == 48
    assert len(set(instructions)) == len(initial_messages) == 48
    assert sum(map(len, entity_sets)) == len(new_entities), "Each case requires its own independent entities"
    overlaps = {
        "prior_case_id_overlap": set(case_ids) & old_ids,
        "prior_world_overlap": worlds & prior_worlds,
        "prior_entity_overlap": new_entities & prior_entities,
        "prior_initial_message_overlap": initial_messages & prior_messages,
        "prior_instruction_skeleton_overlap": set(instructions) & prior_instructions,
        "prior_catalog_label_overlap": catalogs & prior_catalogs,
    }
    assert all(not values for values in overlaps.values()), overlaps
    with tempfile.TemporaryDirectory(prefix="resolvedesk-cuda-eval2-input-audit-") as tmp:
        store = ResolutionStore(Path(tmp) / "prompts.sqlite")
        prompt_hashes = {case["id"]: digest_value(initial_prompt(case, store)) for case in cases}
        old_prompt_hashes = {digest_value(initial_prompt(case, store)) for case in historical}
    assert len(set(prompt_hashes.values())) == 48
    prompt_overlap = set(prompt_hashes.values()) & old_prompt_hashes
    assert not prompt_overlap
    interactions = {canonical({key: case[key] for key in INTERACTION_FIELDS}) for case in cases}
    assert len(interactions) == 48
    return {
        "historical_datasets_checked": len(paths),
        "historical_cases_checked": len(historical),
        "historical_scope": (
            "Eight prior datasets (1227 rows) plus all 48 cases of the interrupted first CUDA "
            "evaluation; no outputs or success rates read."
        ),
        "historical_datasets_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "unique_case_ids": len(set(case_ids)),
        "unique_interactions": len(interactions),
        "unique_order_worlds": len(worlds),
        "unique_initial_model_prompts": len(set(prompt_hashes.values())),
        "unique_entity_ids": len(new_entities),
        "unique_instruction_skeletons": len(set(instructions)),
        "initial_prompt_sha256": prompt_hashes,
        "duplicate_case_id_groups": [],
        "duplicate_interaction_groups": [],
        "duplicate_initial_prompt_groups": [],
        "duplicate_instruction_skeleton_groups": [],
        **{key: sorted(values) for key, values in overlaps.items()},
        "prior_v7_projection_prompt_overlap": sorted(prompt_overlap),
        "prompt_comparison_scope": (
            "Actual new initial prompts and all 1275 older case inputs projected using the "
            "unchanged V7 model-facing messages; old workflow scores are not reclassified."
        ),
        "gold_and_scripted_answers_excluded_from_model_projection": True,
        "authorship_scope": (
            "New English wording, independent entities and new catalog labels under the same "
            "authored policy/families; not an independent human, official or production "
            "benchmark."
        ),
        "previous_model_results_used_for_construction": False,
    }


def reference_audit(cases):
    audit = policy_reference_audit(cases)
    audit["cohort"] = "cuda_eval2"
    audit["reference_helper_sha256"] = sha(ROOT / "training/cuda_eval/prepare.py")
    audit["scope"] = (
        "Second-cohort deterministic CPU oracle feasibility, not model predictions, SFT "
        "data or a model-quality result. Frozen V7 controller/grader/protocol remain "
        "unchanged."
    )
    return audit


def generate():
    verify_frozen()
    assert not (REPORTS / "analysis-plan.json").exists(), "Never change a planned separate evaluation"
    assert not (REPORTS / "evaluation.json").exists() and not (REPORTS / "evaluation-partial.json").exists()
    cases = [make_case(family, n) for family in FAMILIES for n in range(2)]
    quality = quality_audit(cases)
    reference = reference_audit(cases)
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (DATA / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (DATA / "public.json").write_text(json.dumps([public_case(case) for case in cases], indent=2) + "\n")
    reference["cases_sha256"] = sha(DATA / "cases.json")
    (REPORTS / "reference-audit.json").write_text(json.dumps(reference, indent=2) + "\n")
    sources = (
        "training/cuda_eval2/__init__.py",
        "training/cuda_eval2/fixtures.py",
        "training/cuda_eval2/prepare.py",
        "tests/test_cuda_eval2_data.py",
        "training/cuda_eval/fixtures.py",
        "training/cuda_eval/prepare.py",
    )
    summary = {
        "scope": (
            "A separately authorized evaluation-only cohort authored before any second-run "
            "model inference. Same fixed models/policy/budget; no new training and no "
            "previous partial scores used to choose cases."
        ),
        "model_inference_started": False,
        "cases_sha256": sha(DATA / "cases.json"),
        "public_sha256": sha(DATA / "public.json"),
        "reference_audit_sha256": sha(REPORTS / "reference-audit.json"),
        "source_inputs_sha256": {name: sha(ROOT / name) for name in sources},
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
