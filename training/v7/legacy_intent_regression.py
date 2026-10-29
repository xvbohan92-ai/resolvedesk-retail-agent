"""Development-only legacy first-output retention; never a fresh benchmark or selection input."""

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

from resolve_desk.resolution_v7 import ResolutionStore, messages
from training.common import ROOT, sha
from training.v7.common import DATA, REPORTS, verify_frozen
from training.v7.evaluation import ARMS, REQUEST_FIELDS, enum_errors, raw_action

PLAN = REPORTS / "legacy-intent-regression-plan.json"
OUTPUT = REPORTS / "legacy-intent-regression.json"
ARCHIVED_ARM = "qlora_v6_1"
ARCHIVE = ROOT / "reports/v6_1/evaluation.json"
OLD_CASES = ROOT / "training/v6_1/datasets/cases.json"
SERVICE = "http://127.0.0.1:8772"
SCOPE = (
    "DEVELOPMENT regression on all 48 previously published V6.1 test interactions, not fresh or held-out "
    "evidence. Compare the archived selected V6.1 FIRST raw model_tool output with one V7 /generate "
    "response per original case. Workflow and weights both differ; no isolated training attribution. "
    "No continuation, action execution, correction, selection, gate change or pooled benchmark score."
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def prompt_hash(prompt):
    return hashlib.sha256(canonical(prompt).encode()).hexdigest()


def first_metrics(raw, case):
    """Use the frozen V7 FIRST-only structural and enum rules without backend repair."""
    action = raw_action(raw)
    args = action.get("arguments") if action else None
    intent = (
        {k: v for k, v in args.items() if k != "evidence"}
        if action
        and action.get("tool") == "record_request"
        and isinstance(args, dict)
        and set(args) == set(REQUEST_FIELDS)
        and all(isinstance(v, str) for v in args.values())
        else None
    )
    errors = enum_errors(action)
    gold = case["intent_initial"]
    return {
        "intent_initial": intent,
        "initial_raw_parse_failure": intent is None,
        "initial_enum_errors": errors,
        "intent_enum_valid": intent is not None and not errors,
        "permission_authorized": gold["fallback_return"] == "yes",
        "permission_yes": bool(intent and intent["fallback_return"] == "yes"),
        "intent_field_correct": {k: bool(intent and intent[k] == value) for k, value in gold.items()},
    }


def legacy_cases():
    cases = [c for c in json.loads(OLD_CASES.read_text()) if c["split"] == "test"]
    assert len(cases) == len({c["id"] for c in cases}) == 48
    assert Counter(c["intent_initial"]["fallback_return"] for c in cases) == {
        "yes": 6,
        "no": 12,
        "unknown": 30,
    }
    return cases


def make_prompt(case):
    # Explicit projection prevents gold labels and scripted replies from entering even temporary state.
    public = {k: copy.deepcopy(case[k]) for k in ["id", "initial_message", "customer", "orders", "products"]}
    with tempfile.TemporaryDirectory(prefix="resolvedesk-legacy-first-") as tmp:
        store = ResolutionStore(Path(tmp) / "sandbox.sqlite")
        prompt = messages(store.create(public))
    visible = json.loads(prompt[1]["content"])
    assert [m["role"] for m in prompt] == ["system", "user"]
    assert visible["workflow_version"] == "v7"
    assert visible["messages"] == [{"role": "customer", "content": case["initial_message"]}]
    return prompt


def archived_rows(cases):
    report = json.loads(ARCHIVE.read_text())
    records = [r for r in report["records"] if r["variant"] == ARCHIVED_ARM]
    index = {r["case_id"]: r for r in records}
    assert len(records) == len(index) == len(cases) and set(index) == {c["id"] for c in cases}
    rows = []
    for case in cases:
        record = index[case["id"]]
        first = next(e["detail"] for e in record["events"] if e["kind"] == "model_tool")
        assert first == record["trace"][0] and isinstance(first["raw"], str)
        assert first["timing"]["provider"] == "mlx" and first["timing"]["variant"] == ARCHIVED_ARM
        rows.append(
            {
                "case_id": case["id"],
                "raw": first["raw"],
                "timing": first["timing"],
                "response_valid": True,
                "metrics": first_metrics(first["raw"], case),
            }
        )
    return rows


def summarize(rows, cases):
    expected = {c["id"]: c for c in cases}
    assert len({r["case_id"] for r in rows}) == len(rows)
    available = [r for r in rows if r["response_valid"]]
    fields = list(cases[0]["intent_initial"])
    return {
        "cohort_tasks": len(cases),
        "attempted_tasks": len(rows),
        "raw_responses_available": len(available),
        "unavailable_raw_responses": len(rows) - len(available),
        "authorized_tasks": sum(c["intent_initial"]["fallback_return"] == "yes" for c in cases),
        "non_authorized_tasks": sum(c["intent_initial"]["fallback_return"] != "yes" for c in cases),
        "true_permission_yes": sum(
            r["metrics"]["permission_authorized"] and r["metrics"]["permission_yes"] for r in available
        ),
        "unsupported_permission_yes": sum(
            not r["metrics"]["permission_authorized"] and r["metrics"]["permission_yes"] for r in available
        ),
        "initial_raw_parse_failures": sum(r["metrics"]["initial_raw_parse_failure"] for r in available),
        "initial_enum_error_tasks": sum(bool(r["metrics"]["initial_enum_errors"]) for r in available),
        "initial_reason_enum_error_tasks": sum(
            "reason" in r["metrics"]["initial_enum_errors"] for r in available
        ),
        "intent_field_correct": {
            k: sum(r["metrics"]["intent_field_correct"][k] for r in available) for k in fields
        },
        "intent_joint_correct": sum(all(r["metrics"]["intent_field_correct"].values()) for r in available),
        "missing_case_ids": [cid for cid in expected if cid not in {r["case_id"] for r in rows}],
    }


def inputs():
    names = [
        "training/v7/legacy_intent_regression.py",
        "tests/test_v7_legacy_intent_regression.py",
        "training/v7/evaluation.py",
        "resolve_desk/resolution_v7.py",
        "training/v7/serve.py",
        "training/base-model-spec.json",
        "training/v6_1/datasets/cases.json",
        "reports/v6_1/evaluation.json",
        "reports/v6_1/training.json",
        "training/v6_1/datasets/manifest.json",
        "reports/v7/training.json",
        "training/v7/datasets/manifest.json",
    ]
    return {name: sha(ROOT / name) for name in names}


def write_plan():
    verify_frozen()
    assert not PLAN.exists() and not OUTPUT.exists(), "Never replace a pre-inference retention plan or result"
    cases = legacy_cases()
    current = json.loads((REPORTS / "training.json").read_text())
    prior = json.loads((ROOT / "reports/v6_1/training.json").read_text())
    archive = json.loads(ARCHIVE.read_text())
    initializer = ROOT / current["config"]["initialization"] / "adapters.safetensors"
    selected = ROOT / current["config"]["best_path"] / "adapters.safetensors"
    assert archive["training_sha256"] == sha(ROOT / "reports/v6_1/training.json")
    assert sha(initializer) == current["initialization_sha256"] == prior["selected_adapter_sha256"]
    assert sha(selected) == current["selected_adapter_sha256"]
    plan = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": SCOPE,
        "retention_inference_started": False,
        "source_inputs_sha256": inputs(),
        "case_ids": [c["id"] for c in cases],
        "permission_cohort": {"yes": 6, "no": 12, "unknown": 30, "non_authorized": 42},
        "archived_variant": ARCHIVED_ARM,
        "current_variant": "qlora_v7",
        "model_interface": "v7",
        "initialization_sha256": sha(initializer),
        "selected_adapter_sha256": sha(selected),
        "selected_iteration": current["best_checkpoint"]["iteration"],
        "base": current["config"]["model_repo"],
        "base_revision": current["config"]["model_revision"],
        "initialization_path": str(initializer.relative_to(ROOT)),
        "selected_path": str(selected.relative_to(ROOT)),
        "new_prompt_sha256": {c["id"]: prompt_hash(make_prompt(c)) for c in cases},
        "archived_first_output_summary": summarize(archived_rows(cases), cases),
        "service": SERVICE,
        "max_generate_requests": 48,
        "requests_per_case": 1,
        "main_heldout_must_complete_before_generate": True,
        "checkpoint_selection_modified": False,
        "business_gate_modified": False,
        "no_correction_or_continuation": True,
        "resume_only_untouched_cases": True,
    }
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def validate_plan():
    verify_frozen()
    plan = json.loads(PLAN.read_text())
    assert plan["source_inputs_sha256"] == inputs(), "Retention source/inputs changed after its plan"
    assert not plan["retention_inference_started"]
    assert not plan["checkpoint_selection_modified"] and not plan["business_gate_modified"]
    assert sha(ROOT / plan["initialization_path"]) == plan["initialization_sha256"]
    assert sha(ROOT / plan["selected_path"]) == plan["selected_adapter_sha256"]
    cases = legacy_cases()
    assert plan["case_ids"] == [c["id"] for c in cases] and plan["max_generate_requests"] == len(cases) == 48
    assert plan["new_prompt_sha256"] == {c["id"]: prompt_hash(make_prompt(c)) for c in cases}
    # This probe runs only after the frozen main held-out experiment has finished, without influencing it.
    main = json.loads((REPORTS / "evaluation.json").read_text())
    test_ids = {c["id"] for c in json.loads((DATA / "cases.json").read_text()) if c["split"] == "test"}
    assert Counter((r["variant"], r["case_id"]) for r in main["records"]) == Counter(
        {(arm, cid): 1 for arm in ARMS for cid in test_ids}
    )
    assert len(test_ids) == 48 and main["training_sha256"] == sha(REPORTS / "training.json")
    return plan, cases


def check_identity(identity, plan):
    for key, value in {
        "provider": "mlx",
        "variant": "qlora_v7",
        "workflow": "v7",
        "model_interface": "v7",
        "adapter_sha256": plan["selected_adapter_sha256"],
        "training_report_sha256": plan["source_inputs_sha256"]["reports/v7/training.json"],
        "manifest_sha256": plan["source_inputs_sha256"]["training/v7/datasets/manifest.json"],
        "base": plan["base"],
        "base_revision": plan["base_revision"],
        "cloud_api": False,
    }.items():
        assert identity[key] == value, (key, identity.get(key), value)


def execute_calls(client, plan, cases, baseline, destination, *, resume=False):
    """One request per untouched case; an attempted transport failure is retained and never retried."""
    previous = []
    if destination.exists():
        assert resume, "Existing retention evidence requires --resume; do not overwrite it"
        original = json.loads(destination.read_text())
        assert original["plan_sha256"] == sha(PLAN) and original["status"] == "partial"
        assert original["source_inputs_sha256"] == plan["source_inputs_sha256"]
        assert original["selected_adapter_sha256"] == plan["selected_adapter_sha256"]
        assert original["archived_records"] == baseline
        previous = original["records"]
        assert len({r["case_id"] for r in previous}) == len(previous)
        assert [r["case_id"] for r in previous] == plan["case_ids"][: len(previous)]
        case_index = {c["id"]: c for c in cases}
        for row in previous:
            assert row["attempted_generate_requests"] == 1
            assert row["request_sha256"] == plan["new_prompt_sha256"][row["case_id"]]
            assert prompt_hash(row["request_messages"]) == row["request_sha256"]
            if row["response_valid"]:
                assert row["metrics"] == first_metrics(row["raw"], case_index[row["case_id"]])
        snapshot = destination.with_name(f"legacy-intent-before-resume-{len(previous)}.json")
        assert not snapshot.exists()
        shutil.copyfile(destination, snapshot)
    identity = client.get("/health")
    identity.raise_for_status()
    health = identity.json()
    check_identity(health, plan)
    output = {
        "scope": SCOPE,
        "plan_sha256": sha(PLAN),
        "source_inputs_sha256": plan["source_inputs_sha256"],
        "selected_adapter_sha256": plan["selected_adapter_sha256"],
        "identity": health,
        "status": "partial",
        "archived_records": baseline,
        "records": copy.deepcopy(previous),
        "checkpoint_selection_modified": False,
        "business_gate_modified": False,
    }

    def save():
        output["summary"] = {
            "archived_v6_1": summarize(baseline, cases),
            "current_v7": summarize(output["records"], cases),
        }
        destination.write_text(json.dumps(output, indent=2) + "\n")

    done = {r["case_id"] for r in previous}
    for case in cases:
        if case["id"] in done:
            continue
        prompt = make_prompt(case)
        assert prompt_hash(prompt) == plan["new_prompt_sha256"][case["id"]]
        row = {
            "case_id": case["id"],
            "started_at_utc": datetime.now(UTC).isoformat(),
            "request_messages": prompt,
            "request_sha256": prompt_hash(prompt),
            "response_valid": False,
            "raw": None,
            "metrics": None,
            "attempted_generate_requests": 1,
        }
        output["records"].append(row)
        save()  # Record the attempted case before I/O; an ambiguous interrupted request is never retried.
        try:
            response = client.post("/generate", json={"messages": prompt})
            row["response_status"] = response.status_code
            row["response_body"] = response.text
            save()
            response.raise_for_status()
            body = response.json()
            row["response_json"] = body
            assert isinstance(body, dict) and isinstance(body.get("text"), str)
            timing = body["timing"]
            assert timing["provider"] == "mlx" and timing["variant"] == "qlora_v7"
            assert timing["adapter_sha256"] == plan["selected_adapter_sha256"]
            row.update(
                raw=body["text"],
                timing=timing,
                response_valid=True,
                metrics=first_metrics(body["text"], case),
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AssertionError) as error:
            row["transport_or_service_error"] = f"{type(error).__name__}: {error}"
            save()
            raise
        save()
        print(json.dumps({"development_legacy_case": case["id"], "first_output_retained": True}), flush=True)
    output["status"] = "complete"
    output["completed_at_utc"] = datetime.now(UTC).isoformat()
    assert len(output["records"]) == len(cases)
    save()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.plan:
        assert not args.resume
        plan = write_plan()
        print(
            json.dumps(
                {
                    "plan_written": str(PLAN),
                    "plan_sha256": sha(PLAN),
                    "baseline": plan["archived_first_output_summary"],
                }
            )
        )
        return
    plan, cases = validate_plan()
    with httpx.Client(base_url=SERVICE, trust_env=False, timeout=240) as client:
        result = execute_calls(client, plan, cases, archived_rows(cases), OUTPUT, resume=args.resume)
    print(json.dumps({"status": result["status"], "scope": result["scope"], "summary": result["summary"]}))


if __name__ == "__main__":
    main()
