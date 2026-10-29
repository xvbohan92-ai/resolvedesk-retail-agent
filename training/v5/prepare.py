import json
import random
import tempfile
from collections import Counter
from pathlib import Path

from resolve_desk.resolution import ResolutionStore, decision_gate, messages
from training.common import ROOT, sha
from training.v2.scoring import score
from training.v5.common import CONFIG, DATA, REPORTS
from training.v5.fixtures import FAMILIES, make_case


def reference(case, state):
    gate = decision_gate(state)["next_action"]
    if gate["tool"] == "record_request":
        intent = case["intent_initial"] if len(state["messages"]) == 1 else case["intent_complete"]
        return {
            "tool": "record_request",
            "arguments": {
                **intent,
                "evidence": case["evidence_initial"]
                if len(state["messages"]) == 1
                else state["messages"][-1]["content"],
            },
        }
    return gate


def generate():
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    assert not (DATA / "manifest.json").exists(), "Never overwrite a frozen experiment"
    cases = [
        make_case(split, family, n)
        for split, count in [("train", 4), ("valid", 2), ("test", 2)]
        for family in FAMILIES
        for n in range(count)
    ]
    (DATA / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    rows, audits = {s: [] for s in ["train", "valid", "test"]}, []
    with tempfile.TemporaryDirectory() as tmp:
        store = ResolutionStore(Path(tmp) / "references.sqlite")
        for case in cases:
            state, questions = store.create(case), []
            for _ in range(12):
                if state["status"] == "awaiting_customer":
                    field = state["observations"][-1]["result"]["field"]
                    questions.append(field)
                    state = store.customer_message(state["id"], case["answers"][field], state["revision"])
                if state["status"] != "running":
                    break
                action = reference(case, state)
                rows[case["split"]].append(
                    {
                        "case_id": case["id"],
                        "origin": action["tool"],
                        "messages": [
                            *messages(state),
                            {"role": "assistant", "content": json.dumps(action, separators=(",", ":"))},
                        ],
                    }
                )
                state = store.apply(
                    state["id"], json.dumps(action), {"provider": "reference"}, state["revision"]
                )
                result = state["observations"][-1]["result"]
                assert "error" not in result or (
                    case["family"] == "foreign_order" and action["tool"] == "get_order"
                ), result
            before, proposal = store.orders(state["id"]), state["proposal"]
            if state["status"] == "awaiting_confirmation":
                store.confirm(state["id"], proposal["id"], state["revision"], case["confirmation"])
                state = store.get(state["id"])
            judged = score(
                case, {**state, "proposal": proposal}, store.orders(state["id"]), before, questions
            )
            assert judged["task_success"], (case["id"], judged, state)
            audits.append({"case_id": case["id"], "model_decisions": len(state["observations"]), **judged})
    import copy

    contrasts = [
        r
        for r in rows["train"]
        if r["origin"] == "record_request"
        and any(
            "-" + f + "-" in r["case_id"]
            for f in [
                "exchange_fallback",
                "exchange_fallback_available",
                "exchange_unavailable",
                "exchange_shortfall",
            ]
        )
    ]
    for source in contrasts:
        extra = copy.deepcopy(source)
        extra["origin"] = "record_request_contrast_repeat"
        rows["train"].append(extra)
    random.Random(202610275).shuffle(rows["train"])
    for split, samples in rows.items():
        (DATA / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    (REPORTS / "reference-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    valid = [
        {k: c[k] for k in ("id", "family", "initial_message", "customer", "orders", "products")}
        for c in cases
        if c["split"] == "valid"
    ]
    (ROOT / "resolve_desk/fixtures/validation-v5.json").write_text(json.dumps(valid, indent=2) + "\n")
    (REPORTS / "data-summary.json").write_text(
        json.dumps(
            {
                "tasks": dict(Counter(c["split"] for c in cases)),
                "rows": {s: len(samples) for s, samples in rows.items()},
                "duplicated_contrast_snapshots": len(contrasts),
                "train_actions": dict(Counter(r["origin"] for r in rows["train"])),
                "all_reference_tasks_pass": True,
                "scope": (
                    "Self-authored synthetic scenarios. Counterfactual worlds share IDs within a split only."
                ),
            },
            indent=2,
        )
        + "\n"
    )


def freeze():
    assert not (REPORTS / "evaluation-partial.json").exists()
    assert not (DATA / "manifest.json").exists()
    files = [
        *sorted((ROOT / "training/v5").glob("*.py")),
        CONFIG,
        ROOT / "resolve_desk/resolution.py",
        ROOT / "resolve_desk/agent_v5.py",
        ROOT / "training/base-model-spec.json",
        ROOT / "training/runtime.py",
        ROOT / "training/scoring.py",
        ROOT / "training/v2/scoring.py",
        ROOT / "training/v4/tokenization.py",
        ROOT / "training/v4/evaluation.py",
        ROOT / "training/v3/evaluation.py",
        ROOT / "training/v2/evaluation.py",
        ROOT / "resolve_desk/progress.py",
        ROOT / "resolve_desk/workflow.py",
        ROOT / "resolve_desk/store.py",
        ROOT / "resolve_desk/protocol.py",
        *sorted(DATA.glob("*.jsonl")),
        DATA / "cases.json",
    ]
    excluded = {
        "visualize.py",
        "verify_results.py",
        "serve.py",
        "write_summary.py",
        "integration.py",
        "check_package.py",
    }
    files = [p for p in files if p.name not in excluded]
    previous = json.loads((ROOT / "reports/v4/training.json").read_text())
    manifest = {
        "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "data": json.loads((REPORTS / "data-summary.json").read_text()),
        "previous_training_sha256": sha(ROOT / "reports/v4/training.json"),
        "initial_adapter_sha256": previous["selected_adapter_sha256"],
        "test_inference_started_at_freeze": False,
        "scope": (
            "V4 test is development evidence. Fresh v5 test is held out until validation-only selection."
        ),
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["data"]), flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["generate", "freeze"])
    generate() if p.parse_args().mode == "generate" else freeze()
