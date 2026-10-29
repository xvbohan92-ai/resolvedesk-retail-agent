import argparse
import json
import random
import tempfile
from collections import Counter
from pathlib import Path

from resolve_desk.progress import clarification_memory
from resolve_desk.resolution_v6 import ResolutionStore, decision_gate, messages
from training.common import ROOT, sha
from training.v6.common import CONFIG, DATA, REPORTS
from training.v6.fixtures import FAMILIES, make_case
from training.v6.scoring import score


def reference(case, state):
    gate = decision_gate(state)["next_action"]
    if gate["tool"] != "record_request":
        return gate
    intent = dict(case["intent_initial"])
    for p in clarification_memory(state):
        key = "requested_option" if p["field"] == "new_option" else p["field"]
        if key != "fallback_return":
            intent[key] = case["intent_complete"][key]
    return {
        "tool": "record_request",
        "arguments": {
            **intent,
            "evidence": case["evidence_initial"]
            if len(state["messages"]) == 1
            else state["messages"][-1]["content"],
        },
    }


def generate():
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    assert not (DATA / "manifest.json").exists(), "Never overwrite a frozen experiment"
    cases = [
        make_case(s, f, n)
        for s, count in [("train", 4), ("valid", 2), ("test", 2)]
        for f in FAMILIES
        for n in range(count)
    ]
    (DATA / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    rows, audits = {s: [] for s in ["train", "valid", "test"]}, []
    recoveries = {
        "exchange_unavailable",
        "exchange_stockout_refusal",
        "exchange_stockout_negative_quote",
        "exchange_stockout_question",
    }
    with tempfile.TemporaryDirectory() as tmp:
        store = ResolutionStore(Path(tmp) / "references.sqlite")

        def walk(case, recovery=False):
            state, questions = store.create(case), []
            if recovery:
                wrong = reference(case, state)
                wrong["arguments"]["fallback_return"] = "yes"
                # This intentionally wrong parser output is input state only,
                # never a supervised target. Train subsequent safe recovery.
                state = store.apply(
                    state["id"], json.dumps(wrong), {"provider": "synthetic_error"}, state["revision"]
                )
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
                        "trajectory": "false_yes_recovery" if recovery else "clean_reference",
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
                ), (case["id"], result)
            before, proposal = store.orders(state["id"]), state["proposal"]
            if state["status"] == "awaiting_confirmation":
                store.confirm(state["id"], proposal["id"], state["revision"], case["confirmation"])
                state = store.get(state["id"])
            judged = score(
                case, {**state, "proposal": proposal}, store.orders(state["id"]), before, questions
            )
            assert judged["task_success"], (case["id"], judged, state)
            audits.append(
                {"case_id": case["id"], "trajectory": "recovery" if recovery else "clean", **judged}
            )

        for case in cases:
            walk(case)
            if case["split"] == "train" and case["family"] in recoveries:
                walk(case, recovery=True)
    random.Random(202610286).shuffle(rows["train"])
    for split, samples in rows.items():
        (DATA / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    (REPORTS / "reference-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    valid = [
        {k: c[k] for k in ("id", "family", "initial_message", "customer", "orders", "products")}
        for c in cases
        if c["split"] == "valid"
    ]
    (ROOT / "resolve_desk/fixtures/validation-v6.json").write_text(json.dumps(valid, indent=2) + "\n")
    summary = {
        "tasks": dict(Counter(c["split"] for c in cases)),
        "rows": {s: len(r) for s, r in rows.items()},
        "recovery_trajectories": sum(a["trajectory"] == "recovery" for a in audits),
        "train_actions": dict(Counter(r["origin"] for r in rows["train"])),
        "all_reference_tasks_pass": True,
        "scope": (
            "Self-authored synthetic worlds and language. Intentional within-split counterfactual sharing; "
            "recovery states are not new tasks or wrong supervised labels."
        ),
    }
    (REPORTS / "data-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def freeze():
    assert not (REPORTS / "evaluation-partial.json").exists()
    assert not (DATA / "manifest.json").exists()
    helpers = {
        "visualize.py",
        "verify_results.py",
        "serve.py",
        "write_summary.py",
        "integration.py",
        "check_package.py",
    }
    files = [p for p in sorted((ROOT / "training/v6").glob("*.py")) if p.name not in helpers]
    files += [
        CONFIG,
        ROOT / "resolve_desk/resolution_v6.py",
        ROOT / "resolve_desk/agent_v6.py",
        ROOT / "resolve_desk/resolution.py",
        ROOT / "resolve_desk/progress.py",
        ROOT / "resolve_desk/workflow.py",
        ROOT / "resolve_desk/store.py",
        ROOT / "resolve_desk/protocol.py",
        ROOT / "training/base-model-spec.json",
        ROOT / "training/runtime.py",
        ROOT / "training/scoring.py",
        ROOT / "training/v2/scoring.py",
        ROOT / "training/v4/tokenization.py",
        ROOT / "training/v4/evaluation.py",
        ROOT / "training/v3/evaluation.py",
        ROOT / "training/v2/evaluation.py",
        ROOT / "training/v5/fixtures.py",
        ROOT / "training/v4/fixtures.py",
        *sorted(DATA.glob("*.jsonl")),
        DATA / "cases.json",
    ]
    previous = json.loads((ROOT / "reports/v5/training.json").read_text())
    manifest = {
        "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "data": json.loads((REPORTS / "data-summary.json").read_text()),
        "previous_training_sha256": sha(ROOT / "reports/v5/training.json"),
        "initial_adapter_sha256": previous["selected_adapter_sha256"],
        "test_inference_started_at_freeze": False,
        "scope": (
            "V5 tests are development evidence. V6 held-out inference starts only after validation selection."
        ),
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["data"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "freeze"])
    generate() if parser.parse_args().mode == "generate" else freeze()
