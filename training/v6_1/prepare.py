import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from resolve_desk.progress import clarification_memory
from resolve_desk.resolution_v6_1 import ResolutionStore, decision_gate, messages
from training.common import ROOT, sha
from training.v6_1.common import CONFIG, DATA, REPORTS
from training.v6_1.fixtures import FAMILIES, make_case
from training.v6_1.scoring import score


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
        make_case(s, f, n) for s, count in [("valid", 2), ("test", 2)] for f in FAMILIES for n in range(count)
    ]
    (DATA / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    rows, audits = {s: [] for s in ["valid", "test"]}, []
    recoveries = {
        "exchange_unavailable",
        "exchange_stockout_missing_email",
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
    for split, samples in rows.items():
        (DATA / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    (REPORTS / "reference-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    valid = [
        {k: c[k] for k in ("id", "family", "initial_message", "customer", "orders", "products")}
        for c in cases
        if c["split"] == "valid"
    ]
    (ROOT / "resolve_desk/fixtures/validation-v6_1.json").write_text(json.dumps(valid, indent=2) + "\n")
    summary = {
        "tasks": dict(Counter(c["split"] for c in cases)),
        "rows": {s: len(r) for s, r in rows.items()},
        "recovery_trajectories": sum(a["trajectory"] == "recovery" for a in audits),
        "train_actions": dict(Counter(r["origin"] for r in rows.get("train", []))),
        "all_reference_tasks_pass": True,
        "scope": (
            "Self-authored synthetic worlds and language. Intentional within-split counterfactual sharing; "
            "recovery states are not new tasks or wrong supervised labels."
        ),
    }
    (REPORTS / "data-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def freeze():
    from training.v6.common import verify_frozen as verify_training

    verify_training()
    assert not (DATA / "manifest.json").exists()
    names = [
        "common.py",
        "fixtures.py",
        "scoring.py",
        "prepare.py",
        "evaluation.py",
        "evaluate.py",
        "select.py",
        "provider.py",
        "resources.py",
    ]
    files = [ROOT / "training/v6_1" / n for n in names]
    files += [
        CONFIG,
        ROOT / "resolve_desk/resolution_v6_1.py",
        ROOT / "resolve_desk/agent_v6_1.py",
        ROOT / "resolve_desk/resolution.py",
        ROOT / "training/runtime.py",
        ROOT / "training/v6/datasets/manifest.json",
        ROOT / "reports/v6/updates.json",
        REPORTS / "analysis-plan.json",
        ROOT / "training/v6/scoring.py",
        *sorted(DATA.glob("*.jsonl")),
        DATA / "cases.json",
    ]
    m = {
        "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "data": json.loads((REPORTS / "data-summary.json").read_text()),
        "training_manifest_sha256": sha(ROOT / "training/v6/datasets/manifest.json"),
        "training_updates_sha256": sha(ROOT / "reports/v6/updates.json"),
        "initial_adapter_sha256": json.loads((ROOT / "reports/v6/updates.json").read_text())[
            "initialization_sha256"
        ],
        "test_inference_started_at_freeze": False,
        "scope": (
            "Evaluation/controller freeze after V6 validation-driven development. "
            "SFT weights are unchanged V6 candidates. No prior held-out inference is used."
        ),
    }
    (DATA / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m["data"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "freeze"])
    generate() if parser.parse_args().mode == "generate" else freeze()
