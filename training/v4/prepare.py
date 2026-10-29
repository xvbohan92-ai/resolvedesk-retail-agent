# ruff: noqa: E501 -- Authored synthetic utterances retain full source lines.
import json
import random
import tempfile
from collections import Counter
from pathlib import Path

from resolve_desk.progress import ProgressStore, messages
from training.common import ROOT, sha
from training.v3.prepare import finish
from training.v4.common import DATA, REPORTS
from training.v4.fixtures import FAMILIES, make_case, next_reference


def apply(store, state, action):
    return store.apply(state["id"], json.dumps(action), {"provider": "reference_only"}, state["revision"])


def row(state, action, origin):
    return {
        "messages": messages(state)
        + [{"role": "assistant", "content": json.dumps(action, separators=(",", ":"))}],
        "origin": origin,
    }


def reply(store, state, case, field):
    state = store.customer_message(state["id"], case["answers"][field], state["revision"])
    # Used by the reference author only. messages() never exposes this annotation.
    state["messages"][-1]["field"] = field
    with store.connect() as c:
        c.execute("UPDATE sessions SET body=? WHERE id=?", (json.dumps(state), state["id"]))
    return state


def prefix(store, case):
    state, examples, questions = store.create(case), [], []
    for _ in range(10):
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            questions.append(field)
            state = reply(store, state, case, field)
        action = next_reference(case, state)
        if action["tool"].startswith("propose_") or action["tool"] == "handoff":
            return state, action, examples, questions
        examples.append(row(state, action, "complete_prefix"))
        state = apply(store, state, action)
    raise ValueError("Reference did not terminate")


def main():
    if (DATA / "manifest.json").exists():
        raise ValueError("V4 already frozen")
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    cases = [
        make_case(s, f, n)
        for s, count in [("train", 4), ("valid", 2), ("test", 2)]
        for f in FAMILIES
        for n in range(count)
    ]
    (DATA / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    rows, audits = {s: [] for s in ("train", "valid", "test")}, []
    with tempfile.TemporaryDirectory() as tmp:
        store = ProgressStore(Path(tmp) / "reference.sqlite")
        for case in cases:
            state, action, upstream, questions = prefix(store, case)
            terminal = row(state, action, "clean_terminal")
            rows[case["split"]].extend([*upstream, terminal])
            if case["split"] == "train" and case["expected"]["kind"] in {"handoff", "exchange"}:
                rows["train"].append(terminal)
            audits.append(
                {"case_id": case["id"], "origin": "clean", **finish(store, case, state, action, questions)}
            )
            if case["split"] != "train":
                continue
            if case["required_clarification"]:
                state, action, _, questions = prefix(store, case)
                asked = next(
                    o["action"] for o in state["observations"] if o["action"]["tool"] == "ask_customer"
                )
                state = apply(store, state, asked)
                assert state["observations"][-1]["result"]["error"] == "duplicate_clarification_blocked"
                rows["train"].append(row(state, action, "answered_question_recovery"))
                audits.append(
                    {
                        "case_id": case["id"],
                        "origin": "answered_question_recovery",
                        **finish(store, case, state, action, questions),
                    }
                )
            if case["expected"]["kind"] == "handoff":
                state, action, _, questions = prefix(store, case)
                field = "reason" if case["family"] == "forbidden_status" else "payment_method"
                state = apply(
                    store,
                    state,
                    {
                        "tool": "ask_customer",
                        "arguments": {
                            "field": field,
                            "message": f"Please provide your {field.replace('_', ' ')}.",
                        },
                    },
                )
                state = reply(store, state, case, field)
                rows["train"].append(row(state, action, "policy_boundary_after_unnecessary_question"))
                audits.append(
                    {
                        "case_id": case["id"],
                        "origin": "handoff_recovery",
                        **finish(store, case, state, action, questions),
                    }
                )
            if case["expected"]["kind"] == "exchange" or case["family"] == "exchange_fallback":
                state, action, _, questions = prefix(store, case)
                args = {
                    "order_id": case["answers"]["order_id"],
                    "item_id": case["answers"]["item_id"],
                    "payment_method": case["answers"]["payment_method"],
                }
                if case["expected"]["kind"] == "exchange":
                    bad = {"tool": "propose_return", "arguments": args}
                    text = f"I requested an exchange to {case['answers']['new_option']}, not an item return. Use {case['answers']['payment_method']} for the difference."
                    origin = "preserve_exchange_intent"
                else:
                    alt = next(
                        v
                        for v in next(iter(case["products"].values()))["variants"]
                        if v["available"] and v["item_id"] != case["answers"]["item_id"]
                    )
                    bad = {"tool": "propose_exchange", "arguments": {**args, "new_item_id": alt["item_id"]}}
                    text = f"No substitute variant. I authorize a return only because {case['answers']['new_option']} is unavailable; refund original payment."
                    origin = "unavailable_fallback_recovery"
                state = apply(store, state, bad)
                assert state["status"] == "awaiting_confirmation"
                store.confirm(state["id"], state["proposal"]["id"], state["revision"], False)
                state = store.get(state["id"])
                state = store.customer_message(state["id"], text, state["revision"])
                rows["train"].append(row(state, action, origin))
                audits.append(
                    {"case_id": case["id"], "origin": origin, **finish(store, case, state, action, questions)}
                )
            if case["family"] in {"return", "missing_email", "missing_payment"}:
                state, action, _, questions = prefix(store, case)
                state = apply(
                    store,
                    state,
                    {
                        "tool": "get_product",
                        "arguments": {"product_id": case["orders"][0]["items"][0]["product_id"]},
                    },
                )
                rows["train"].append(row(state, action, "preserve_return_after_catalog"))
                audits.append(
                    {
                        "case_id": case["id"],
                        "origin": "return_after_catalog",
                        **finish(store, case, state, action, questions),
                    }
                )
    random.Random(202610274).shuffle(rows["train"])
    for split, examples in rows.items():
        (DATA / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in examples))
    sources = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "training/v4").glob("*.py")
        if p.name
        not in {
            "visualize.py",
            "verify_results.py",
            "serve.py",
            "write_summary.py",
            "integration.py",
            "check_package.py",
        }
    ]
    sources += [
        "training/v4/config.json",
        "training/base-model-spec.json",
        "resolve_desk/progress.py",
        "resolve_desk/workflow.py",
        "resolve_desk/store.py",
        "resolve_desk/protocol.py",
        "training/common.py",
        "training/runtime.py",
        "training/evaluate.py",
        "training/scoring.py",
        "training/v2/scoring.py",
        "training/v2/fixtures.py",
        "training/v3/fixtures.py",
        "training/v3/prepare.py",
        "training/v3/evaluation.py",
        "training/v2/evaluation.py",
    ]
    sources += [str(p.relative_to(ROOT)) for p in DATA.glob("*") if p.is_file()]
    manifest = {
        "cases": dict(Counter(c["split"] for c in cases)),
        "rows": {s: len(r) for s, r in rows.items()},
        "files": {p: sha(ROOT / p) for p in sources},
        "families": FAMILIES,
        "training_origins": dict(Counter(r["origin"] for r in rows["train"])),
        "training_actions": dict(
            Counter(json.loads(r["messages"][-1]["content"])["tool"] for r in rows["train"])
        ),
        "scope": "Fresh synthetic six-character entities and split-specific phrasing. 16 shared families; within-split counterfactual worlds intentionally shared. Full clean trajectories, one extra handoff/exchange terminal, and targeted recovery labels; erroneous actions never supervised. Old tests are development evidence; new tests never select weights.",
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (REPORTS / "data-audit.json").write_text(
        json.dumps(
            {"manifest": manifest, "all_reference_tasks_pass": True, "reference_outcomes": audits}, indent=2
        )
        + "\n"
    )
    print(json.dumps({k: manifest[k] for k in ["cases", "rows", "training_origins", "training_actions"]}))


if __name__ == "__main__":
    main()
