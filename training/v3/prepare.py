import json
import random
import tempfile
from collections import Counter
from pathlib import Path

from resolve_desk.workflow import WorkflowStore, messages
from training.common import ROOT, sha
from training.v2.scoring import score
from training.v3.common import DATA, REPORTS
from training.v3.fixtures import FAMILIES, make_case, next_reference


def apply(store, state, action):
    return store.apply(state["id"], json.dumps(action), {"provider": "reference_only"}, state["revision"])


def training_row(state, action, origin):
    return {
        "messages": messages(state)
        + [{"role": "assistant", "content": json.dumps(action, separators=(",", ":"))}],
        "origin": origin,
    }


def prefix(store, case):
    state, rows, questions = store.create(case), [], []
    for _ in range(10):
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            questions.append(field)
            state = store.customer_message(state["id"], case["answers"][field], state["revision"])
            state["messages"][-1]["field"] = field
            with store.connect() as c:
                c.execute("UPDATE sessions SET body=? WHERE id=?", (json.dumps(state), state["id"]))
        action = next_reference(case, state)
        if action["tool"].startswith("propose_") or action["tool"] == "handoff":
            return state, action, rows, questions
        rows.append(training_row(state, action, "upstream"))
        state = apply(store, state, action)
    raise ValueError("Reference prefix did not terminate")


def finish(store, case, state, action, questions):
    state = apply(store, state, action)
    before, proposal = store.orders(state["id"]), state.get("proposal")
    if proposal:
        store.confirm(state["id"], proposal["id"], state["revision"], case["confirmation"])
        state = {**store.get(state["id"]), "proposal": proposal}
    result = score(case, state, store.orders(state["id"]), before, questions)
    assert result["task_success"], (case["id"], result)
    return result


def main():
    if (DATA / "manifest.json").exists():
        raise ValueError("V3 data already frozen")
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    cases = [
        make_case(s, f, n)
        for s, count in [("train", 8), ("valid", 1), ("test", 3)]
        for f in FAMILIES
        for n in range(count)
    ]
    (DATA / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    rows, audits = {s: [] for s in ("train", "valid", "test")}, []
    with tempfile.TemporaryDirectory() as tmp:
        store = WorkflowStore(Path(tmp) / "reference.sqlite")
        for case in cases:
            state, action, upstream, questions = prefix(store, case)
            terminal = training_row(state, action, "clean_terminal")
            if case["split"] == "train":
                copies = 3 if case["expected"]["kind"] == "exchange" else 1
                rows["train"].extend([terminal] * copies)
                rows["train"].extend(
                    r for r in upstream if json.loads(r["messages"][-1]["content"])["tool"] == "ask_customer"
                )
                reads = [
                    r for r in upstream if json.loads(r["messages"][-1]["content"])["tool"] != "ask_customer"
                ]
                if reads:
                    rows["train"].append(reads[int(case["id"].rsplit("-", 1)[1]) % len(reads)])
            else:
                # Reference snapshots for token audits; model selection uses full validation tasks.
                rows[case["split"]].extend([*upstream, terminal])
            audits.append(
                {"case_id": case["id"], "kind": "clean", **finish(store, case, state, action, questions)}
            )
            if case["split"] != "train":
                continue
            if case["expected"]["kind"] == "exchange":
                state, action, _, questions = prefix(store, case)
                bad = {
                    "tool": "propose_return",
                    "arguments": {
                        "order_id": case["answers"]["order_id"],
                        "item_id": case["answers"]["item_id"],
                        "payment_method": case["answers"]["payment_method"],
                    },
                }
                state = apply(store, state, bad)
                assert state["status"] == "awaiting_confirmation"
                store.confirm(state["id"], state["proposal"]["id"], state["revision"], False)
                state = store.get(state["id"])
                text = (
                    f"I asked for an exchange to {case['answers']['new_option']}, not an item return. "
                    f"Use {case['answers']['payment_method']} for the difference."
                )
                state = store.customer_message(state["id"], text, state["revision"])
                rows["train"].extend([training_row(state, action, "wrong_return_correction")] * 2)
                audits.append(
                    {
                        "case_id": case["id"],
                        "kind": "correction",
                        **finish(store, case, state, action, questions),
                    }
                )
            elif case["expected"]["kind"] == "return" and case["family"] in {
                "return",
                "missing_payment",
                "missing_email",
            }:
                state, action, _, questions = prefix(store, case)
                state = apply(
                    store,
                    state,
                    {
                        "tool": "get_product",
                        "arguments": {"product_id": case["orders"][0]["items"][0]["product_id"]},
                    },
                )
                rows["train"].append(training_row(state, action, "return_after_catalog"))
                audits.append(
                    {
                        "case_id": case["id"],
                        "kind": "return_context",
                        **finish(store, case, state, action, questions),
                    }
                )
    random.Random(202610273).shuffle(rows["train"])
    for split, records in rows.items():
        (DATA / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    sources = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "training/v3").glob("*.py")
        if p.name not in {"visualize.py", "serve.py", "write_summary.py", "verify_results.py"}
    ]
    sources += [
        "training/v3/config.json",
        "resolve_desk/workflow.py",
        "resolve_desk/store.py",
        "resolve_desk/protocol.py",
        "training/common.py",
        "training/runtime.py",
        "training/v2/fixtures.py",
        "training/v2/scoring.py",
        "training/v2/evaluation.py",
        "training/scoring.py",
        "training/evaluate.py",
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
        "scope": "Fresh self-authored synthetic data; v1/v2 errors are development evidence. "
        "Disjoint six-character IDs, price draws and split-specific utterances; shared author, "
        "policy, schema and task families. Positive exchange terminal snapshots are oversampled "
        "threefold; explicit customer correction snapshots twofold. No test model output before selection.",
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (REPORTS / "data-audit.json").write_text(
        json.dumps(
            {"manifest": manifest, "all_reference_tasks_pass": True, "reference_outcomes": audits}, indent=2
        )
        + "\n"
    )
    print(json.dumps({k: manifest[k] for k in ("cases", "rows", "training_origins", "training_actions")}))


if __name__ == "__main__":
    main()
