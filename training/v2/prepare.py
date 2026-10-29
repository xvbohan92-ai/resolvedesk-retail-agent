import json
import random
import tempfile
from collections import Counter
from pathlib import Path

from resolve_desk.workflow import WorkflowStore, messages
from training.common import ROOT, sha
from training.v2.common import DATA, REPORTS
from training.v2.fixtures import FAMILIES, make_case, next_reference
from training.v2.scoring import score


def apply(store, state, action):
    return store.apply(
        state["id"], json.dumps(action), {"provider": "programmatic_demonstrator"}, state["revision"]
    )


def demonstrate(store, case, recovery=False):
    state, rows, questions = store.create(case), [], []
    injected = set()
    for _ in range(14):
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            questions.append(field)
            state = store.customer_message(state["id"], case["answers"][field], state["revision"])
            state["messages"][-1]["field"] = field
            with store.connect() as c:
                c.execute("UPDATE sessions SET body=? WHERE id=?", (json.dumps(state), state["id"]))
        if state["status"] != "running":
            break
        action = next_reference(case, state)
        if recovery and state["user_id"] and "repeat" not in injected:
            bad = {"tool": "identify_customer", "arguments": {"email": case["customer"]["email"]}}
            state = apply(store, state, bad)
            injected.add("repeat")
        if recovery and action["tool"] == "get_product" and "catalog" not in injected:
            bad = {
                "tool": "propose_exchange",
                "arguments": {
                    "order_id": case["answers"]["order_id"],
                    "item_id": case["answers"]["item_id"],
                    "new_item_id": "item_not_observed",
                    "payment_method": case["answers"]["payment_method"],
                },
            }
            state = apply(store, state, bad)
            injected.add("catalog")
        if recovery and action["tool"] == "propose_exchange" and "wrong_variant" not in injected:
            bad = {
                "tool": "propose_exchange",
                "arguments": {**action["arguments"], "new_item_id": case["answers"]["item_id"]},
            }
            state = apply(store, state, bad)
            injected.add("wrong_variant")
        rows.append(
            {
                "messages": messages(state)
                + [{"role": "assistant", "content": json.dumps(action, separators=(",", ":"))}]
            }
        )
        state = apply(store, state, action)
    before = store.orders(state["id"])
    proposal = state.get("proposal")
    if proposal:
        store.confirm(state["id"], proposal["id"], state["revision"], case["confirmation"])
        state = {**store.get(state["id"]), "proposal": proposal}
    graded = score(case, state, store.orders(state["id"]), before, questions)
    assert graded["task_success"], (case["id"], recovery, graded, state["observations"])
    return rows, {"case_id": case["id"], "split": case["split"], "recovery": recovery, **graded}


def main():
    if (DATA / "manifest.json").exists():
        raise ValueError("V2 data already frozen; use a new experiment for revisions")
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    cases = [
        make_case(s, f, n)
        for s, count in [("train", 8), ("valid", 2), ("test", 3)]
        for f in FAMILIES
        for n in range(count)
    ]
    sets = [{c["orders"][0]["id"] for c in cases if c["split"] == s} for s in ("train", "valid", "test")]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    (DATA / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    rows, audits = {s: [] for s in ("train", "valid", "test")}, []
    with tempfile.TemporaryDirectory() as tmp:
        store = WorkflowStore(Path(tmp) / "demonstrations.sqlite")
        for case in cases:
            clean, audit = demonstrate(store, case)
            rows[case["split"]].extend(clean)
            audits.append(audit)
            if case["split"] == "train":
                repair, audit = demonstrate(store, case, recovery=True)
                rows["train"].extend(repair)
                audits.append(audit)
    random.Random(20261027).shuffle(rows["train"])
    for split, records in rows.items():
        (DATA / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    sources = [
        "resolve_desk/workflow.py",
        "resolve_desk/store.py",
        "resolve_desk/protocol.py",
        "training/v2/fixtures.py",
        "training/v2/prepare.py",
        "training/v2/scoring.py",
        "training/v2/config.json",
        "training/v2/evaluate.py",
        "training/v2/evaluation.py",
    ]
    sources.extend(
        [
            "training/v2/train.py",
            "training/v2/audit.py",
            "training/v2/common.py",
            "training/common.py",
            "training/runtime.py",
            "training/scoring.py",
            "training/evaluate.py",
        ]
    )
    sources.extend(str(p.relative_to(ROOT)) for p in DATA.glob("*") if p.is_file())
    manifest = {
        "cases": dict(Counter(c["split"] for c in cases)),
        "rows": {s: len(r) for s, r in rows.items()},
        "files": {p: sha(ROOT / p) for p in sources},
        "families": FAMILIES,
        "train_action_counts": dict(
            Counter(json.loads(r["messages"][-1]["content"])["tool"] for r in rows["train"])
        ),
        "scope": "New self-authored synthetic task suite; v1 test errors informed design. "
        "Disjoint IDs, price draws, opening templates and requested colors across splits; "
        "shared author, policy, schema, task families and per-family request cores. "
        "Training includes clean and injected-error recovery demonstrations; no held-out "
        "model outputs before training/checkpoint selection. Not official tau-bench.",
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (REPORTS / "data-audit.json").write_text(
        json.dumps(
            {"manifest": manifest, "all_reference_tasks_pass": True, "reference_outcomes": audits}, indent=2
        )
        + "\n"
    )
    print(json.dumps({k: manifest[k] for k in ("cases", "rows", "train_action_counts")}))


if __name__ == "__main__":
    main()
