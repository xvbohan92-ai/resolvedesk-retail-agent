import json
import tempfile
from collections import Counter
from pathlib import Path

from resolve_desk.protocol import messages
from resolve_desk.store import Store
from training.common import ROOT, sha
from training.fixtures import FAMILIES, make_case, next_reference
from training.scoring import score


def main():
    destination = ROOT / "training/datasets"
    if (destination / "manifest.json").exists():
        raise ValueError("Frozen data exists; create a new experiment instead of overwriting")
    cases = [
        make_case(split, family, n)
        for split, count in [("train", 8), ("valid", 2), ("test", 3)]
        for family in FAMILIES
        for n in range(count)
    ]
    (destination / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    split_rows, audit = {s: [] for s in ("train", "valid", "test")}, []
    with tempfile.TemporaryDirectory() as folder:
        store = Store(Path(folder) / "demonstrations.sqlite")
        for case in cases:
            state = store.create(case)
            for _ in range(10):
                if state["status"] == "awaiting_customer":
                    field = state["observations"][-1]["result"]["field"]
                    state = store.customer_message(state["id"], case["answers"][field], state["revision"])
                    # Keep field tags only in demonstrator metadata; model sees plain customer utterances.
                    state["messages"][-1]["field"] = field
                    # Persist this metadata for the reference, but strip it from model-visible messages below.
                    with store.connect() as c:
                        c.execute("UPDATE sessions SET body=? WHERE id=?", (json.dumps(state), state["id"]))
                if state["status"] != "running":
                    break
                action = next_reference(case, state)
                prompt = messages(state)
                split_rows[case["split"]].append(
                    {
                        "messages": prompt
                        + [{"role": "assistant", "content": json.dumps(action, separators=(",", ":"))}]
                    }
                )
                state = store.apply(
                    state["id"],
                    json.dumps(action),
                    {"provider": "programmatic_demonstrator"},
                    state["revision"],
                )
                if any("error" in o["result"] for o in state["observations"]):
                    # The foreign-order rejection is expected; every other demonstration must be executable.
                    assert case["family"] == "foreign_order", case["id"]
            before = store.orders(state["id"])
            proposal = state.get("proposal")
            if proposal:
                store.confirm(state["id"], proposal["id"], state["revision"], case["confirmation"])
                # Declines clear displayed proposals; preserve its contents for outcome grading.
                state = {**store.get(state["id"]), "proposal": proposal}
            judged = score(case, state, store.orders(state["id"]), before)
            assert judged["task_success"], (case["id"], judged)
            audit.append({"id": case["id"], "split": case["split"], **judged})
    for split, rows in split_rows.items():
        (destination / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    paths = [
        "training/fixtures.py",
        "training/prepare.py",
        "training/scoring.py",
        "training/config.json",
        "resolve_desk/protocol.py",
        "training/datasets/cases.json",
    ]
    paths += [f"training/datasets/{s}.jsonl" for s in split_rows]
    manifest = {
        "cases": dict(Counter(c["split"] for c in cases)),
        "rows": {s: len(rows) for s, rows in split_rows.items()},
        "files": {p: sha(ROOT / p) for p in paths},
        "families": FAMILIES,
        "scope": "Self-authored synthetic English cases; disjoint IDs, numbers and request templates "
        "across splits; shared policy, schema, task families and author. Programmatic demonstrations; "
        "not human annotations, not official tau-bench scores, not real retailer outcomes.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "reports/data-audit.json").write_text(
        json.dumps(
            {"manifest": manifest, "reference_outcomes": audit, "all_reference_tasks_pass": True}, indent=2
        )
        + "\n"
    )
    print(json.dumps({"cases": manifest["cases"], "rows": manifest["rows"]}))


if __name__ == "__main__":
    main()
