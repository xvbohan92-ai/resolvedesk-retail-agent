"""One actual Ollama regression probe, kept separate from held-out MLX accuracy."""

import json
import tempfile
from pathlib import Path

from resolve_desk.agent import advance
from resolve_desk.providers import Ollama
from resolve_desk.workflow import WorkflowStore
from training.common import ROOT, sha
from training.fixtures import load_cases
from training.v2.common import REPORTS, verify_frozen


def main():
    verify_frozen()
    destination = REPORTS / "live-ollama-regression.json"
    if destination.exists():
        raise ValueError("Live probe exists; preserve the first result")
    baseline = json.loads((ROOT / "reports/live-verification.json").read_text())["ollama_custom_request"]
    case = next(c for c in load_cases("valid") if c["id"] == "valid-cancel-00")
    case["initial_message"] = baseline["request"]
    provider = Ollama()
    identity = provider.identity()
    with tempfile.TemporaryDirectory() as folder:
        store = WorkflowStore(Path(folder) / "probe.sqlite")
        state = store.create(case)
        state = advance(store, state["id"], provider)
        proposal = state.get("proposal")
        correct = bool(
            state["status"] == "awaiting_confirmation"
            and proposal
            and proposal["kind"] == "cancel"
            and proposal["order_id"] == "ORD-20000"
            and proposal["arguments"]["reason"] == "ordered by mistake"
        )
        assert store.orders(state["id"]) == case["orders"], "No confirmation means no changed order"
        report = {
            "scope": "Single validation integration; separate from the held-out MLX comparison.",
            "request": baseline["request"],
            "identity": identity,
            "source_sha256": sha(__file__),
            "workflow_sha256": sha(ROOT / "resolve_desk/workflow.py"),
            "baseline_report_sha256": sha(ROOT / "reports/live-verification.json"),
            "baseline_calls": baseline["model_calls"],
            "model_calls": len(state["observations"]),
            "proposal_correct": correct,
            "status": state["status"],
            "proposal": proposal,
            "order_unchanged": True,
            "observations": state["observations"],
            "events": store.timeline(state["id"]),
        }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("baseline_calls", "model_calls", "proposal_correct", "status")}))


if __name__ == "__main__":
    main()
