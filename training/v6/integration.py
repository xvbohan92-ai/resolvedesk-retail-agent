"""Actual loopback app + MLX validation smokes, excluded from held-out scores."""

import argparse
import json
import shutil

import httpx

from resolve_desk.progress import clarification_memory
from training.common import sha
from training.v6.common import REPORTS, verify_frozen
from training.v6.fixtures import FAMILIES, load_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    verify_frozen()
    training = json.loads((REPORTS / "training.json").read_text())
    cases = {c["family"]: c for c in load_cases("valid") if c["id"].endswith("-00")}
    representatives = set(
        json.loads((REPORTS / "analysis-plan.json").read_text())["splits"]["valid"]["representatives"]
    )
    cases = {f: c for f, c in cases.items() if c["id"] in representatives}
    destination = REPORTS / "live-api-verification.json"
    previous = {}
    if destination.exists():
        if not args.resume:
            raise ValueError("Live probe exists; use --resume for the remaining validation probes")
        original = json.loads(destination.read_text())
        if original.get("status") == "complete":
            raise ValueError("Live probe already completed")
        previous = {r["family"]: r for r in original["records"]}
        archive = REPORTS / "live-api-initial.json"
        if archive.exists():
            raise ValueError("Initial probe archive already exists")
        shutil.copyfile(destination, archive)
    output = {
        "scope": "Actual application/adapter validation smokes; at most two field-keyed scripted replies. "
        "Each API advance retains its twelve-step budget. Not held-out accuracy or human satisfaction.",
        "source_sha256": sha(__file__),
        "selected_adapter_sha256": training["selected_adapter_sha256"],
        "records": [],
    }
    with httpx.Client(base_url="http://127.0.0.1:8771", trust_env=False, timeout=240) as client:

        def call(path, body=None):
            r = client.get(path) if body is None else client.post(path, json=body)
            r.raise_for_status()
            return r.json()

        identity = call("/api/health")
        assert identity["models"]["finetuned"]["adapter_sha256"] == training["selected_adapter_sha256"]
        output["identity"] = identity
        for family in FAMILIES:
            if family not in cases:
                continue
            case = cases[family]
            if family in previous:
                # Preserve every finished probe, including failures. Resuming
                # must not re-run a committed transaction or cherry-pick a pass.
                output["records"].append(previous[family])
                continue
            before = initial = call("/api/sessions", {"case_id": case["id"], "provider": "finetuned"})
            replies = []
            for _ in range(2):
                state = before["session"]
                if state["status"] != "awaiting_customer":
                    break
                field = state["observations"][-1]["result"]["field"]
                text = case["answers"][field]
                before = call(
                    f"/api/sessions/{state['id']}/messages",
                    {"revision": state["revision"], "message": text, "provider": "finetuned"},
                )
                replies.append({"requested_field": field, "scripted_reply": text, "response": before})
            state = before["session"]
            assert before["orders"] == case["orders"]
            assert all(
                o["timing"]["adapter_sha256"] == training["selected_adapter_sha256"]
                for o in state["observations"]
            )
            proposal = state.get("proposal")
            if (
                proposal
                and proposal["kind"] == "return"
                and case["intent_initial"]["operation"] == "exchange"
            ):
                assert any(
                    r["requested_field"] == "fallback_return" and r["scripted_reply"] == "YES"
                    for r in replies
                )
            correct = bool(
                proposal
                and proposal["kind"] == case["expected"]["kind"]
                and proposal["amount_cents"] == case["expected"]["amount_cents"]
                and proposal["payment_method"] == case["answers"]["payment_method"]
                and (
                    proposal["kind"] != "exchange"
                    or proposal["new_item_id"] == case["expected"]["new_item_id"]
                )
            )
            if case["expected"]["kind"] == "handoff":
                correct = bool(
                    state["status"] == "handoff"
                    and state["user_id"] == case["customer"]["id"]
                    and any(o["action"] and o["action"]["tool"] == "handoff" for o in state["observations"])
                    and any(
                        o["action"]
                        and o["action"]["tool"] == "get_order"
                        and o["action"]["arguments"]["order_id"] == case["expected"]["order_id"]
                        for o in state["observations"]
                    )
                )
            required = case["required_clarification"]
            if required and not any(p["field"] == required for p in clarification_memory(state)):
                correct = False
            after, replay = None, None
            if state["status"] == "awaiting_confirmation":
                body = {"revision": state["revision"], "proposal_id": proposal["id"], "accepted": correct}
                endpoint = f"/api/sessions/{state['id']}/confirm"
                after = call(endpoint, body)
                replay = call(endpoint, body)
                assert replay["confirmation_result"] == after["confirmation_result"]
                events = [e for e in after["timeline"] if e["kind"] == "operation_committed"]
                assert len(events) == int(correct)
                if correct:
                    assert after["orders"][0]["status"] == case["expected"]["target_status"]
                else:
                    assert after["orders"] == case["orders"]
            output["records"].append(
                {
                    "family": family,
                    "outcome_correct": correct,
                    "initial_response": initial,
                    "scripted_replies": replies,
                    "before_confirmation": before,
                    "after_confirmation": after,
                    "confirmation_replay": replay,
                }
            )
            output["status"] = "partial"
            destination.write_text(json.dumps(output, indent=2) + "\n")
            print(json.dumps({"validation_family": family, "outcome_correct": correct}), flush=True)
    output["status"] = "complete"
    if previous:
        output["initial_probe_sha256"] = sha(REPORTS / "live-api-initial.json")
    destination.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
