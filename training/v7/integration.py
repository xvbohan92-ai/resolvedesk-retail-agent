"""All public validation worlds through the real app/adapter; separate from held-out scores."""

import argparse
import json
import shutil

import httpx

from training.common import sha
from training.v7.common import REPORTS, verify_frozen
from training.v7.fixtures import load_cases
from training.v7.scoring import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    verify_frozen()
    training = json.loads((REPORTS / "training.json").read_text())
    cases = load_cases("valid")
    plan = json.loads((REPORTS / "analysis-plan.json").read_text())
    assert set(plan["splits"]["valid"]["representatives"]) == {c["id"] for c in cases}
    destination = REPORTS / "live-api-verification.json"
    previous = {}
    if destination.exists():
        if not args.resume:
            raise ValueError("Live probe exists; use --resume only for untouched validation cases")
        original = json.loads(destination.read_text())
        assert original["source_sha256"] == sha(__file__)
        assert original["selected_adapter_sha256"] == training["selected_adapter_sha256"]
        if original.get("status") == "complete":
            raise ValueError("Live probes already completed")
        previous = {r["case_id"]: r for r in original["records"]}
        archive = REPORTS / f"live-api-before-resume-{len(previous)}.json"
        assert not archive.exists(), "Do not overwrite a prior interrupted probe"
        shutil.copyfile(destination, archive)
    output = {
        "scope": (
            "All public validation cases through the actual application/selected adapter. At most four "
            "field-keyed scripted customer replies, each API advance with twelve model calls. "
            "Confirmation follows the fixture's declared choice, not whether the proposal is correct. "
            "Semantic safety/invariant failures retain completed responses and continue to untouched cases. "
            "Infrastructure failures stop with all completed/partial evidence retained. No case is retried. "
            "These integration smokes are excluded from held-out accuracy."
        ),
        "source_sha256": sha(__file__),
        "selected_adapter_sha256": training["selected_adapter_sha256"],
        "planned_case_ids": [c["id"] for c in cases],
        "status": "partial",
        "records": [],
    }
    with httpx.Client(base_url="http://127.0.0.1:8771", trust_env=False, timeout=240) as client:

        def call(path, body=None):
            response = client.get(path) if body is None else client.post(path, json=body)
            response.raise_for_status()
            return response.json()

        identity = call("/api/health")
        model = identity["models"]["finetuned"]
        assert model["variant"] == "qlora_v7"
        assert model["adapter_sha256"] == training["selected_adapter_sha256"]
        output["identity"] = identity
        for case in cases:
            case_id = case["id"]
            if case_id in previous:
                # Finished failures are evidence, never requests to retry for a pass.
                output["records"].append(previous[case_id])
                continue
            record = {
                "case_id": case_id,
                "family": case["family"],
                "outcome_correct": False,
                "probe_phase": "start_session",
                "scripted_replies": [],
            }
            try:
                before = initial = call("/api/sessions", {"case_id": case_id, "provider": "finetuned"})
                record["initial_response"] = initial
                record["before_confirmation"] = before
                replies = record["scripted_replies"]
                for _ in range(4):
                    state = before["session"]
                    if state["status"] != "awaiting_customer":
                        break
                    field = state["observations"][-1]["result"]["field"]
                    reply = case["answers"][field]
                    record["probe_phase"] = "await_customer_reply"
                    reply_record = {"requested_field": field, "scripted_reply": reply}
                    replies.append(reply_record)
                    before = call(
                        f"/api/sessions/{state['id']}/messages",
                        {"revision": state["revision"], "message": reply, "provider": "finetuned"},
                    )
                    reply_record["response"] = before
                    record["before_confirmation"] = before
                state = before["session"]
                record["probe_phase"] = "validate_before_confirmation"
                assert before["orders"] == case["orders"], "Order mutated before confirmation"
                assert all(
                    o["timing"]["adapter_sha256"] == training["selected_adapter_sha256"]
                    for o in state["observations"]
                ), "Selected adapter identity changed mid-probe"
                proposal = state.get("proposal")
                if (
                    proposal
                    and proposal["kind"] == "return"
                    and case["intent_initial"]["operation"] == "exchange"
                ):
                    assert any(
                        r["requested_field"] == "fallback_return" and r["scripted_reply"] == "YES"
                        for r in replies
                    ), "Exchange-origin return lacks a real scoped YES reply"
                after = before
                replay = None
                if state["status"] == "awaiting_confirmation":
                    body = {
                        "revision": state["revision"],
                        "proposal_id": proposal["id"],
                        "accepted": case["confirmation"],
                    }
                    endpoint = f"/api/sessions/{state['id']}/confirm"
                    record["probe_phase"] = "confirm"
                    record["confirmation_request"] = body
                    after = call(endpoint, body)
                    record["after_confirmation"] = after
                    record["probe_phase"] = "confirm_replay"
                    replay_response = client.post(endpoint, json=body)
                    replay = {"status_code": replay_response.status_code, "body": replay_response.json()}
                    record["confirmation_replay"] = replay
                    if case["confirmation"]:
                        assert replay_response.status_code == 200
                        assert replay["body"]["confirmation_result"] == after["confirmation_result"]
                    else:
                        assert replay_response.status_code == 409
                        assert after["orders"] == before["orders"]
                    commits = [e for e in after["timeline"] if e["kind"] == "operation_committed"]
                    assert len(commits) == int(case["confirmation"])
                record["probe_phase"] = "grade"
                judged = score(
                    case,
                    {**after["session"], "proposal": proposal},
                    after["orders"],
                    before["orders"],
                    [r["requested_field"] for r in replies],
                )
                record.update(
                    {
                        "outcome_correct": judged["task_success"],
                        "probe_phase": "complete",
                        "business_score": judged,
                        "initial_response": initial,
                        "scripted_replies": replies,
                        "before_confirmation": before,
                        "after_confirmation": after,
                        "confirmation_replay": replay,
                    }
                )
            except AssertionError as error:
                record["integration_error"] = f"{type(error).__name__}: {error}"
                record["integration_error_type"] = "semantic_or_safety_invariant"
                output["records"].append(record)
                destination.write_text(json.dumps(output, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "validation_case": case_id,
                            "outcome_correct": False,
                            "semantic_failure_preserved": True,
                        }
                    ),
                    flush=True,
                )
                continue
            except (httpx.HTTPError, ValueError, KeyError) as error:
                record["integration_error"] = f"{type(error).__name__}: {error}"
                record["integration_error_type"] = "infrastructure_or_invalid_structure"
                if isinstance(error, httpx.HTTPStatusError):
                    record["infrastructure_status"] = error.response.status_code
                    try:
                        record["infrastructure_response"] = error.response.json()
                    except ValueError:
                        record["infrastructure_response"] = error.response.text[:4000]
                elif isinstance(error, json.JSONDecodeError):
                    record["infrastructure_response"] = error.doc[:4000]
                output["records"].append(record)
                destination.write_text(json.dumps(output, indent=2) + "\n")
                raise
            output["records"].append(record)
            destination.write_text(json.dumps(output, indent=2) + "\n")
            print(
                json.dumps({"validation_case": case_id, "outcome_correct": record["outcome_correct"]}),
                flush=True,
            )
    output["status"] = "complete"
    destination.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
