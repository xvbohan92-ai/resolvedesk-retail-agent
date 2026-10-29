"""Independent raw-action replay, first-output diagnostics and export reconciliation."""

import copy
import csv
import json
import math
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from resolve_desk.evidence_v7 import validated_report
from resolve_desk.resolution_v6_1 import ResolutionStore as PreviousStore
from resolve_desk.resolution_v7 import ResolutionStore
from training.common import ROOT, sha
from training.v6_1.verify_results import verify as verify_previous
from training.v7 import protocol_audit as protocol_checks
from training.v7.check_initialization import AUDIT as INITIALIZATION_AUDIT
from training.v7.check_initialization import verify_audit
from training.v7.common import DATA, REPORTS, verify_frozen
from training.v7.evaluation import ARMS, promotion, selection_key, summarize
from training.v7.fixtures import load_cases
from training.v7.scoring import score

ENUMS = {
    "operation": {"cancel", "return", "exchange", "handoff", "unknown"},
    "payment_method": {"original", "gift_card", "unknown"},
    "reason": {"no longer needed", "ordered by mistake", "unknown"},
    "fallback_return": {"yes", "no", "unknown"},
}
FIELDS = {"operation", "email", "order_id", "requested_option", "payment_method", "reason", "fallback_return"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def raw_record(raw):
    """Decode raw fields without accepting their enum values or skipping a failed first output."""
    try:
        action = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(action, dict) or action.get("tool") != "record_request":
        return None
    args = action.get("arguments")
    if not isinstance(args, dict) or set(args) != FIELDS | {"evidence"}:
        return None
    if not all(isinstance(v, str) for v in args.values()):
        return None
    return {k: v for k, v in args.items() if k != "evidence"}


def decoded_action(raw):
    try:
        action = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return action if isinstance(action, dict) else None


def raw_enum_errors(raw):
    action = decoded_action(raw)
    if not action or action.get("tool") != "record_request" or not isinstance(action.get("arguments"), dict):
        return {}
    fields = action["arguments"]
    return {
        k: fields[k]
        for k, allowed in ENUMS.items()
        if k in fields and (not isinstance(fields[k], str) or fields[k] not in allowed)
    }


def entity_keys(case):
    keys = {case["customer"]["id"], case["customer"]["email"]}
    if "gift_card" in case["customer"]:
        keys.add(case["customer"]["gift_card"]["id"])
    for product_id, product in case["products"].items():
        keys.add(product_id)
        keys.update(v["item_id"] for v in product["variants"])
    for order in case["orders"]:
        keys.update(order[k] for k in ["id", "user_id", "original_payment"])
        keys.update(item[k] for item in order["items"] for k in ["item_id", "product_id"])
    return keys


def population_plan(manifest):
    plan_path = REPORTS / "analysis-plan.json"
    assert str(plan_path.relative_to(ROOT)) in manifest["files"]
    plan = json.loads(plan_path.read_text())
    assert not plan["test_inference_started"]
    assert plan["cases_sha256"] == sha(DATA / "cases.json")
    split_ids, split_initial, split_entities, details = {}, {}, {}, {}
    for split in ["train", "valid", "test"]:
        cases = load_cases(split)
        groups = defaultdict(list)
        for case in cases:
            groups[canonical({k: case[k] for k in plan["group_fields"]})].append(case["id"])
        actual = {
            "rows": len(cases),
            "unique": len(groups),
            "representatives": [ids[0] for ids in groups.values()],
            "duplicate_groups": [ids for ids in groups.values() if len(ids) > 1],
        }
        for key, value in actual.items():
            assert plan["splits"][split][key] == value, (split, key)
        assert actual["rows"] == actual["unique"]
        split_ids[split] = {order["id"] for c in cases for order in c["orders"]}
        split_initial[split] = {c["initial_message"] for c in cases}
        split_entities[split] = set.union(*(entity_keys(c) for c in cases))
        assert len(split_ids[split]) == len(split_initial[split]) == len(cases)
        assert plan["initial_model_inputs"][split] == {
            "rows": len(cases),
            "unique": len(split_initial[split]),
            "duplicate_groups": [],
        }
        details[split] = {
            "rows": len(cases),
            "unique_interactions": len(groups),
            "unique_worlds": len(split_ids[split]),
            "unique_initial_requests": len(split_initial[split]),
            "unique_entities": len(split_entities[split]),
        }
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        assert not split_ids[a] & split_ids[b] and not split_initial[a] & split_initial[b]
        assert not split_entities[a] & split_entities[b]
    all_current = set.union(*split_ids.values())
    all_entities = set.union(*split_entities.values())
    all_initial = set.union(*split_initial.values())
    # Keep historical verification stable when later development data is added.
    for prefix in ["training", *[f"training/v{n}" for n in [2, 3, 4, 5, 6]], "training/v6_1"]:
        source = ROOT / prefix / "datasets/cases.json"
        older = json.loads(source.read_text())
        assert not {o["id"] for c in older for o in c["orders"]} & all_current, source
        assert not set.union(*(entity_keys(c) for c in older)) & all_entities, source
        assert not {c["initial_message"] for c in older} & all_initial, source
    train_rows = [json.loads(line) for line in (DATA / "train.jsonl").read_text().splitlines()]
    unique_pairs = {canonical(r["messages"]) for r in train_rows}
    assert len(unique_pairs) == len(train_rows), "Physical SFT pairs must be deduplicated"
    assert plan["training_snapshots"] == {
        "rows": len(train_rows),
        "unique_prompt_completion_pairs": len(unique_pairs),
    }
    return (
        plan,
        train_rows,
        {
            "splits": details,
            "physical_training_pairs": len(train_rows),
            "unique_training_pairs": len(unique_pairs),
        },
    )


def replay_actions(case, record):
    """Replay exact actions, actual replies and confirmation, without an LLM or expected labels."""
    with tempfile.TemporaryDirectory() as tmp:
        cls = PreviousStore if record["variant"] == "previous_old_flow" else ResolutionStore
        store = cls(Path(tmp) / "replay.sqlite")
        state = store.create(case)
        proposal_ids = {}
        for event in record["events"]:
            if event["kind"] == "customer_message":
                state = store.customer_message(state["id"], event["detail"]["content"], state["revision"])
            elif event["kind"] == "model_tool":
                actual = event["detail"]
                state = store.apply(state["id"], actual["raw"], {}, state["revision"])
                got = state["observations"][-1]
                assert got["action"] == actual["action"]
                want, result = copy.deepcopy(actual["result"]), copy.deepcopy(got["result"])
                if "error" in want:
                    # Pydantic diagnostic prose is not a semantic model/policy field.
                    want.pop("detail", None)
                    result.pop("detail", None)
                if "proposal" in want:
                    original_id = want["proposal"]["id"]
                    replay_id = result["proposal"]["id"]
                    assert isinstance(original_id, str) and isinstance(replay_id, str)
                    assert original_id not in proposal_ids or proposal_ids[original_id] == replay_id
                    assert (
                        replay_id not in proposal_ids.values() or proposal_ids.get(original_id) == replay_id
                    )
                    proposal_ids[original_id] = replay_id
                    want["proposal"]["id"] = replay_id
                    # The fingerprint hashes deterministic business fields and
                    # must match. Only the opaque proposal ID is randomized.
                assert result == want, (case["id"], actual["action"])
            elif event["kind"] in {"operation_committed", "confirmation_declined"}:
                assert state["status"] == "awaiting_confirmation"
                payload = copy.deepcopy(event["detail"])
                original_id = payload["proposal_id"]
                assert isinstance(original_id, str) and original_id in proposal_ids
                replay_id = proposal_ids[original_id]
                assert replay_id == state["proposal"]["id"], "Confirmation refers to a different proposal"
                payload["proposal_id"] = replay_id
                receipt = store.confirm(
                    state["id"],
                    replay_id,
                    state["revision"],
                    event["kind"] == "operation_committed",
                )
                replay_event = store.timeline(state["id"])[-1]
                assert replay_event["kind"] == event["kind"]
                assert replay_event["detail"] == payload, "Confirmation event payload differs"
                if event["kind"] == "operation_committed":
                    assert receipt == payload, "Committed confirmation receipt differs"
                else:
                    # Decline events contain only a proposal ID; the HTTP/store
                    # return contract is a separate, smaller result.
                    assert receipt == {"status": "declined", "order_changed": False}
                state = store.get(state["id"])
        assert state["status"] == record["status_final"]
        assert store.orders(state["id"]) == record["order_after_confirmation"]


def audit_record(record, case, expected_provider="mlx"):
    trace = record["trace"]
    assert len(trace) == record["model_calls"] <= 12
    assert trace and all(o["timing"]["provider"] == expected_provider for o in trace)
    assert [e["detail"] for e in record["events"] if e["kind"] == "model_tool"] == trace
    initial = raw_record(trace[0]["raw"])
    errors = raw_enum_errors(trace[0]["raw"])
    assert record["intent_initial"] == initial
    assert record["initial_raw_parse_failure"] == (initial is None)
    assert record["initial_parse_missing"] == (initial is None)
    assert record["initial_record_valid"] == ("resolution_request" in trace[0]["result"])
    assert record["intent_enum_valid"] == (initial is not None and not errors)
    assert record["initial_enum_errors"] == errors
    assert record["schema_valid"] == sum(
        o["action"] is not None and not raw_enum_errors(o["raw"]) for o in trace
    )
    questions, pending, turns = [], None, [case["initial_message"]]
    invalid_by_turn, reason_by_turn = set(), set()
    recovered, reason_recovered, enum_attempts, reason_attempts, record_rejections = False, False, 0, 0, 0
    for event in record["events"]:
        if event["kind"] == "customer_message":
            assert pending is not None
            assert event["detail"]["content"] == case["answers"][pending]
            turns.append(event["detail"]["content"])
            questions.append(pending)
            pending = None
        elif event["kind"] == "model_tool":
            observation = event["detail"]
            action, result = observation["action"], observation["result"]
            fields = raw_record(observation["raw"])
            field_errors = raw_enum_errors(observation["raw"])
            if field_errors:
                enum_attempts += 1
                invalid_by_turn.add(len(turns))
            if "reason" in field_errors:
                reason_attempts += 1
                reason_by_turn.add(len(turns))
            decoded = decoded_action(observation["raw"])
            if decoded and decoded.get("tool") == "record_request" and "error" in result:
                record_rejections += 1
            if action is not None:
                assert json.loads(observation["raw"]) == action
            if "question" in result and "error" not in result:
                pending = result["field"]
            if "resolution_request" in result:
                assert not field_errors
                assert result["source_message_count"] == len(turns)
                evidence = result["evidence"]
                assert evidence in turns[-1] and len(evidence) >= min(8, len(turns[-1]))
                assert evidence == action["arguments"]["evidence"]
                effective = dict(fields)
                for change in result.get("slot_normalizations", []):
                    assert effective[change["field"]] == change["raw"]
                    effective[change["field"]] = change["effective"]
                assert result["resolution_request"] == effective
                recovered |= len(turns) in invalid_by_turn
                reason_recovered |= len(turns) in reason_by_turn
    assert record["questions"] == questions
    assert record["enum_error_attempts"] == enum_attempts
    assert record["enum_recovered"] == recovered
    assert record["reason_enum_errors"] == reason_attempts
    assert record["reason_enum_recovered"] == reason_recovered
    assert record["record_rejections"] == record_rejections
    for key, expected in case["intent_initial"].items():
        assert record["intent_field_correct"][key] == bool(initial and initial[key] == expected)
    assert record["permission_authorized"] == (case["intent_initial"]["fallback_return"] == "yes")
    assert record["permission_yes"] == bool(initial and initial["fallback_return"] == "yes")
    assert record["missing_email_case"] == ("email" in case["required_answers"])
    proposals = [o["result"]["proposal"] for o in trace if "proposal" in o["result"]]
    proposal = proposals[-1] if proposals else None
    assert record["predicted_outcome"] == (
        proposal["kind"] if proposal else "handoff" if record["status_final"] == "handoff" else "unfinished"
    )
    judged = score(
        case,
        {"status": record["status_final"], "proposal": proposal, "observations": trace},
        record["order_after_confirmation"],
        record["order_before_confirmation"],
        questions,
    )
    assert all(record[key] == value for key, value in judged.items())
    assert record["order_before_confirmation"] == case["orders"]
    ask_fields = [
        o["action"]["arguments"]["field"]
        for o in trace
        if o["action"] and o["action"]["tool"] == "ask_customer"
    ]
    assert record["repeated_question_attempts"] == sum(n - 1 for n in Counter(ask_fields).values())
    unnecessary = case["expected"]["kind"] != "handoff"
    assert record["unnecessary_handoff_attempts"] == sum(
        bool(o["action"] and o["action"]["tool"] == "handoff" and unnecessary) for o in trace
    )
    assert record["unnecessary_handoff_executed"] == int(
        unnecessary and any(o["result"].get("handoff") for o in trace)
    )
    replay_actions(case, record)


def reconcile_protocol_rows(report, protocol, cases):
    """Recompute every consent-evidence row; persisted summaries alone are insufficient."""
    expected_rows = Counter({(variant, cid): 1 for variant in ARMS for cid in cases})
    assert Counter((r["variant"], r["case_id"]) for r in report["records"]) == expected_rows
    assert Counter((r["variant"], r["case_id"]) for r in protocol["records"]) == expected_rows
    actual = [protocol_checks.audit_record(record, cases[record["case_id"]]) for record in report["records"]]
    assert protocol["records"] == actual
    assert protocol["summary"] == protocol_checks.summarize(actual)
    assert protocol["records_audited"] == len(actual) == len(report["records"])


def verify_protocol_evidence(report):
    """Recompute the separately planned consent audit without rewriting any business metric."""
    source = ROOT / "training/v7/protocol_audit.py"
    expected_source = "ba9d97df3c6e820b859e6e5997c64df51b21b3822f9b3a8f1079454cab706b03"
    expected_plan = "0e69b9ade6927e2a2efe1877b72d00bbdf6a5235eb82f1ca678e486969d52a93"
    plan = json.loads(protocol_checks.PLAN.read_text())
    protocol = json.loads(protocol_checks.OUTPUT.read_text())
    assert sha(source) == plan["source_sha256"] == protocol["source_sha256"] == expected_source
    assert not plan["test_inference_started"]
    assert plan["manifest_sha256"] == protocol["manifest_sha256"] == sha(DATA / "manifest.json")
    assert plan["cases_sha256"] == sha(DATA / "cases.json")
    assert plan["variants"] == ARMS
    assert protocol["evaluation_sha256"] == sha(REPORTS / "evaluation.json")
    assert protocol["plan_sha256"] == sha(protocol_checks.PLAN) == expected_plan
    assert protocol["plan_created_at_utc"] == plan["created_at_utc"]
    assert protocol["plan_precedes_heldout_events"]
    for artifact in [plan, protocol]:
        assert artifact["business_scores_or_release_gate_modified"] is False
        assert artifact["selection_rule_modified"] is False
    cases = {c["id"]: c for c in load_cases("test")}
    assert len(cases) == plan["expected_tasks_per_arm"]
    assert datetime.fromisoformat(plan["created_at_utc"]).timestamp() < min(
        event["created"] for record in report["records"] for event in record["events"]
    )
    reconcile_protocol_rows(report, protocol, cases)
    assert protocol["scope"] == plan["scope"] and protocol["limitations"] == plan["limitations"]
    return protocol


def verify():
    verify_previous()
    manifest = verify_frozen()
    plan, train_rows, quality = population_plan(manifest)
    report = json.loads((REPORTS / "evaluation.json").read_text())
    training = json.loads((REPORTS / "training.json").read_text())
    updates = json.loads((REPORTS / "updates.json").read_text())
    preflight = json.loads((REPORTS / "preflight.json").read_text())
    values = json.loads((ROOT / "docs/figures/v7/values.json").read_text())
    cases = {c["id"]: c for c in load_cases("test")}
    assert Counter((r["variant"], r["case_id"]) for r in report["records"]) == Counter(
        {(v, cid): 1 for v in ARMS for cid in cases}
    )
    test_start = min(e["created"] for r in report["records"] for e in r["events"])
    assert datetime.fromisoformat(plan["created_at_utc"]).timestamp() < test_start
    assert training["dataset"] == updates["dataset"] == manifest
    assert training["config"] == updates["config"] == report["config"]
    iterations = training["config"]["iters"]
    assert training["completed_updates"] == updates["completed_updates"] == iterations
    assert len(updates["metrics"]) == iterations
    assert [r["iteration"] for r in updates["metrics"]] == list(range(1, iterations + 1))
    assert all(math.isfinite(r["loss"]) and r["loss"] >= 0 for r in updates["metrics"])
    assert updates["adapter_l2_change"] > 0 and not updates["preflight_only"]
    initialization = verify_audit(updates["trainable_parameters"])
    assert initialization["stages"]["formal"]["planned_formal_updates"] == updates["completed_updates"]
    assert values["initialization_audit_sha256"] == sha(INITIALIZATION_AUDIT)
    assert not updates["preflight_weights_reused"] and preflight["preflight_only"]
    assert preflight["metrics"][0]["post_update_same_sample_loss"] < preflight["metrics"][0]["loss"]
    assert preflight["token_audit"]["loss_error_vs_full_output_projection"] < 0.001
    assert updates["token_audit"]["all_targets_covered"]
    for key, value in [
        ("ordinary_completion_weight", 1),
        ("permission_value_weight", 8),
        ("reason_value_weight", 4),
    ]:
        assert updates["token_audit"][key] == value
    assert (
        updates["token_audit"]["permission_value_tokens"] > 0
        and updates["token_audit"]["reason_value_tokens"] > 0
    )
    assert training["initialization_sha256"] == manifest["initial_adapter_sha256"]
    assert training["updates_sha256"] == sha(REPORTS / "updates.json")
    assert report["training_sha256"] == sha(REPORTS / "training.json")
    assert report["manifest_sha256"] == sha(DATA / "manifest.json")
    assert report["evaluator_sha256"] == sha(ROOT / "training/v7/evaluate.py")
    assert not training["test_used_for_selection"]
    sampling = updates["sampling"]
    indices = [m["sample_index"] for m in updates["metrics"]]
    for metric in updates["metrics"]:
        row = train_rows[metric["sample_index"]]
        assert metric["case_id"] == row["case_id"] and metric["tool"] == row["origin"]
        assert metric["sampling_bucket"] == (
            "intent_and_record_repair" if row["origin"] == "record_request" else "execution_and_reply"
        )
    selected_steps = training["best_checkpoint"]["iteration"]
    selected_indices = indices[:selected_steps]
    selected_coverage = {
        "completed_updates": selected_steps,
        "physical_training_rows": len(train_rows),
        "unique_rows_sampled": len(set(selected_indices)),
        "unique_cases_sampled": len({train_rows[i]["case_id"] for i in selected_indices}),
        "all_physical_rows_used": len(set(selected_indices)) == len(train_rows),
    }
    selected_sampling = training["selected_sampling"]
    assert selected_sampling == {
        "completed_updates": selected_steps,
        "unique_rows_sampled": len(set(selected_indices)),
        "physical_rows": len(train_rows),
        "bucket_updates": dict(Counter(m["sampling_bucket"] for m in updates["metrics"][:selected_steps])),
    }
    assert sampling["physical_rows"] == len(train_rows)
    assert sampling["unique_rows_sampled"] == len(set(indices))
    assert sampling["unique_cases_sampled"] == len({train_rows[i]["case_id"] for i in indices})
    assert sampling["all_physical_rows_used"] == (len(set(indices)) == len(train_rows))
    assert sampling["bucket_updates"] == dict(Counter(m["sampling_bucket"] for m in updates["metrics"]))
    assert sampling["bucket_rows"] == dict(
        Counter(
            "intent_and_record_repair" if row["origin"] == "record_request" else "execution_and_reply"
            for row in train_rows
        )
    )
    audits = json.loads((REPORTS / "reference-audit.json").read_text())
    assert all(r["task_success"] for r in audits)
    assert sum(r["trajectory"] == "clean_reference" for r in audits) == sum(
        len(load_cases(s)) for s in ["train", "valid", "test"]
    )
    candidates = [
        json.loads((REPORTS / f"validation-{n}.json").read_text())
        for n in training["config"]["checkpoint_candidates_completed_updates"]
    ]
    best = max(candidates, key=lambda c: selection_key(c["summary"]))
    assert best["iteration"] == training["best_checkpoint"]["iteration"]
    assert best["adapter_sha256"] == training["selected_adapter_sha256"]
    valid_cases = {c["id"]: c for c in load_cases("valid")}
    for candidate in candidates:
        assert {r["case_id"] for r in candidate["records"]} == set(valid_cases)
        assert len(candidate["records"]) == len(valid_cases)
        assert candidate["summary"] == summarize(candidate["records"])
        assert max(e["created"] for r in candidate["records"] for e in r["events"]) < test_start
    for record in report["records"]:
        audit_record(record, cases[record["case_id"]])
    with (ROOT / "docs/figures/v7/per-case.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(report["records"])
    exported = {(r["variant"], r["case_id"]): r for r in rows}
    for record in report["records"]:
        row = exported[record["variant"], record["case_id"]]
        for key in [
            "task_success",
            "unauthorized_return",
            "permission_authorized",
            "permission_yes",
            "missing_email_case",
            "initial_raw_parse_failure",
            "enum_recovered",
            "reason_enum_recovered",
            "initial_record_valid",
            "intent_enum_valid",
        ]:
            assert row[key] == str(record[key])
        for key in [
            "model_calls",
            "repeated_question_attempts",
            "repeated_questions_delivered",
            "unnecessary_handoff_attempts",
            "unnecessary_handoff_executed",
            "quote_rejections",
            "policy_gate_rejections",
            "slot_normalizations",
            "enum_error_attempts",
            "reason_enum_errors",
            "record_rejections",
        ]:
            assert int(row[key]) == record[key]
        for key in ["intent_initial", "initial_enum_errors"]:
            assert json.loads(row[key]) == record[key]
    assert values["summary"] == values["primary_summary"] == report["summary"]
    assert values["analysis_plan_sha256"] == sha(REPORTS / "analysis-plan.json")
    for variant in ARMS:
        records = [r for r in report["records"] if r["variant"] == variant]
        assert report["summary"][variant] == summarize(records)
        for family, counts in values["family_success"].items():
            assert counts[variant] == sum(r["task_success"] for r in records if r["family"] == family)
        diagnostic = values["permission_diagnostics"][variant]
        assert diagnostic["true_affirmative"] == sum(
            r["permission_authorized"] and r["permission_yes"] for r in records
        )
        assert diagnostic["unsupported_affirmative"] == sum(
            not r["permission_authorized"] and r["permission_yes"] for r in records
        )
        assert diagnostic["authorized_tasks"] == sum(r["permission_authorized"] for r in records)
        assert diagnostic["non_authorized_tasks"] == sum(not r["permission_authorized"] for r in records)
        recovery = values["recovery_diagnostics"][variant]
        assert recovery["enum_error_tasks"] == sum(r["enum_error_attempts"] > 0 for r in records)
        assert recovery["enum_recovered_tasks"] == sum(r["enum_recovered"] for r in records)
        assert recovery["reason_enum_error_tasks"] == sum(r["reason_enum_errors"] > 0 for r in records)
        assert recovery["reason_enum_recovered_tasks"] == sum(r["reason_enum_recovered"] for r in records)
        assert recovery["initial_reason_enum_error_tasks"] == sum(
            "reason" in r["initial_enum_errors"] for r in records
        )
    assert report["release_gate"] == promotion(report["records"])
    protocol = verify_protocol_evidence(report)
    assert values["protocol_audit"] == protocol
    assert values["protocol_audit_sha256"] == sha(protocol_checks.OUTPUT)
    assert values["protocol_audit_source_sha256"] == sha(ROOT / "training/v7/protocol_audit.py")
    assert values["protocol_audit_plan_sha256"] == sha(protocol_checks.PLAN)
    assert validated_report(ROOT)["available"]
    result = {
        "records_reconciled": len(report["records"]),
        "fresh_test_tasks": len(cases),
        "unique_test_tasks": len(cases),
        "analysis_plan_sha256": sha(REPORTS / "analysis-plan.json"),
        "no_test_based_selection": True,
        "business_scores_recomputed": True,
        "raw_actions_replayed": True,
        "raw_actions_and_confirmations_replayed": True,
        "raw_first_predictions_before_repair": True,
        "intent_and_attempt_metrics_reconciled": True,
        "sampling_coverage_recomputed": True,
        "saved_initializers_exactly_match_predecessor": True,
        "initialization_audit_sha256": sha(INITIALIZATION_AUDIT),
        "protocol_audit_sha256": sha(protocol_checks.OUTPUT),
        "protocol_audit_records_reconciled": protocol["records_audited"],
        "protocol_audit_counts_recomputed": True,
        "supplemental_protocol_safety_separate_from_business_gate": True,
        "selected_checkpoint_training_coverage": selected_coverage,
        "evaluation_sha256": sha(REPORTS / "evaluation.json"),
        "source_sha256": sha(__file__),
        "scope": "Independent code/trace/export verification; not independent human annotation.",
    }
    (REPORTS / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    (REPORTS / "data-quality.json").write_text(
        json.dumps(
            {
                **quality,
                "analysis_plan_sha256": sha(REPORTS / "analysis-plan.json"),
                "analysis_plan_precedes_test_events": True,
                "physical_training_pairs_deduplicated": True,
                "prior_worlds_not_reused": True,
            },
            indent=2,
        )
        + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(verify()))
