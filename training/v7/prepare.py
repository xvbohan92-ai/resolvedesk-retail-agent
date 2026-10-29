"""Independent reference trajectories and deduplicated SFT inputs; root owns freezing."""

import argparse
import json
import random
import tempfile
from collections import Counter
from pathlib import Path

from resolve_desk.progress import clarification_memory
from resolve_desk.resolution_v7 import ResolutionStore, decision_gate, messages
from training.common import ROOT
from training.v7.fixtures import FAMILIES, make_case
from training.v7.scoring import score

DATA, REPORTS = ROOT / "training/v7/datasets", ROOT / "reports/v7"
INTERACTION_FIELDS = (
    "initial_message",
    "customer",
    "orders",
    "products",
    "answers",
    "confirmation",
    "required_answers",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def reference(case, state):
    gate = decision_gate(state)["next_action"]
    if gate["tool"] != "record_request":
        return gate
    intent = dict(case["intent_initial"])
    for pair in clarification_memory(state):
        key = "requested_option" if pair["field"] == "new_option" else pair["field"]
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


def run_reference(case, store, recovery=None):
    """Return (correct model-target rows, independently judged audit), with no LLM.

    A synthetic wrong action is applied only to construct a recovery INPUT. It is
    never appended as an assistant training target. Every subsequent target is a
    valid model-issued tool action and all customer replies follow actual asks.
    """
    state, rows, questions, injected = store.create(case), [], [], False
    input_faults = []

    def fault(action, expected_error=None):
        nonlocal state, injected
        state = store.apply(
            state["id"], json.dumps(action), {"provider": "synthetic_input_fault"}, state["revision"]
        )
        result = state["observations"][-1]["result"]
        if expected_error:
            assert result.get("error") == expected_error, (case["id"], recovery, result)
        input_faults.append({"action": action, "result": result})
        injected = True

    if recovery in {"enum_exchange", "enum_cancel"}:
        invalid = reference(case, state)
        invalid["arguments"]["reason"] = (
            "Do not send this item back; a person should assist"
            if recovery == "enum_exchange"
            else "I submitted the purchase accidentally"
            if case["expected"]["reason"] == "ordered by mistake"
            else "I have no remaining use for this purchase"
        )
        fault(invalid, "intent_validation_error")
        assert state["status"] == "running"
    elif recovery == "false_yes":
        invalid = reference(case, state)
        invalid["arguments"]["fallback_return"] = "yes"
        fault(invalid)
    for _ in range(12):
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            questions.append(field)
            state = store.customer_message(state["id"], case["answers"][field], state["revision"])
        if state["status"] != "running":
            break
        gate = decision_gate(state)["next_action"]
        if (
            recovery == "stale_evidence"
            and not injected
            and len(state["messages"]) > 1
            and gate["tool"] == "record_request"
        ):
            wrong = reference(case, state)
            wrong["arguments"]["evidence"] = case["evidence_initial"]
            fault(wrong)
            assert "error" in input_faults[-1]["result"]
        if (
            recovery == "repeat_ask"
            and not injected
            and clarification_memory(state)
            and gate["tool"] != "record_request"
        ):
            previous = clarification_memory(state)[-1]
            fault(
                {
                    "tool": "ask_customer",
                    "arguments": {"field": previous["field"], "message": previous["question"]},
                }
            )
            assert "error" in input_faults[-1]["result"]
        action = reference(case, state)
        rows.append(
            {
                "case_id": case["id"],
                "origin": action["tool"],
                "trajectory": recovery or "clean_reference",
                "sample_role": "initial_record"
                if not state["observations"]
                else "record_recovery"
                if action["tool"] == "record_request"
                else "tool_action",
                "messages": [
                    *messages(state),
                    {"role": "assistant", "content": json.dumps(action, separators=(",", ":"))},
                ],
            }
        )
        state = store.apply(state["id"], json.dumps(action), {"provider": "reference"}, state["revision"])
        result = state["observations"][-1]["result"]
        assert "error" not in result or (
            case["family"] == "foreign_order" and action["tool"] == "get_order"
        ), (case["id"], recovery, result)
    if recovery:
        assert injected, (case["id"], "requested recovery did not occur", recovery)
    before, proposal = store.orders(state["id"]), state["proposal"]
    if state["status"] == "awaiting_confirmation":
        store.confirm(state["id"], proposal["id"], state["revision"], case["confirmation"])
        state = store.get(state["id"])
    judged = score(case, {**state, "proposal": proposal}, store.orders(state["id"]), before, questions)
    assert judged["task_success"], (case["id"], recovery, judged, state)
    audit = {
        "case_id": case["id"],
        "trajectory": recovery or "clean_reference",
        "input_faults": input_faults,
        **judged,
    }
    return rows, audit


def recovery_kinds(case):
    if case["split"] != "train":
        return []
    f = case["family"]
    kinds = []
    if f in {
        "exchange_unavailable",
        "exchange_available_refusal",
        "exchange_stockout_negative_quote",
        "exchange_stockout_question",
        "exchange_fallback",
        "exchange_fallback_missing_payment",
    }:
        kinds.append("enum_exchange")
    if f in {"cancel", "missing_reason"}:
        kinds.append("enum_cancel")
    if f in {"missing_email", "exchange_missing_option", "exchange_missing_payment", "missing_email_payment"}:
        kinds.append("repeat_ask")
    if int(case["id"].rsplit("-", 1)[1]) % 2 and f in {
        "missing_reason",
        "missing_payment",
        "exchange_missing_option",
        "exchange_missing_payment",
        "exchange_fallback_missing_payment",
        "missing_email_payment",
    }:
        kinds.append("stale_evidence")
    if f in {
        "exchange_unavailable",
        "exchange_stockout_missing_email",
        "exchange_stockout_negative_quote",
        "exchange_stockout_question",
    }:
        kinds.append("false_yes")
    return kinds


def uniqueness(cases, rows):
    def entity_ids(case):
        values = {case["customer"]["id"], case["customer"]["email"]}
        if "gift_card" in case["customer"]:
            values.add(case["customer"]["gift_card"]["id"])
        for product_id, product in case["products"].items():
            values.add(product_id)
            values.update(v["item_id"] for v in product["variants"])
        for order in case["orders"]:
            values.update(order[k] for k in ["id", "user_id", "original_payment"])
            values.update(item[k] for item in order["items"] for k in ["item_id", "product_id"])
        return values

    groups = {}
    initial = {}
    worlds = {}
    entities = {}
    for split in ["train", "valid", "test"]:
        cohort = [c for c in cases if c["split"] == split]
        groups[split] = len({canonical({k: c[k] for k in INTERACTION_FIELDS}) for c in cohort})
        with tempfile.TemporaryDirectory() as tmp:
            store = ResolutionStore(Path(tmp) / "initial.sqlite")
            initial[split] = {canonical(messages(store.create(c))) for c in cohort}
        worlds[split] = {c["orders"][0]["id"] for c in cohort}
        entities[split] = set.union(*(entity_ids(c) for c in cohort))
        assert groups[split] == len(cohort) == len(initial[split]) == len(worlds[split])
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        assert not worlds[a] & worlds[b] and not initial[a] & initial[b] and not entities[a] & entities[b]
    previous_worlds, previous_entities, previous_messages = set(), set(), set()
    for path in (ROOT / "training").glob("**/datasets/cases.json"):
        if path.parent == DATA:
            continue
        old_cases = json.loads(path.read_text())
        previous_worlds.update(o["id"] for c in old_cases for o in c["orders"])
        previous_entities.update(set.union(*(entity_ids(c) for c in old_cases)))
        previous_messages.update(c["initial_message"] for c in old_cases)
    assert not previous_worlds & set.union(*worlds.values())
    assert not previous_entities & set.union(*entities.values())
    assert not previous_messages & {c["initial_message"] for c in cases}
    return {
        "unique_interactions": groups,
        "unique_initial_model_prompts": {s: len(v) for s, v in initial.items()},
        "unique_order_worlds": {s: len(v) for s, v in worlds.items()},
        "unique_entities": {s: len(v) for s, v in entities.items()},
        "unique_training_prompt_completion_pairs": len({canonical(r["messages"]) for r in rows["train"]}),
        "split_entities_and_initial_inputs_disjoint": True,
        "no_prior_order_worlds_reused": True,
        "no_prior_entities_or_initial_messages_reused": True,
        "interaction_fields": list(INTERACTION_FIELDS),
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
    with tempfile.TemporaryDirectory() as tmp:
        store = ResolutionStore(Path(tmp) / "references.sqlite")
        for case in cases:
            for recovery in [None, *recovery_kinds(case)]:
                examples, audit = run_reference(case, store, recovery)
                rows[case["split"]].extend(examples)
                audits.append(audit)
    raw_train = len(rows["train"])
    seen = set()
    deduped = []
    for row in rows["train"]:
        key = canonical(row["messages"])
        if key not in seen:
            deduped.append(row)
            seen.add(key)
    rows["train"] = deduped
    random.Random(202610287).shuffle(rows["train"])
    quality = uniqueness(cases, rows)
    for split, samples in rows.items():
        (DATA / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    (REPORTS / "reference-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    valid = [
        {k: c[k] for k in ("id", "family", "initial_message", "customer", "orders", "products")}
        for c in cases
        if c["split"] == "valid"
    ]
    (ROOT / "resolve_desk/fixtures/validation-v7.json").write_text(json.dumps(valid, indent=2) + "\n")
    summary = {
        "tasks": dict(Counter(c["split"] for c in cases)),
        "rows": {s: len(r) for s, r in rows.items()},
        "raw_train_rows_before_deduplication": raw_train,
        "duplicate_training_pairs_removed": raw_train - len(rows["train"]),
        "recovery_trajectories": sum(a["trajectory"] != "clean_reference" for a in audits),
        "recovery_types": dict(
            Counter(a["trajectory"] for a in audits if a["trajectory"] != "clean_reference")
        ),
        "train_actions": dict(Counter(r["origin"] for r in rows["train"])),
        "train_sample_roles": dict(Counter(r["sample_role"] for r in rows["train"])),
        "permissions": {
            s: dict(Counter(c["intent_initial"]["fallback_return"] for c in cases if c["split"] == s))
            for s in rows
        },
        "training_record_permission_targets": dict(
            Counter(
                json.loads(r["messages"][-1]["content"])["arguments"]["fallback_return"]
                for r in rows["train"]
                if r["origin"] == "record_request"
            )
        ),
        "all_reference_tasks_pass": True,
        **quality,
        "scope": (
            "Self-authored synthetic language and independent worlds. Wrong synthetic actions are "
            "recovery input state only, never supervised labels. Repeated inputs are physically removed "
            "from SFT; sampling repetitions during optimization must be reported separately."
        ),
    }
    (REPORTS / "data-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate"])
    parser.parse_args()
    generate()
