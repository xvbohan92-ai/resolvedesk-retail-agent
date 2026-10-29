"""Model-free V8 development construction and executable teacher trajectories.

No V8 test module or file is read here. The historical list is explicit so future
test additions cannot enter development-data construction through a directory glob.
"""

import argparse
import copy
import json
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

from resolve_desk.progress import clarification_memory
from training.common import ROOT, sha
from training.cuda_eval.prepare import (
    FORBIDDEN_MODEL_KEYS,
    canonical,
    catalog_labels,
    denominators,
    digest_value,
    entities,
    public_case,
)
from training.v7.scoring import score
from training.v8.common import CONFIG, DATA, INITIAL_ADAPTER_SHA256, read_rows
from training.v8.fixtures import FAMILIES, SPLITS, make_case

HISTORICAL_NAMES = (
    "training/datasets/cases.json",
    "training/v2/datasets/cases.json",
    "training/v3/datasets/cases.json",
    "training/v4/datasets/cases.json",
    "training/v5/datasets/cases.json",
    "training/v6/datasets/cases.json",
    "training/v6_1/datasets/cases.json",
    "training/v7/datasets/cases.json",
    "training/cuda_eval/datasets/cases.json",
    "training/cuda_eval2/datasets/cases.json",
)
INTERACTION_FIELDS = (
    "initial_message",
    "customer",
    "orders",
    "products",
    "answers",
    "reply_sequences",
    "confirmation",
    "required_answers",
)
PRIVATE_KEYS = FORBIDDEN_MODEL_KEYS | {
    "reply_sequences",
    "instruction_text",
    "development_profile",
    "split",
    "evidence_initial",
}


def runtime():
    from resolve_desk.resolution_v8 import ResolutionStore, decision_gate, messages

    return ResolutionStore, decision_gate, messages


def instruction_skeleton(case):
    text = case.get("instruction_text", case.get("evidence_initial", case["initial_message"]))
    for token in sorted(entities(case) | catalog_labels(case), key=len, reverse=True):
        text = re.sub(re.escape(token), "<entity>", text, flags=re.IGNORECASE)
    return " ".join(text.casefold().split())


def checked_messages(state):
    _, _, messages = runtime()
    result = messages(state)
    visible = json.loads(result[1]["content"])
    assert set(visible).isdisjoint(PRIVATE_KEYS), set(visible) & PRIVATE_KEYS
    return result


def reference(case, state):
    """Authored semantic labels merge only actual replies to actual questions."""
    _, decision_gate, _ = runtime()
    action = copy.deepcopy(decision_gate(state)["next_action"])
    if action["tool"] != "record_request":
        return action
    intent = dict(case["intent_initial"])
    for pair in clarification_memory(state):
        field = pair["field"]
        if field == "fallback_return":
            continue
        choices = case["reply_sequences"].get(field, [])
        matched = [answer for answer in choices if answer["text"] == pair["reply"]]
        assert len(matched) == 1, (case["id"], pair)
        intent["requested_option" if field == "new_option" else field] = matched[0]["value"]
    return {
        "tool": "record_request",
        "arguments": {
            **intent,
            "evidence": case["evidence_initial"]
            if len(state["messages"]) == 1
            else state["messages"][-1]["content"],
        },
    }


def recovery_kinds(case):
    if case["split"] != "v8-train":
        return []
    family, n = case["family"], case["development_profile"]
    kinds = []
    if family in {"exchange_fallback", "exchange_fallback_available", "exchange_available_refusal", "cancel"}:
        kinds.append("nonliteral_quote")
    if family.startswith("exchange") and "new_option" not in case["required_answers"] and n % 3 != 2:
        kinds.append("option_product_suffix")
    if "payment_method" in case["required_answers"]:
        kinds.append("order_is_not_payment")
    if family in {"cancel", "missing_reason", "exchange_unavailable", "exchange_stockout_negative_quote"}:
        kinds.append("reason_enum")
    if family in {
        "exchange_unavailable",
        "exchange_stockout_missing_email",
        "exchange_stockout_negative_quote",
        "exchange_stockout_question",
    }:
        kinds.append("unsupported_grant")
    if (
        family in {"missing_reason", "exchange_missing_option", "missing_email", "exchange_missing_email"}
        and n % 2
    ):
        kinds.append("stale_quote")
    if family in {
        "missing_email",
        "exchange_missing_email",
        "exchange_missing_option",
        "missing_email_payment",
    }:
        kinds.append("repeated_usable_question")
    if family == "exchange_missing_option":
        kinds.append("operation_change")
    return kinds


def scoped_return_check(state, action):
    """Reconstruct the real scoped YES from observed data, without business gold."""
    from resolve_desk.resolution_v8 import current_request
    from resolve_desk.workflow import completed_reads

    request = current_request(state)
    assert request and request["operation"] == "exchange"
    reads, args = completed_reads(state), action["arguments"]
    order = next(
        o["result"]
        for o in reversed(reads)
        if o["action"]["tool"] == "get_order" and o["result"].get("id") == args["order_id"]
    )
    item = next(item for item in order["items"] if item["item_id"] == args["item_id"])
    product = next(
        o["result"]
        for o in reversed(reads)
        if o["action"]["tool"] == "get_product" and o["result"].get("product_id") == item["product_id"]
    )
    target = next(
        variant
        for variant in product["variants"]
        if variant["option"] == request["requested_option"] and variant["item_id"] != item["item_id"]
    )
    assert target["available"] is False and order["status"] == "delivered"
    scope = {
        "order_id": order["id"],
        "new_item_id": target["item_id"],
        "order_revision": state["read_orders"][order["id"]],
        "request": {
            key: request[key]
            for key in ("operation", "email", "order_id", "requested_option", "payment_method")
        },
    }
    active = {pair["reply_index"] for pair in clarification_memory(state)}
    matches = [
        observation["result"]
        for observation in state["observations"]
        if observation["action"]
        and observation["action"]["tool"] == "ask_customer"
        and observation["action"]["arguments"]["field"] == "fallback_return"
        and "error" not in observation["result"]
        and observation["result"].get("consent_scope") == scope
        and observation["result"].get("asked_at_message_count") in active
    ]
    assert (
        matches
        and state["messages"][matches[-1]["asked_at_message_count"]]["content"].strip().casefold() == "yes"
    )
    assert args["order_id"] == request["order_id"]
    if request["payment_method"] == "original":
        assert args["payment_method"] == "original"
    else:
        customer = next(o["result"] for o in reversed(reads) if o["action"]["tool"] == "identify_customer")
        assert args["payment_method"] == customer["gift_card"]["id"]
    return {"scope": scope, "reply_index": matches[-1]["asked_at_message_count"], "scoped_yes_verified": True}


def run_reference(case, store, recovery=None):
    """Execute correct targets; synthetic mistakes are recovery inputs only."""
    _, decision_gate, _ = runtime()
    state = store.create(public_case(case))
    rows, questions, replies, faults, scope_checks = [], [], [], [], []
    reply_counts, injected, total_calls = Counter(), False, 0

    def apply(action, *, wrong=False, expected_error=None):
        nonlocal state, total_calls, injected
        total_calls += 1
        assert total_calls <= 12, (case["id"], recovery, "12-call reference budget exhausted")
        input_hash = digest_value(checked_messages(state))
        if (
            not wrong
            and action["tool"] == "propose_return"
            and case["intent_initial"]["operation"] == "exchange"
        ):
            scope_checks.append(scoped_return_check(state, action))
        if not wrong:
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
                        *checked_messages(state),
                        {"role": "assistant", "content": json.dumps(action, separators=(",", ":"))},
                    ],
                }
            )
        state = store.apply(
            state["id"],
            json.dumps(action),
            {"provider": "synthetic_input_fault" if wrong else "cpu_reference_oracle"},
            state["revision"],
        )
        result = state["observations"][-1]["result"]
        if wrong:
            if expected_error:
                assert result.get("error") == expected_error, (case["id"], recovery, result)
            faults.append({"action": action, "result": result, "input_prompt_sha256": input_hash})
            injected = True
        else:
            assert "error" not in result or (
                case["family"] == "foreign_order" and action["tool"] == "get_order"
            ), (case["id"], recovery, result)

    if recovery in {
        "nonliteral_quote",
        "option_product_suffix",
        "order_is_not_payment",
        "reason_enum",
        "unsupported_grant",
    }:
        wrong = reference(case, state)
        expected_error = None
        if recovery == "nonliteral_quote":
            wrong["arguments"]["evidence"] = "In other words: " + case["evidence_initial"]
            expected_error = "evidence_validation_error"
        elif recovery == "option_product_suffix":
            wrong["arguments"]["requested_option"] += " " + (
                "hoodie" if case["split"] == "v8-train" else "pullover"
            )
        elif recovery == "order_is_not_payment":
            wrong["arguments"]["payment_method"] = "original"
        elif recovery == "reason_enum":
            wrong["arguments"]["reason"] = (
                "I do not need the purchase any more"
                if wrong["arguments"]["operation"] == "cancel"
                else "Exchange only; this is not a cancellation"
            )
            expected_error = "intent_validation_error"
        else:
            wrong["arguments"]["fallback_return"] = "yes"
        apply(wrong, wrong=True, expected_error=expected_error)
    while state["status"] in {"running", "awaiting_customer"}:
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            questions.append(field)
            if field == "fallback_return":
                text, value = case["answers"][field], case["answers"][field].lower()
            else:
                assert field in case["reply_sequences"], (case["id"], recovery, "unexpected question", field)
                sequence = case["reply_sequences"][field]
                index = reply_counts[field]
                assert index < len(sequence), (case["id"], recovery, "unnecessary repeated question", field)
                text, value = sequence[index]["text"], sequence[index]["value"]
            reply_counts[field] += 1
            replies.append({"field": field, "text": text, "value": value, "after_actual_question": True})
            state = store.customer_message(state["id"], text, state["revision"])
        gate = decision_gate(state)["next_action"]
        if (
            recovery == "operation_change"
            and not injected
            and len(state["messages"]) > 1
            and gate["tool"] == "record_request"
        ):
            wrong = reference(case, state)
            if wrong["arguments"]["requested_option"] != "unknown":
                wrong["arguments"]["operation"] = "return"
                apply(wrong, wrong=True, expected_error="operation_change_validation_error")
                # The next gate requires real customer operation confirmation.
                continue
        if (
            recovery == "stale_quote"
            and not injected
            and len(state["messages"]) > 1
            and gate["tool"] == "record_request"
        ):
            wrong = reference(case, state)
            wrong["arguments"]["evidence"] = case["evidence_initial"]
            apply(wrong, wrong=True, expected_error="evidence_validation_error")
        if recovery == "repeated_usable_question" and not injected and gate["tool"] != "record_request":
            usable = [
                pair
                for pair in clarification_memory(state)
                if any(
                    answer["text"] == pair["reply"] and answer["value"] != "unknown"
                    for answer in case["reply_sequences"].get(pair["field"], [])
                )
            ]
            if usable:
                previous = usable[-1]
                apply(
                    {
                        "tool": "ask_customer",
                        "arguments": {"field": previous["field"], "message": previous["question"]},
                    },
                    wrong=True,
                )
                assert "error" in faults[-1]["result"], (case["id"], recovery, faults[-1])
        apply(reference(case, state))
    assert state["status"] in {"awaiting_confirmation", "handoff"}, (case["id"], recovery, state)
    if recovery:
        assert injected, (case["id"], "recovery was not exercised", recovery)
    before, proposal = store.orders(state["id"]), state["proposal"]
    if state["status"] == "awaiting_confirmation":
        store.confirm(state["id"], proposal["id"], state["revision"], case["confirmation"])
        state = store.get(state["id"])
    judged = score(case, {**state, "proposal": proposal}, store.orders(state["id"]), before, questions)
    assert judged["task_success"], (case["id"], recovery, judged, state)
    return rows, {
        "case_id": case["id"],
        "trajectory": recovery or "clean_reference",
        "model_calls": total_calls,
        "questions": questions,
        "actual_replies": replies,
        "input_faults": faults,
        "scoped_return_checks": scope_checks,
        "proposal": proposal,
        "final_status": state["status"],
        **judged,
    }


def quality_audit(cases, rows):
    Store, _, _ = runtime()
    historical = [case for name in HISTORICAL_NAMES for case in json.loads((ROOT / name).read_text())]
    assert len(historical) == 1323
    old_entities = set.union(*(entities(case) for case in historical))
    old_worlds = {order["id"] for case in historical for order in case["orders"]}
    old_catalog = set.union(*(catalog_labels(case) for case in historical))
    old_skeletons = {instruction_skeleton(case) for case in historical}
    old_messages = {case["initial_message"] for case in historical}
    old_ids = {case["id"] for case in historical}
    by_split, fingerprints = {}, {}
    with tempfile.TemporaryDirectory(prefix="resolvedesk-v8-inputs-") as tmp:
        store = Store(Path(tmp) / "inputs.sqlite")
        old_prompts = {digest_value(checked_messages(store.create(public_case(case)))) for case in historical}
        for split, count in SPLITS.items():
            cohort = [case for case in cases if case["split"] == split]
            ids = {case["id"] for case in cohort}
            worlds = {order["id"] for case in cohort for order in case["orders"]}
            entity_sets = [entities(case) for case in cohort]
            all_entities = set.union(*entity_sets)
            labels = set.union(*(catalog_labels(case) for case in cohort))
            skeletons = {instruction_skeleton(case) for case in cohort}
            texts = {case["initial_message"] for case in cohort}
            prompts = {
                case["id"]: digest_value(checked_messages(store.create(public_case(case)))) for case in cohort
            }
            interactions = {canonical({key: case[key] for key in INTERACTION_FIELDS}) for case in cohort}
            assert (
                len(cohort)
                == 24 * count
                == len(ids)
                == len(worlds)
                == len(skeletons)
                == len(texts)
                == len(set(prompts.values()))
                == len(interactions)
            )
            assert sum(map(len, entity_sets)) == len(all_entities)
            overlaps = {
                "case_ids": ids & old_ids,
                "worlds": worlds & old_worlds,
                "entities": all_entities & old_entities,
                "catalog_labels": labels & old_catalog,
                "instruction_skeletons": skeletons & old_skeletons,
                "initial_messages": texts & old_messages,
                "initial_model_prompts": set(prompts.values()) & old_prompts,
            }
            assert not any(overlaps.values()), overlaps
            by_split[split] = {
                "case_ids": ids,
                "worlds": worlds,
                "entities": all_entities,
                "catalog_labels": labels,
                "instruction_skeletons": skeletons,
                "initial_messages": texts,
                "initial_model_prompts": set(prompts.values()),
            }
            fingerprints[split] = {
                "case_ids": sorted(digest_value(value) for value in ids),
                "worlds": sorted(digest_value(value) for value in worlds),
                "entities": sorted(digest_value(value) for value in all_entities),
                "catalog_labels": sorted(digest_value(value) for value in labels),
                "instruction_skeletons": sorted(digest_value(value) for value in skeletons),
                "initial_messages": sorted(digest_value(value) for value in texts),
                "public_inputs": sorted(digest_value(public_case(case)) for case in cohort),
                "initial_model_prompts": prompts,
            }
    assert all(not by_split["v8-train"][key] & by_split["v8-valid"][key] for key in by_split["v8-train"])
    training_pairs = [canonical(row["messages"]) for row in rows["v8-train"]]
    assert len(set(training_pairs)) == len(training_pairs)
    return {
        "historical_datasets_checked": len(HISTORICAL_NAMES),
        "historical_cases_checked": len(historical),
        "historical_inputs_sha256": {name: sha(ROOT / name) for name in HISTORICAL_NAMES},
        "counts": {
            split: {key: len(values) for key, values in group.items()} for split, group in by_split.items()
        },
        "fingerprints": fingerprints,
        "hash_encoding": "sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode())",
        "skeleton_scope": (
            "Full authored instruction body, casefolded/whitespace normalized after replacing "
            "catalog labels and independent entity strings; no artificial split prefix."
        ),
        "all_prior_and_cross_split_exact_overlaps_empty": True,
        "unique_training_prompt_completion_pairs": len(training_pairs),
        "v8_test_inputs_read": False,
        "authorship_scope": (
            "Self-authored synthetic English under inherited policy/families; not an external, "
            "official or independently human-authored benchmark."
        ),
    }


def generate():
    from training.v7.common import verify_frozen

    verify_frozen()
    assert not (DATA / "manifest.json").exists(), "Never modify sealed V8 development data"
    cases = [
        make_case(split, family, n)
        for split, count in SPLITS.items()
        for family in FAMILIES
        for n in range(count)
    ]
    rows, audits = {split: [] for split in SPLITS}, []
    Store, _, _ = runtime()
    with tempfile.TemporaryDirectory(prefix="resolvedesk-v8-reference-") as tmp:
        store = Store(Path(tmp) / "references.sqlite")
        for case in cases:
            for recovery in [None, *recovery_kinds(case)]:
                targets, audit = run_reference(case, store, recovery)
                rows[case["split"]].extend(targets)
                audits.append(audit)
    raw_counts = {split: len(values) for split, values in rows.items()}
    for split in rows:
        seen, unique = set(), []
        for row in rows[split]:
            key = canonical(row["messages"])
            if key not in seen:
                unique.append(row)
                seen.add(key)
        rows[split] = unique
    random.Random(202610288).shuffle(rows["v8-train"])
    quality = quality_audit(cases, rows)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (DATA / "public-valid.json").write_text(
        json.dumps([public_case(case) for case in cases if case["split"] == "v8-valid"], indent=2) + "\n"
    )
    for split, targets in rows.items():
        (DATA / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in targets))
    reference_receipt = {
        "scope": (
            "CPU oracle feasibility and executed correct teacher targets, not model accuracy. "
            "Synthetic faults occur only as recovery inputs."
        ),
        "cases_sha256": sha(DATA / "cases.json"),
        "all_business_references_pass": all(audit["task_success"] for audit in audits),
        "all_references_within_12_total_calls_including_faults": all(
            audit["model_calls"] <= 12 for audit in audits
        ),
        "actual_scripted_replies_only_after_requested_field": True,
        "scoped_return_proposals_checked": sum(len(audit["scoped_return_checks"]) for audit in audits),
        "all_exchange_return_proposals_have_actual_scoped_yes": all(
            check["scoped_yes_verified"] for audit in audits for check in audit["scoped_return_checks"]
        ),
        "reference_sources_sha256": {
            name: sha(ROOT / name)
            for name in (
                "resolve_desk/resolution_v8.py",
                "resolve_desk/progress.py",
                "training/v6/scoring.py",
                "training/v2/scoring.py",
                "training/scoring.py",
                "training/v8/fixtures.py",
                "training/v8/prepare.py",
            )
        },
        "records": audits,
    }
    (DATA / "reference-audit.json").write_text(json.dumps(reference_receipt, indent=2) + "\n")
    pools = Counter(
        "intent_and_record_repair" if row["origin"] == "record_request" else "execution_and_reply"
        for row in rows["v8-train"]
    )
    summary = {
        "scope": quality["authorship_scope"],
        "initial_adapter_sha256": INITIAL_ADAPTER_SHA256,
        "tasks": dict(Counter(case["split"] for case in cases)),
        "rows": {split: len(targets) for split, targets in rows.items()},
        "raw_rows_before_physical_deduplication": raw_counts,
        "removed_exact_prompt_completion_duplicates": {
            split: raw_counts[split] - len(targets) for split, targets in rows.items()
        },
        "train_sampling_buckets": dict(pools),
        "train_sample_roles": dict(Counter(row["sample_role"] for row in rows["v8-train"])),
        "train_actions": dict(Counter(row["origin"] for row in rows["v8-train"])),
        "recovery_trajectories": dict(
            Counter(audit["trajectory"] for audit in audits if audit["trajectory"] != "clean_reference")
        ),
        "denominators": {
            split: denominators([case for case in cases if case["split"] == split]) for split in SPLITS
        },
        "cases_sha256": sha(DATA / "cases.json"),
        "reference_audit_sha256": sha(DATA / "reference-audit.json"),
        "data_sha256": {f"{split}.jsonl": sha(DATA / f"{split}.jsonl") for split in SPLITS},
        "quality": quality,
        "all_reference_tasks_pass": reference_receipt["all_business_references_pass"],
        "model_inference_performed": False,
        "training_repetitions_are_not_additional_unique_pairs": True,
    }
    (DATA / "data-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "quality"}, indent=2))
    return summary


def audit_tokens(tokenizer):
    """Check every full development pair on CPU without truncating any target."""
    from training.v7.tokenization import encode_row
    from training.v7.train import balanced_sample_indices, projection_batch, sampling_buckets

    assert not (DATA / "manifest.json").exists(), "Never overwrite a sealed token audit"
    config = json.loads(CONFIG.read_text())
    assert config["max_seq_length"] == 3200
    assert config["projection_window"] == 192
    result = {
        "scope": "CPU tokenizer and deterministic schedule audit, not training or model-quality evidence.",
        "model_or_test_inference_performed": False,
        "initial_adapter_sha256": INITIAL_ADAPTER_SHA256,
        "projection_window": 192,
        "max_seq_length": 3200,
        "sequence_length_decision": {
            "initial_limit": 3072,
            "chosen_limit": 3200,
            "reason": (
                "Before model training or V8 test inference, retain every authored operation-change "
                "recovery and all context/targets. Two training rows exceeded 3072; the parent "
                "explicitly chose 3200 instead of deleting complex trajectories or truncating inputs."
            ),
            "test_inputs_or_candidate_outputs_used": False,
        },
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_inputs_sha256": {
            str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / config["model"]).glob("*.json"))
        },
        "config_sha256": sha(CONFIG),
        "splits": {},
    }
    for split in ("train", "valid"):
        rows = read_rows(split)
        encoded = [encode_row(row, tokenizer, 8.0, 4.0) for row in rows]
        tails, padding = [], []
        for example in encoded:
            tokens, weights = projection_batch(example, 192, 3200)
            original = example["weights"]
            assert sum(weights) == sum(original)
            assert sum(weight > 0 for weight in weights) == sum(weight > 0 for weight in original)
            assert not any(original[: example["prompt_length"] - 1])
            padded = original + [0.0] * (len(tokens) - 1 - len(original))
            first = next(index for index, weight in enumerate(padded) if weight)
            tails.append(len(padded) - first)
            padding.append(len(tokens) - len(example["tokens"]))
        over_original = [
            {
                "case_id": rows[index]["case_id"],
                "trajectory": rows[index]["trajectory"],
                "origin": rows[index]["origin"],
                "sequence_tokens": len(example["tokens"]),
                "completion_tokens": len(example["tokens"]) - example["prompt_length"],
            }
            for index, example in enumerate(encoded)
            if len(example["tokens"]) > 3072
        ]
        result["splits"][split] = {
            "physical_rows": len(rows),
            "data_sha256": sha(DATA / f"v8-{split}.jsonl"),
            "longest_sequence": max(len(example["tokens"]) for example in encoded),
            "longest_completion": max(
                len(example["tokens"]) - example["prompt_length"] for example in encoded
            ),
            "maximum_tail_including_padding": max(tails),
            "maximum_padding_tokens": max(padding),
            "original_3072_limit_exceeded_rows": over_original,
            "all_inputs_and_supervised_targets_retained": True,
            "all_prompt_and_padding_weights_zero": True,
            "permission_value_tokens": sum(example["permission_tokens"] for example in encoded),
            "reason_value_tokens": sum(example["reason_tokens"] for example in encoded),
        }
        if split == "train":
            schedule = balanced_sample_indices(rows, config["seed"], config["iters"])
            indices = [index for _, index in schedule]
            result["sampling"] = {
                "updates": len(schedule),
                "physical_rows": len(rows),
                "unique_rows_sampled": len(set(indices)),
                "unique_cases_sampled": len({rows[index]["case_id"] for index in indices}),
                "all_physical_rows_used": len(set(indices)) == len(rows),
                "bucket_rows": {name: len(pool) for name, pool in sampling_buckets(rows).items()},
                "bucket_updates": dict(Counter(name for name, _ in schedule)),
                "schedule_sha256": digest_value(schedule),
                "scope": (
                    "Sampling repetitions are optimizer exposures, not additional unique training pairs."
                ),
            }
    (DATA / "token-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = json.loads((DATA / "data-summary.json").read_text())
    summary["token_audit_sha256"] = sha(DATA / "token-audit.json")
    summary["training_sampling_audit"] = result["sampling"]
    summary["full_context_and_target_audit"] = result["splits"]
    summary["sequence_length_decision"] = result["sequence_length_decision"]
    (DATA / "data-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "audit-tokens"])
    args = parser.parse_args()
    if args.mode == "generate":
        generate()
    else:
        from transformers import AutoTokenizer

        configuration = json.loads(CONFIG.read_text())
        tokenizer = AutoTokenizer.from_pretrained(ROOT / configuration["model"], local_files_only=True)
        print(json.dumps(audit_tokens(tokenizer), indent=2))
