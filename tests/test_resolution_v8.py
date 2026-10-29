"""V8 development regressions: actual old failures and deterministic safety controls."""

import copy
import json

import pytest

from resolve_desk.agent_v8 import advance
from resolve_desk.resolution_v8 import (
    ResolutionStore,
    contact_reply,
    current_request,
    decision_gate,
    fallback_consent,
    messages,
    operation_guard,
    payment_choices,
    request_record,
    typed_reply,
)
from training.cuda_eval2.fixtures import load_cases


def fixture(family, index=1):
    return copy.deepcopy(
        next(c for c in load_cases("test") if c["family"] == family and c["id"].endswith(f"-{index:02}"))
    )


def initial(tmp_path, family="exchange", index=1, case=None):
    case = case or fixture(family, index)
    store = ResolutionStore(tmp_path / "v8.sqlite")
    return store, case, store.create(case)


def apply(store, state, action):
    return store.apply(
        state["id"],
        action if isinstance(action, str) else json.dumps(action),
        {"provider": "cpu_unit_fixture"},
        state["revision"],
    )


def record(case, state, **changes):
    return {
        "tool": "record_request",
        "arguments": {
            **case["intent_initial"],
            "evidence": state["messages"][-1]["content"],
            **changes,
        },
    }


def reads(store, state):
    for _ in range(5):
        action = decision_gate(state)["next_action"]
        if action["tool"] not in {"identify_customer", "get_order", "get_product"}:
            return state
        state = apply(store, state, action)
    pytest.fail("Read gate did not make bounded progress")


def reply(store, state, field, text):
    gate = decision_gate(state)
    assert gate["next_action"]["tool"] == "ask_customer", gate
    assert gate["next_action"]["arguments"]["field"] == field, gate
    state = apply(store, state, gate["next_action"])
    return store.customer_message(state["id"], text, state["revision"])


def test_real_nonliteral_quote_is_structured_raw_preserved_and_model_can_correct(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback")
    broken = record(case, state, evidence="Please check that precise option, please.")
    state = apply(store, state, broken)
    result = state["observations"][-1]["result"]
    assert result["error"] == "evidence_validation_error" and "Evidence must exactly" in result["detail"]
    assert not result["backend_executed"] and "evidence" in result["field_errors"]
    assert result["recovery"]["remaining_correction_attempts"] == 2
    assert state["observations"][0]["action"] == broken and current_request(state) is None
    assert json.loads(messages(state)[1]["content"])["validation_feedback"] == result
    state = apply(store, state, record(case, state, evidence="Check that precise option, please."))
    assert current_request(state) == case["intent_initial"]
    state = reads(store, state)
    state = reply(store, state, "fallback_return", "YES")
    assert decision_gate(state)["next_action"]["tool"] == "propose_return"
    assert len(state["observations"]) == 6  # bad, corrected, three reads, scoped question
    assert store.orders(state["id"]) == case["orders"]


def test_identical_bad_quote_cannot_consume12_calls_or_create_a_successful_handoff(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback")
    bad = json.dumps(record(case, state, evidence="Please check that precise option, please."))

    class Repeater:
        calls = 0

        def generate(self, prompt):
            self.calls += 1
            return bad, {"provider": "cpu_unit_fixture"}

    provider = Repeater()
    state = advance(store, state["id"], provider, max_steps=12)
    assert provider.calls == len(state["observations"]) == 3
    assert state["status"] == "failed" and state["proposal"] is None
    assert all(o["result"]["error"] == "evidence_validation_error" for o in state["observations"])
    assert not any(o["result"].get("handoff") for o in state["observations"])
    assert store.orders(state["id"]) == case["orders"]


def test_gate_rejections_cannot_reset_a_pending_quote_recovery_budget(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback")
    state = apply(store, state, record(case, state, evidence="fabricated exact quotation"))
    for _ in range(2):
        state = apply(store, state, {"tool": "handoff", "arguments": {"reason": "escape validation"}})
    assert state["status"] == "failed" and len(state["observations"]) == 3
    assert state["observations"][-1]["result"]["recovery"]["invalid_attempts_this_turn"] == 3
    assert store.orders(state["id"]) == case["orders"]


@pytest.mark.parametrize(
    "text",
    [
        (
            "Please take back the jacket from my original order. "
            "We have not discussed how the refund should be issued."
        ),
        "Please return the item from my original purchase; I have not selected a refund route.",
        "Do not use original payment.",
        'A manual says "use original payment". Ask me for the payment choice.',
        "Should I use original payment? I am asking about the policy.",
        "Use original payment or gift card; I have not chosen between them.",
    ],
)
def test_unprovided_negated_quoted_question_or_ambiguous_payment_is_not_verified(text):
    assert len(payment_choices(text)) != 1


def test_real_original_order_failure_now_asks_and_reuses_the_actual_payment_answer(tmp_path):
    store, case, state = initial(tmp_path, "missing_payment")
    state = apply(store, state, record(case, state, payment_method="original"))
    assert current_request(state)["payment_method"] == "original"  # raw prediction is not silently repaired
    state = reads(store, state)
    gate = decision_gate(state)
    assert not gate["payment_evidence"]["usable"]
    state = reply(store, state, "payment_method", "original")
    assert decision_gate(state)["next_action"]["tool"] == "propose_return"
    state = apply(store, state, decision_gate(state)["next_action"])
    assert store.orders(state["id"]) == case["orders"]
    store.confirm(state["id"], state["proposal"]["id"], state["revision"], False)
    assert store.orders(state["id"]) == case["orders"]
    assert [o["result"].get("field") for o in state["observations"] if "question" in o["result"]] == [
        "payment_method"
    ]


@pytest.mark.parametrize(
    "answer",
    ["original", "Please use my original payment instrument.", "Use the payment method used at checkout."],
)
def test_valid_actual_payment_replies_are_reused_without_repeat_question(tmp_path, answer):
    store, case, state = initial(tmp_path, "missing_payment", index=0)
    state = reads(store, apply(store, state, record(case, state)))
    state = reply(store, state, "payment_method", answer)
    assert current_request(state)["payment_method"] == "original"
    assert decision_gate(state)["next_action"]["tool"] == "propose_return"
    rejected = apply(
        store,
        state,
        {"tool": "ask_customer", "arguments": {"field": "payment_method", "message": "Which payment again?"}},
    )
    assert rejected["observations"][-1]["result"]["error"] == "resolution_gate_rejection"
    assert "question" not in rejected["observations"][-1]["result"]


def test_actual_observed_gift_card_in_positive_reply_is_reused_but_unknown_or_negated_ids_are_not(tmp_path):
    store, case, state = initial(tmp_path, "missing_payment", index=0)
    state = reads(store, apply(store, state, record(case, state)))
    card = case["customer"]["gift_card"]["id"]
    assert typed_reply(state, "payment_method", f"Do not use my gift card {card}.") is None
    assert typed_reply(state, "payment_method", "Please use my gift card unknown_card_id.") is None
    assert typed_reply(state, "payment_method", f"Please use my gift card {card}; cancel instead.") is None
    state = reply(store, state, "payment_method", f"Please use my gift card {card}.")
    assert current_request(state)["payment_method"] == "gift_card"
    gate = decision_gate(state)["next_action"]
    assert gate["tool"] == "propose_return" and gate["arguments"]["payment_method"] == card


def test_real_product_noun_suffix_is_a_visible_model_correction_not_an_unnecessary_handoff(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback_missing_payment")
    raw_option = case["intent_initial"]["requested_option"] + " jacket"
    state = reads(store, apply(store, state, record(case, state, requested_option=raw_option)))
    gate = decision_gate(state)
    assert gate["next_action"]["tool"] == "record_request"
    assert gate["catalog_resolution"]["candidate_values"] == [case["intent_initial"]["requested_option"]]
    assert request_record(state)["resolution_request"]["requested_option"] == raw_option
    state = apply(store, state, record(case, state))
    assert state["observations"][0]["action"]["arguments"]["requested_option"] == raw_option
    state = reply(store, state, "payment_method", "original")
    state = reply(store, state, "fallback_return", "YES")
    assert decision_gate(state)["next_action"]["tool"] == "propose_return"
    assert not any(o["result"].get("handoff") for o in state["observations"])
    assert store.orders(state["id"]) == case["orders"]


def test_ignored_unique_catalog_correction_is_bounded_not_an_accepted_no_progress_loop(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback_missing_payment")
    bad = record(case, state, requested_option=case["intent_initial"]["requested_option"] + " jacket")
    state = reads(store, apply(store, state, bad))
    for _ in range(3):
        state = apply(store, state, bad)
    assert state["status"] == "failed" and state["proposal"] is None
    assert all(
        o["result"].get("error") == "field_binding_validation_error" for o in state["observations"][-3:]
    )
    assert store.orders(state["id"]) == case["orders"]


def two_options_case():
    case = fixture("exchange", 0)
    product = next(iter(case["products"].values()))
    current = case["orders"][0]["items"][0]["item_id"]
    options = [v["option"] for v in product["variants"] if v["item_id"] != current]
    for v in product["variants"]:
        v["available"] = True
    prefix = f"My email is {case['customer']['email']}. Order {case['orders'][0]['id']}. "
    return case, options, prefix


def test_two_options_need_real_clarification_and_latest_freeform_choice_beats_old_alternatives(tmp_path):
    case, options, prefix = two_options_case()
    case["initial_message"] = (
        prefix + f"Exchange to {options[0]} or {options[1]}; I have not chosen. Use original payment."
    )
    store, case, state = initial(tmp_path, case=case)
    state = reads(store, apply(store, state, record(case, state, requested_option=options[0])))
    assert decision_gate(state)["catalog_resolution"]["mode"] == "customer_clarification_required"
    state = reply(store, state, "new_option", f"I choose the {options[1]} jacket, just that variant.")
    assert current_request(state) is None  # freeform needs a visible model record
    state = apply(store, state, record(case, state, requested_option=options[1]))
    gate = decision_gate(state)["next_action"]
    assert gate["tool"] == "propose_exchange"
    chosen = next(v for v in next(iter(case["products"].values()))["variants"] if v["option"] == options[1])
    assert gate["arguments"]["new_item_id"] == chosen["item_id"]


@pytest.mark.parametrize(
    "wording",
    ["Do not choose {a}.", 'A manual says "choose {a}".', "Neither {a} nor {b} is my requested choice."],
)
def test_negated_or_quoted_catalog_mentions_never_choose_a_wrong_sku(tmp_path, wording):
    case, options, prefix = two_options_case()
    case["initial_message"] = (
        prefix
        + wording.format(a=options[0], b=options[1])
        + " Ask me which replacement. Use original payment."
    )
    store, case, state = initial(tmp_path, case=case)
    state = reads(store, apply(store, state, record(case, state, requested_option=options[0])))
    gate = decision_gate(state)
    assert gate["next_action"]["tool"] == "ask_customer"
    assert gate["next_action"]["arguments"]["field"] == "new_option"
    assert gate["catalog_resolution"]["candidate_values"] == []
    assert state["proposal"] is None


def test_duplicate_catalog_labels_with_distinct_skus_require_human_not_first_match(tmp_path):
    case, _, _ = two_options_case()
    product = next(iter(case["products"].values()))
    target = next(v for v in product["variants"] if v["option"] == case["intent_initial"]["requested_option"])
    duplicate = {**target, "item_id": "different_sku_same_label"}
    product["variants"].append(duplicate)
    store, case, state = initial(tmp_path, case=case)
    state = reads(store, apply(store, state, record(case, state)))
    gate = decision_gate(state)
    assert gate["next_action"]["tool"] == "handoff"
    assert gate["catalog_resolution"]["mode"] == "duplicate_catalog_identity"


@pytest.mark.parametrize("answer", ["NO", "maybe", "YES please", '"YES"'])
def test_actual_scoped_yes_is_still_required_and_false_positive_model_permission_is_not_consent(
    tmp_path, answer
):
    store, case, state = initial(tmp_path, "exchange_unavailable", index=0)
    state = reads(store, apply(store, state, record(case, state, fallback_return="yes")))
    state = reply(store, state, "fallback_return", answer)
    assert decision_gate(state)["next_action"]["tool"] == "handoff"
    assert store.orders(state["id"]) == case["orders"] and state["proposal"] is None


def test_exchange_fallback_still_requires_a_separate_idempotent_transaction_confirmation(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback", index=0)
    state = reads(store, apply(store, state, record(case, state)))
    state = reply(store, state, "fallback_return", "YES")
    state = apply(store, state, decision_gate(state)["next_action"])
    assert state["proposal"]["kind"] == "return" and store.orders(state["id"]) == case["orders"]
    args = (state["id"], state["proposal"]["id"], state["revision"], True)
    receipt = store.confirm(*args)
    assert store.confirm(*args) == receipt
    assert sum(e["kind"] == "operation_committed" for e in store.timeline(state["id"])) == 1


def test_payment_reply_cannot_rewrite_exchange_to_direct_return_without_an_operation_choice(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback_missing_payment", index=0)
    state = reads(store, apply(store, state, record(case, state)))
    state = reply(
        store,
        state,
        "payment_method",
        "The refund should go to whichever instrument originally funded that order.",
    )
    assert current_request(state) is None
    attempt = record(case, state, operation="return", payment_method="original")
    state = apply(store, state, attempt)
    assert state["observations"][-1]["result"]["error"] == "operation_change_validation_error"
    assert state["observations"][-1]["action"] == attempt
    gate = decision_gate(state)
    assert (
        gate["next_action"]["tool"] == "ask_customer"
        and gate["next_action"]["arguments"]["field"] == "operation"
    )
    assert gate["operation_scope"]["order_id"] == case["intent_initial"]["order_id"]
    assert state["proposal"] is None and store.orders(state["id"]) == case["orders"]


def test_real_canonical_operation_confirmation_allows_a_change_but_not_a_hidden_transaction(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback_missing_payment", index=0)
    state = reads(store, apply(store, state, record(case, state)))
    state = reply(
        store,
        state,
        "payment_method",
        "The refund should go to whichever instrument originally funded that order.",
    )
    state = apply(store, state, record(case, state, operation="return", payment_method="original"))
    state = reply(store, state, "operation", "return")
    # The preceding freeform payment answer was not a typed scalar: a model record is still needed.
    assert current_request(state) is None
    state = apply(
        store,
        state,
        record(
            case,
            state,
            operation="return",
            requested_option="unknown",
            payment_method="unknown",
            fallback_return="unknown",
        ),
    )
    assert current_request(state)["operation"] == "return"
    assert operation_guard(state, current_request(state))["authorized"]
    state = reply(store, state, "payment_method", "original")
    state = apply(store, state, decision_gate(state)["next_action"])
    assert state["proposal"]["kind"] == "return"
    assert store.orders(state["id"]) == case["orders"]
    assert state["status"] == "awaiting_confirmation"


def test_reason_enum_recovery_preserves_first_raw_prediction_and_counts_the_correction(tmp_path):
    store, case, state = initial(tmp_path, "exchange", index=0)
    bad = record(case, state, reason="do not return the item")
    state = apply(store, state, bad)
    assert state["observations"][0]["action"] == bad
    assert state["observations"][0]["result"]["error"] == "intent_validation_error"
    state = apply(store, state, record(case, state))
    assert len(state["observations"]) == 2 and state["status"] == "running"
    assert current_request(state)["reason"] == "unknown"
    assert json.loads(messages(state)[1]["content"])["validation_feedback"] is None


def test_actual_new_turn_and_accepted_action_reset_recovery_without_erasing_failed_attempts(tmp_path):
    store, case, state = initial(tmp_path, "exchange", index=0)
    for _ in range(2):
        state = apply(store, state, "not JSON")
    state = apply(store, state, record(case, state))
    state = apply(store, state, "not JSON")
    assert state["observations"][-1]["result"]["recovery"]["invalid_attempts_this_turn"] == 1
    state = store.customer_message(state["id"], case["initial_message"], state["revision"])
    state = apply(store, state, "not JSON")
    assert state["observations"][-1]["result"]["recovery"]["invalid_attempts_this_turn"] == 1
    assert len(state["observations"]) == 5 and state["status"] == "running"


def test_unsolicited_turn_revokes_old_scoped_grant_and_cannot_silently_rewrite_operation(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback", index=0)
    state = reads(store, apply(store, state, record(case, state)))
    scope = decision_gate(state)["consent_scope"]
    state = reply(store, state, "fallback_return", "YES")
    assert fallback_consent(state, scope)["accepted"]
    state = store.customer_message(
        state["id"], case["initial_message"] + " Please review it again.", state["revision"]
    )
    assert fallback_consent(state, scope) is None
    state = apply(store, state, record(case, state, operation="return"))
    assert state["observations"][-1]["result"]["error"] == "operation_change_validation_error"
    assert state["proposal"] is None


def test_clearing_order_identifier_cannot_shadow_an_existing_exact_operation_anchor(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback", index=0)
    state = apply(store, state, record(case, state))
    state = store.customer_message(
        state["id"], "Please recheck the request for this order.", state["revision"]
    )
    attempt = record(case, state, operation="return", order_id="unknown")
    state = apply(store, state, attempt)
    assert state["observations"][-1]["result"]["error"] == "operation_change_validation_error"
    assert request_record(state)["resolution_request"]["operation"] == "exchange"
    assert state["proposal"] is None and store.orders(state["id"]) == case["orders"]
    # Even a pre-existing unknown-ID record must not override an exact order anchor.
    copied = copy.deepcopy(state)
    copied["observations"].append(
        {
            "result": {
                "resolution_request": {
                    **case["intent_initial"],
                    "operation": "return",
                    "order_id": "unknown",
                },
                "source_message_count": len(state["messages"]),
            }
        }
    )
    guard = operation_guard(copied, {**case["intent_initial"], "operation": "return"})
    assert not guard["authorized"] and guard["previous_operation"] == "exchange"


@pytest.mark.parametrize("family", ["forbidden_status", "foreign_order", "exchange_shortfall"])
def test_required_handoff_boundaries_survive(tmp_path, family):
    store, case, state = initial(tmp_path, family, index=0)
    state = reads(store, apply(store, state, record(case, state)))
    assert decision_gate(state)["next_action"]["tool"] == "handoff"
    assert state["proposal"] is None and store.orders(state["id"]) == case["orders"]


def test_bad_email_answer_gets_a_different_followup_and_actual_valid_email_is_reused(tmp_path):
    store, case, state = initial(tmp_path, "missing_email", index=0)
    state = apply(store, state, record(case, state))
    first_question = decision_gate(state)["next_action"]["arguments"]["message"]
    state = reply(store, state, "email", "I have not found the account address yet.")
    state = apply(store, state, record(case, state))
    assert decision_gate(state)["next_action"]["arguments"]["message"] != first_question
    state = reply(store, state, "email", case["customer"]["email"])
    assert decision_gate(state)["next_action"]["tool"] == "identify_customer"
    assert contact_reply("Do not use " + case["customer"]["email"]) is None


def test_v8_model_projection_has_no_gold_and_does_not_reclassify_initial_permission(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback", index=0)
    state = reads(store, apply(store, state, record(case, state, fallback_return="unknown")))
    visible = json.loads(messages(state)[1]["content"])
    assert visible["workflow_version"] == "v8"
    assert not (
        {"expected", "intent_initial", "intent_complete", "family", "answers", "case_id"} & set(visible)
    )
    assert current_request(state)["fallback_return"] == "unknown"
    assert decision_gate(state)["next_action"]["tool"] == "handoff"
