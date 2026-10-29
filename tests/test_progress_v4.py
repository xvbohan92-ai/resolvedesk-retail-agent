import json
from collections import Counter

from resolve_desk.progress import ProgressStore, clarification_memory, messages, observed_constraints
from tests.test_experiment_v2 import Scripted
from training.v4.common import verify_frozen
from training.v4.evaluation import run_case, selection_key
from training.v4.fixtures import load_cases, make_case
from training.v4.prepare import apply, prefix


def test_answered_duplicate_is_blocked_but_attempt_remains_visible(tmp_path):
    store = ProgressStore(tmp_path / "db")
    case = make_case("valid", "missing_payment", 0)
    state, _, _, _ = prefix(store, case)
    memory = clarification_memory(state)
    assert memory[0]["field"] == "payment_method" and memory[0]["reply"] == "original"
    question = next(o["action"] for o in state["observations"] if o["action"]["tool"] == "ask_customer")
    state = apply(store, state, question)
    assert state["status"] == "running"
    assert state["observations"][-1]["result"]["error"] == "duplicate_clarification_blocked"
    assert store.orders(state["id"]) == case["orders"]


def test_new_unsolicited_request_clears_old_clarification_context(tmp_path):
    store = ProgressStore(tmp_path / "db")
    state, _, _, _ = prefix(store, make_case("valid", "missing_payment", 0))
    assert clarification_memory(state)
    state = store.customer_message(
        state["id"], "Actually I want to discuss a different item.", state["revision"]
    )
    assert clarification_memory(state) == []


def test_specific_followup_can_clarify_an_unusable_answer(tmp_path):
    store = ProgressStore(tmp_path / "db")
    state = store.create(make_case("valid", "return", 0))
    state = apply(
        store, state, {"tool": "ask_customer", "arguments": {"field": "email", "message": "Your email?"}}
    )
    state = store.customer_message(state["id"], "I do not understand", state["revision"])
    assert clarification_memory(state)[0]["reply"] == "I do not understand"
    state = apply(
        store,
        state,
        {
            "tool": "ask_customer",
            "arguments": {"field": "email", "message": "What email did you use for this order?"},
        },
    )
    assert state["status"] == "awaiting_customer"


def test_policy_observations_exist_only_after_actual_tool_results(tmp_path):
    store = ProgressStore(tmp_path / "db")
    case = make_case("valid", "foreign_order", 0)
    state = store.create(case)
    assert observed_constraints(state) == []
    state = apply(
        store, state, {"tool": "identify_customer", "arguments": {"email": case["answers"]["email"]}}
    )
    assert observed_constraints(state) == []
    state = apply(store, state, {"tool": "get_order", "arguments": {"order_id": case["answers"]["order_id"]}})
    assert observed_constraints(state) == [{"order_id": case["answers"]["order_id"], "access_denied": True}]
    visible = json.loads(messages(state)[1]["content"])
    assert set(visible) == {
        "messages",
        "completed_reads",
        "recent_events",
        "allowed_tools",
        "clarification_memory",
        "observed_constraints",
    }
    assert case["customer"]["id"] not in json.dumps(visible["observed_constraints"])


def test_repeated_model_attempt_is_not_hidden_by_delivery_guard(tmp_path):
    case = make_case("valid", "foreign_order", 0)
    ask = {"tool": "ask_customer", "arguments": {"field": "payment_method", "message": "Your payment?"}}
    actions = [
        {"tool": "identify_customer", "arguments": {"email": case["answers"]["email"]}},
        {"tool": "get_order", "arguments": {"order_id": case["answers"]["order_id"]}},
        ask,
        ask,
        {"tool": "handoff", "arguments": {"reason": "Order access denied."}},
    ]
    row = run_case(ProgressStore(tmp_path / "db"), case, Scripted(actions))
    assert row["task_success"] and row["model_handoff"]
    assert row["repeated_question_attempts"] == 1 and row["repeated_questions_delivered"] == 0


def test_selection_balances_exchange_fallback_and_handoff_instead_of_one_goal():
    summary = {
        "exchange_as_return": 0,
        "fulfillable_exchange_tasks": 10,
        "fallback_tasks": 2,
        "handoff_tasks": 8,
        "fulfillable_exchange_success": 10,
        "fallback_success": 0,
        "handoff_success": 0,
        "task_success": 20,
        "repeated_question_attempts": 0,
    }
    balanced = {
        **summary,
        "fulfillable_exchange_success": 8,
        "fallback_success": 2,
        "handoff_success": 6,
        "task_success": 24,
    }
    assert selection_key(balanced, 0.2) > selection_key(summary, 0.001)


def test_new_frozen_cases_are_disjoint_and_test_confirmation_is_balanced():
    manifest = verify_frozen()
    assert manifest["cases"] == {"train": 64, "valid": 32, "test": 32}
    assert Counter(c["confirmation"] for c in load_cases("test")) == {True: 16, False: 16}
    worlds = {s: {o["id"] for c in load_cases(s) for o in c["orders"]} for s in ("train", "valid", "test")}
    assert [len(x) for x in worlds.values()] == [56, 28, 28]
    assert all(
        not worlds[a] & worlds[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]
    )
    from training.common import ROOT

    def ids(cases):
        found = set()

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if (
                        key in {"id", "user_id", "product_id", "item_id", "original_payment", "email"}
                        and isinstance(item, str)
                        and item != "other_demo_customer"
                    ):
                        found.add(item)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        for case in cases:
            walk([case["customer"], case["orders"], case["products"]])
        return found

    entities = {s: ids(load_cases(s)) for s in worlds}
    assert all(
        not entities[a] & entities[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]
    )

    for folder in ["training/datasets", "training/v2/datasets", "training/v3/datasets"]:
        old = json.loads((ROOT / folder / "cases.json").read_text())
        assert not {o["id"] for c in old for o in c["orders"]} & set.union(*worlds.values())
        assert not ids(old) & set.union(*entities.values())


def test_packaged_v4_worlds_exclude_simulator_answers_and_gold():
    from training.common import ROOT

    cases = json.loads((ROOT / "resolve_desk/fixtures/validation-v4.json").read_text())
    assert len(cases) == 32
    assert all(set(c) == {"id", "family", "initial_message", "customer", "orders", "products"} for c in cases)


def test_delivered_v4_results_reconcile():
    from training.v4.verify_results import verify

    assert verify()["records_reconciled"] == 128


def test_preference_pairs_are_train_only_with_identical_visible_context():
    from training.v4.preference import PAIR_PATH, verify_protocol

    manifest = verify_protocol()
    pairs = [json.loads(line) for line in PAIR_PATH.read_text().splitlines()]
    assert len(pairs) == manifest["pair_count"] == 124
    assert {p["case_id"] for p in pairs} <= {c["id"] for c in load_cases("train")}
    assert Counter(p["category"] for p in pairs) == {
        "avoid_unnecessary_handoff": 48,
        "ask_missing_information": 20,
        "continue_after_answer": 20,
        "preserve_exchange_intent": 20,
        "preserve_policy_handoff": 16,
    }
    for pair in pairs:
        a, b = pair["chosen"]["messages"], pair["rejected"]["messages"]
        assert a[:-1] == b[:-1] and a[-1] != b[-1]
        visible = json.loads(a[1]["content"])
        assert not {"answers", "expected", "confirmation", "required_clarification"} & visible.keys()
        for side in (a, b):
            assert set(json.loads(side[-1]["content"])) == {"tool", "arguments"}


def test_action_preferences_add_genuine_return_controls_without_test_worlds():
    from training.v4.action_preference import PAIR_PATH, verify_protocol

    manifest = verify_protocol()
    pairs = [json.loads(line) for line in PAIR_PATH.read_text().splitlines()]
    assert len(pairs) == manifest["pair_count"] == 136
    training = {c["id"]: c for c in load_cases("train")}
    assert {p["case_id"] for p in pairs} <= training.keys()
    added = [p for p in pairs if p["category"] == "preserve_genuine_return"]
    assert len(added) == 12
    for pair in pairs:
        a, b = pair["chosen"]["messages"], pair["rejected"]["messages"]
        assert a[:-1] == b[:-1] and a[-1] != b[-1]
        assert json.loads(a[-1]["content"])["tool"] != json.loads(b[-1]["content"])["tool"]
    for pair in added:
        assert training[pair["case_id"]]["expected"]["kind"] == "return"
        assert json.loads(pair["chosen"]["messages"][-1]["content"])["tool"] == "propose_return"
        assert json.loads(pair["rejected"]["messages"][-1]["content"])["tool"] == "propose_exchange"


def test_decision_facts_use_only_completed_tools_and_keep_all_candidates(tmp_path):
    from resolve_desk.decision_facts import augment_messages

    store = ProgressStore(tmp_path / "db")
    case = make_case("valid", "exchange_shortfall", 0)
    initial = store.create(case)
    assert json.loads(augment_messages(messages(initial))[1]["content"])["decision_facts"] == []
    state, _, _, _ = prefix(store, case)
    visible = json.loads(augment_messages(messages(state))[1]["content"])
    catalog = next(o["result"] for o in state["observations"] if o["action"]["tool"] == "get_product")
    card = state["observations"][0]["result"]["gift_card"]
    assert len(visible["decision_facts"]) == len(catalog["exchange_candidates"])
    for fact, candidate in zip(visible["decision_facts"], catalog["exchange_candidates"], strict=True):
        assert fact["candidate_item_id"] == candidate["item_id"]
        assert fact["available"] == candidate["available"]
        assert fact["gift_card_coverage"][0]["covers_difference"] == (
            candidate["price_difference_cents"] <= card["balance_cents"]
        )
    assert not {"expected", "answers", "selected_candidate"} & visible.keys()


def test_recovery_data_does_not_insert_missing_answers_after_reads():
    from training.v4.recovery import PAIR_PATH, verify_protocol

    manifest = verify_protocol()
    pairs = [json.loads(line) for line in PAIR_PATH.read_text().splitlines()]
    training = {c["id"]: c for c in load_cases("train")}
    assert len(pairs) == manifest["pair_count"] == 80
    assert len({p["case_id"] for p in pairs}) == manifest["source_cases"] == 52
    for pair in pairs:
        assert pair["case_id"] in training
        a, b = pair["chosen"]["messages"], pair["rejected"]["messages"]
        assert a[:-1] == b[:-1]
        visible = json.loads(a[1]["content"])
        assert "decision_facts" in visible
        if pair["category"] == "clarify_after_reads_instead_of_guessing":
            assert visible["messages"] == [
                {"role": "customer", "content": training[pair["case_id"]]["initial_message"]}
            ]
            assert not visible["clarification_memory"]
            assert json.loads(a[-1]["content"])["tool"] == "ask_customer"


def test_calibration_guard_rejects_fallback_gain_that_loses_handoff_or_clarification():
    from training.v4.calibration_choice import prefer_calibration

    baseline = {
        "exchange_as_return": 0,
        "task_success": 24,
        "fulfillable_exchange_success": 4,
        "handoff_success": 8,
        "clarifications_obtained": 6,
        "direct_return_success": 8,
        "repeated_question_attempts": 0,
        "fallback_success": 0,
        "model_calls": 129,
    }
    better = {**baseline, "task_success": 28, "fallback_success": 2, "fulfillable_exchange_success": 6}
    assert prefer_calibration(better, baseline)
    assert not prefer_calibration({**better, "handoff_success": 7}, baseline)
    assert not prefer_calibration({**better, "clarifications_obtained": 5}, baseline)
    assert not prefer_calibration({**better, "exchange_as_return": 1}, baseline)


def test_calibration_stock_and_authorization_worlds_are_counterfactual_and_balanced():
    from training.v4.calibration import PAIR_PATH, verify_protocol

    assert verify_protocol()["pair_count"] == 24
    pairs = [json.loads(line) for line in PAIR_PATH.read_text().splitlines()]
    training = {c["id"]: c for c in load_cases("train")}
    assert {p["source_case_id"] for p in pairs} <= training.keys()
    actions = Counter(json.loads(p["chosen"]["messages"][-1]["content"])["tool"] for p in pairs)
    assert actions == {"propose_return": 8, "propose_exchange": 8, "handoff": 8}
    for pair in pairs:
        assert pair["chosen"]["messages"][:-1] == pair["rejected"]["messages"][:-1]
        if pair["contrast_field"]:
            a = json.loads(pair["chosen"]["messages"][-1]["content"])
            b = json.loads(pair["rejected"]["messages"][-1]["content"])
            assert a["tool"] == b["tool"] == "propose_exchange"
            assert a["arguments"]["payment_method"] != b["arguments"]["payment_method"]
