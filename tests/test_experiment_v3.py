import json
import re
from copy import deepcopy

from resolve_desk.workflow import WorkflowStore
from tests.test_experiment_v2 import Scripted, actions_for
from training.common import ROOT
from training.v3.common import verify_frozen
from training.v3.evaluation import run_case, selection_key
from training.v3.fixtures import load_cases, make_case
from training.v3.tokenization import target_weights


def test_stock_counterfactual_changes_only_availability_and_expected_operation():
    unavailable = make_case("train", "exchange_fallback", 0)
    available = make_case("train", "exchange_fallback_available", 0)
    assert unavailable["initial_message"] == available["initial_message"]
    assert unavailable["orders"] == available["orders"] and unavailable["customer"] == available["customer"]
    products = deepcopy(unavailable["products"])
    for v in next(iter(products.values()))["variants"]:
        if v["item_id"] == unavailable["expected"]["new_item_id"]:
            v["available"] = True
    assert products == available["products"]
    assert unavailable["expected"]["kind"] == "return" and available["expected"]["kind"] == "exchange"


def test_return_negative_control_has_the_same_catalog_but_a_different_customer_intent():
    exchange = make_case("valid", "exchange", 0)
    returned = make_case("valid", "return_reject_exchange", 0)
    assert exchange["orders"] == returned["orders"] and exchange["products"] == returned["products"]
    assert exchange["initial_message"] != returned["initial_message"]
    assert "Do not exchange" in returned["initial_message"]


def test_loss_weights_mask_prompt_and_only_upweight_tool_name_overlap():
    offsets = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]
    weights = target_weights(offsets, 3, 6, 10, 8.0)
    assert weights == [0.0, 0.0, 8.0, 8.0, 1.0]


def test_business_selection_rejects_the_lower_loss_all_return_candidate():
    correct = {"fulfillable_exchange_success": 4, "task_success": 10, "exchange_as_return": 0}
    all_returns = {"fulfillable_exchange_success": 0, "task_success": 14, "exchange_as_return": 5}
    assert selection_key(correct, 0.2) > selection_key(all_returns, 0.001)


def test_valid_wrong_return_counts_as_exchange_misclassification(tmp_path):
    case = make_case("valid", "exchange", 0)
    actions = actions_for(case)
    actions[-1] = {
        "tool": "propose_return",
        "arguments": {
            "order_id": case["answers"]["order_id"],
            "item_id": case["answers"]["item_id"],
            "payment_method": "original",
        },
    }
    row = run_case(WorkflowStore(tmp_path / "db"), case, Scripted(actions))
    assert row["schema_valid"] == 4 and row["rejected_write_attempts"] == 0
    assert row["exchange_as_return"] and row["unauthorized_return"] and not row["task_success"]


def test_authorized_unavailable_fallback_is_not_a_false_return(tmp_path):
    case = make_case("valid", "exchange_fallback", 0)
    actions = actions_for(case)
    actions[-1] = {
        "tool": "propose_return",
        "arguments": {
            "order_id": case["answers"]["order_id"],
            "item_id": case["answers"]["item_id"],
            "payment_method": "original",
        },
    }
    row = run_case(WorkflowStore(tmp_path / "db"), case, Scripted(actions))
    assert row["task_success"] and not row["exchange_as_return"] and not row["unauthorized_return"]


def test_v3_frozen_worlds_are_disjoint_across_splits_and_old_experiments():
    verify_frozen()

    def generated_ids(cases):
        # Shared policy literals such as "other_demo_customer" are not generated entities.
        return set(
            re.findall(
                r"(?:ORD-|u_|p_|item_|gift_|card_)[A-Z2-9]{6}|buyer[a-z0-9]{6}@retail\.example",
                json.dumps(cases),
            )
        )

    worlds = {
        split: {o["id"] for c in load_cases(split) for o in c["orders"]}
        for split in ("train", "valid", "test")
    }
    assert [len(worlds[s]) for s in worlds] == [112, 14, 42]
    assert all(
        not worlds[a] & worlds[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]
    )
    entities = {s: generated_ids(load_cases(s)) for s in worlds}
    assert all(
        not entities[a] & entities[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]
    )
    for folder in (ROOT / "training/datasets", ROOT / "training/v2/datasets"):
        old_cases = json.loads((folder / "cases.json").read_text())
        old = {o["id"] for c in old_cases for o in c["orders"]}
        assert not old & set.union(*worlds.values())
        assert not generated_ids(old_cases) & set.union(*entities.values())


def test_packaged_validation_worlds_exclude_all_grading_and_simulator_metadata():
    cases = json.loads((ROOT / "resolve_desk/fixtures/validation-v3.json").read_text())
    assert len(cases) == 16
    assert all(set(c) == {"id", "family", "initial_message", "customer", "products", "orders"} for c in cases)


def test_delivered_v3_results_reconcile_with_raw_traces():
    from training.v3.verify_results import verify

    assert verify()["paired_records"] == 144
