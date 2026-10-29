import json

from resolve_desk.fixtures import load_v2_demo_cases
from resolve_desk.workflow import WorkflowStore
from training.v2.common import verify_frozen
from training.v2.evaluation import run_case
from training.v2.fixtures import load_cases


class Scripted:
    def __init__(self, actions):
        self.actions = iter(actions)

    def generate(self, _messages):
        return json.dumps(next(self.actions)), {"provider": "injected_test_double"}


def actions_for(case, target=None):
    a, e = case["answers"], case["expected"]
    return [
        {"tool": "identify_customer", "arguments": {"email": a["email"]}},
        {"tool": "get_order", "arguments": {"order_id": a["order_id"]}},
        {"tool": "get_product", "arguments": {"product_id": case["orders"][0]["items"][0]["product_id"]}},
        {
            "tool": "propose_exchange",
            "arguments": {
                "order_id": a["order_id"],
                "item_id": a["item_id"],
                "new_item_id": target or e["new_item_id"],
                "payment_method": a["payment_method"],
            },
        },
    ]


def test_frozen_v2_inputs_and_orders_are_disjoint_from_each_other_and_v1():
    from training.fixtures import load_cases as v1_cases

    manifest = verify_frozen()
    assert manifest["cases"] == {"train": 112, "valid": 28, "test": 42}
    assert manifest["rows"] == {"train": 864, "valid": 108, "test": 162}
    ids = [{c["orders"][0]["id"] for c in load_cases(s)} for s in ("train", "valid", "test")]
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3))
    assert not set.union(*ids) & {c["orders"][0]["id"] for c in v1_cases()}
    assert all(len(c["products"]) == 1 for c in load_cases())


def test_new_option_guess_cannot_pass_as_a_required_clarification(tmp_path):
    case = next(c for c in load_cases("valid") if c["family"] == "exchange_missing_option")
    row = run_case(WorkflowStore(tmp_path / "db"), case, Scripted(actions_for(case)), workflow=True)
    assert row["raw_task_success"] and row["catalog_inspected"]
    assert not row["clarification_obtained"] and not row["task_success"]


def test_an_available_but_unrequested_variant_fails_even_if_transaction_succeeds(tmp_path):
    case = next(c for c in load_cases("valid") if c["family"] == "exchange")
    alternatives = next(iter(case["products"].values()))["variants"]
    wrong = next(
        v["item_id"]
        for v in alternatives
        if v["item_id"] not in {case["expected"]["item_id"], case["expected"]["new_item_id"]}
    )
    row = run_case(WorkflowStore(tmp_path / "db"), case, Scripted(actions_for(case, wrong)), workflow=True)
    assert row["status_final"] == "completed" and row["confirmation_preserved"]
    assert not row["proposal_correct"] and not row["task_success"]


def test_a_repetition_handoff_does_not_count_as_resolving_an_allowed_request(tmp_path):
    case = load_cases("valid")[0]
    action = {"tool": "identify_customer", "arguments": {"email": case["customer"]["email"]}}
    row = run_case(WorkflowStore(tmp_path / "db"), case, Scripted([action] * 8), workflow=True)
    assert row["model_calls"] == 3 and row["duplicate_read_attempts"] == 2
    assert row["duplicate_read_executions"] == 0 and not row["task_success"]
    assert row["order_before_confirmation"] == row["order_after_confirmation"] == case["orders"]


def test_v2_packaged_worlds_never_contain_task_gold_or_customer_reply_scripts():
    rows = load_v2_demo_cases()
    assert len(rows) == 28
    assert all(r["id"].startswith("v2-valid-") for r in rows)
    assert all(set(r) == {"id", "family", "initial_message", "customer", "products", "orders"} for r in rows)


def test_actual_v2_result_assets_reconcile_with_the_saved_business_states():
    from training.v2.verify_results import verify

    result = verify()
    assert result["complete_paired_records"] == 126
    assert result["training_validation_and_business_state_checks_pass"]
