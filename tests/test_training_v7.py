"""CPU contracts for loss masks, sampling and honest pre-repair model metrics."""

import json
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from resolve_desk.resolution_v6_1 import ResolutionStore as PreviousStore
from resolve_desk.resolution_v6_1 import messages as previous_messages
from resolve_desk.resolution_v7 import ResolutionStore, messages
from training.v7.evaluation import ARMS, run_case
from training.v7.fixtures import make_case
from training.v7.tokenization import encode_row
from training.v7.train import balanced_sample_indices, projection_batch, sampling_buckets


class OffsetTokenizer:
    """A tiny tokenizer with exact offsets and optional boundary-crossing tokens."""

    def __init__(self, chunk=1):
        self.chunk = chunk

    def tokenize(self, text):
        boundary = text.rfind("<assistant>\n") + len("<assistant>\n")
        spans = [(i, i + 1) for i in range(boundary)]
        spans += [(i, min(i + self.chunk, len(text))) for i in range(boundary, len(text), self.chunk)]
        ids = [sum((ord(c) + 1) * 1114113**n for n, c in enumerate(text[a:b])) for a, b in spans]
        return {"input_ids": ids, "offset_mapping": spans}

    def __call__(self, text, **kwargs):
        return self.tokenize(text)

    def apply_chat_template(self, conversation, *, tokenize=True, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>\n{m['content']}\n" for m in conversation)
        if add_generation_prompt:
            text += "<assistant>\n"
        return self.tokenize(text)["input_ids"] if tokenize else text


@pytest.mark.parametrize("chunk", [1, 3])
def test_value_offsets_weight_only_final_assistant_fields_and_keep_prompt_zero(chunk):
    content = json.dumps(
        {
            "tool": "record_request",
            "arguments": {
                "operation": "exchange",
                "email": "a@example.com",
                "order_id": "ORD-1",
                "requested_option": "blue / M",
                "payment_method": "original",
                "reason": "unknown",
                "fallback_return": "yes",
                "evidence": 'Quoted "reason":"unknown" is not the field.',
            },
        },
        separators=(",", ":"),
    )
    row = {
        "messages": [
            {"role": "system", "content": 'Do not supervise prompt "fallback_return":"yes".'},
            {"role": "user", "content": content},
            {"role": "assistant", "content": content},
        ]
    }
    tokenizer = OffsetTokenizer(chunk)
    actual = encode_row(row, tokenizer, 8, 4)
    rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False)
    offsets = tokenizer(rendered)["offset_mapping"]
    completion_start = rendered.rfind(content)
    ranges = {}
    for key, value in [("reason", "unknown"), ("fallback_return", "yes")]:
        start = completion_start + content.index(f'"{key}":"') + len(f'"{key}":"')
        ranges[key] = (start, start + len(value))
    expected = []
    crossings = []
    for i, (a, b) in enumerate(offsets[1:], 1):
        value = 0 if i < actual["prompt_length"] else 1
        for field, multiplier in [("fallback_return", 8), ("reason", 4)]:
            left, right = ranges[field]
            if value and a < right and b > left:
                value = multiplier
                crossings.append(a < left or b > right)
        expected.append(value)
    assert actual["weights"] == expected
    assert actual["permission_tokens"] == expected.count(8)
    assert actual["reason_tokens"] == expected.count(4)
    assert not any(actual["weights"][: actual["prompt_length"] - 1])
    if chunk > 1:
        assert any(crossings), "Exercise tokens that overlap a value and its punctuation"


def test_other_tools_do_not_receive_record_field_value_weights():
    row = {
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "a request"},
            {"role": "assistant", "content": '{"tool":"handoff","arguments":{"reason":"unknown"}}'},
        ]
    }
    actual = encode_row(row, OffsetTokenizer(), 8, 4)
    assert actual["permission_tokens"] == actual["reason_tokens"] == 0
    assert set(actual["weights"]) == {0, 1}


def test_tail_projection_retains_full_context_correct_next_tokens_and_zero_padding():
    example = {"tokens": list(range(1200)), "weights": [0.0] * 1099 + [4.0] * 100}
    tokens, weights = projection_batch(example, 512, 3072)
    assert tokens[:1200] == example["tokens"]
    assert len(tokens) - 1 == 1216 and len(weights) == 512
    targets = [token for token, weight in zip(tokens[-len(weights) :], weights, strict=True) if weight]
    assert targets == list(range(1100, 1200))
    assert weights[-17:] == [0] * 17


def test_projection_refuses_to_drop_a_completion_target():
    example = {"tokens": list(range(1200)), "weights": [0.0] * 649 + [1.0] * 550}
    with pytest.raises(AssertionError, match="supervised completion token"):
        projection_batch(example, 512, 3072)


def test_semantic_sampling_is_balanced_reproducible_and_without_replacement_per_cycle():
    rows = [{"origin": "record_request"}] * 3 + [{"origin": "get_order"}] * 7
    schedule = balanced_sample_indices(rows, seed=43, iterations=14)
    assert schedule == balanced_sample_indices(rows, seed=43, iterations=14)
    assert Counter(bucket for bucket, _ in schedule) == {
        "intent_and_record_repair": 7,
        "execution_and_reply": 7,
    }
    pools = sampling_buckets(rows)
    for bucket, pool in pools.items():
        actual = [i for b, i in schedule if b == bucket]
        for start in range(0, len(actual), len(pool)):
            cycle = actual[start : start + len(pool)]
            assert len(cycle) == len(set(cycle))
            assert set(cycle) <= set(pool)
    assert {i for _, i in schedule} == set(range(len(rows)))


def test_formal_schedule_covers_every_physical_training_row():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "training/v7/config.json").read_text())
    rows = [json.loads(s) for s in (root / "training/v7/datasets/train.jsonl").read_text().splitlines()]
    schedule = balanced_sample_indices(rows, config["seed"], config["iters"])
    assert config["iters"] % 2 == 0
    assert Counter(b for b, _ in schedule) == {k: config["iters"] // 2 for k in sampling_buckets(rows)}
    assert {i for _, i in schedule} == set(range(len(rows)))


class ReferenceModel:
    def __init__(self, case, first):
        self.case, self.first, self.calls = case, first, 0

    def generate(self, prompt):
        visible = json.loads(prompt[1]["content"])
        self.calls += 1
        if self.calls == 1:
            action = self.first
        else:
            action = visible["decision_gate"]["next_action"]
            if action["tool"] == "record_request":
                intent = dict(self.case["intent_initial"])
                for pair in visible["clarification_memory"]:
                    key = "requested_option" if pair["field"] == "new_option" else pair["field"]
                    if key != "fallback_return":
                        intent[key] = self.case["intent_complete"][key]
                action = {
                    "tool": "record_request",
                    "arguments": {**intent, "evidence": visible["messages"][-1]["content"]},
                }
        return json.dumps(action), {"provider": "fake", "output_tokens": 1}


def bad_initial(case):
    return {
        "tool": "record_request",
        "arguments": {
            **case["intent_initial"],
            "reason": "Do not refund this item",
            "fallback_return": "yes",
            "evidence": case["evidence_initial"],
        },
    }


@pytest.mark.parametrize("arm", ARMS)
def test_all_arms_score_first_raw_prediction_before_enum_repair(tmp_path, arm):
    case = make_case("valid", "exchange_unavailable", 0)
    previous = arm == "previous_old_flow"
    store = (PreviousStore if previous else ResolutionStore)(tmp_path / "eval.sqlite")
    model = ReferenceModel(case, bad_initial(case))
    result = run_case(store, case, model, previous_messages if previous else messages)
    assert result["intent_initial"]["reason"] == "Do not refund this item"
    assert result["intent_initial"]["fallback_return"] == "yes"
    assert result["permission_yes"] and not result["permission_authorized"]
    assert not result["intent_field_correct"]["reason"]
    assert not result["intent_field_correct"]["fallback_return"]
    assert not result["intent_enum_valid"] and not result["initial_record_valid"]
    assert result["initial_enum_errors"] == {"reason": "Do not refund this item"}
    assert result["enum_error_attempts"] == result["reason_enum_errors"] == 1
    assert result["model_calls"] == model.calls
    assert result["enum_recovered"] is (not previous)
    assert result["reason_enum_recovered"] is (not previous)
    assert result["task_success"] is (not previous)
    assert result["schema_valid"] == model.calls - 1


def test_first_non_record_output_is_not_replaced_by_later_good_intent(tmp_path):
    case = make_case("valid", "exchange_unavailable", 0)
    store = ResolutionStore(tmp_path / "eval.sqlite")
    first = {"tool": "get_order", "arguments": {"order_id": case["intent_initial"]["order_id"]}}
    result = run_case(store, case, ReferenceModel(case, first), messages)
    assert result["task_success"]
    assert result["intent_initial"] is None and result["initial_raw_parse_failure"]
    assert not any(result["intent_field_correct"].values())


def test_customer_turn_boundary_prevents_crediting_later_correction_as_same_turn_repair(tmp_path):
    class NewTurnStore(ResolutionStore):
        def apply(self, sid, raw, timing, revision):
            state = super().apply(sid, raw, timing, revision)
            if len(state["observations"]) == 1:
                state = self.customer_message(
                    sid, "Please use the original request above.", state["revision"]
                )
            return state

    case = make_case("valid", "exchange_unavailable", 0)
    result = run_case(
        NewTurnStore(tmp_path / "eval.sqlite"), case, ReferenceModel(case, bad_initial(case)), messages
    )
    assert result["task_success"] and result["enum_error_attempts"] == 1
    assert not result["enum_recovered"] and not result["reason_enum_recovered"]


def test_four_arm_loader_uses_selected_predecessor_path_not_old_optimizer_path(tmp_path, monkeypatch):
    from training.v7 import evaluate

    reports = tmp_path / "reports/v7"
    reports.mkdir(parents=True)
    predecessor = tmp_path / "reports/v6_1/training.json"
    predecessor.parent.mkdir(parents=True)
    predecessor.write_text(
        json.dumps(
            {
                "config": {"best_path": "wrong-stale-training-path"},
                "evaluation_config": {"best_path": "selected-predecessor"},
                "selected_adapter_sha256": "same",
            }
        )
    )
    (reports / "training.json").write_text(json.dumps({"dataset": {}, "selected_adapter_sha256": "same"}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "base", "best_path": "selected-v7", "max_agent_steps": 12}))
    loaded = []
    core = types.ModuleType("mlx.core")
    core.set_wired_limit = core.set_cache_limit = lambda value: None
    core.clear_cache = lambda: None
    mlx = types.ModuleType("mlx")
    mlx.core = core
    lm = types.ModuleType("mlx_lm")

    def load(path, *, adapter_path, trust_remote_code):
        assert trust_remote_code is False
        loaded.append(adapter_path)
        return object(), object()

    lm.load = load
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx_lm", lm)
    for key, value in {"ROOT": tmp_path, "REPORTS": reports, "DATA": tmp_path, "CONFIG": config}.items():
        monkeypatch.setattr(evaluate, key, value)
    monkeypatch.setattr(evaluate, "verify_frozen", lambda: {})
    monkeypatch.setattr(evaluate, "verify_local_base", lambda config: None)
    monkeypatch.setattr(evaluate, "sha", lambda path: "same")
    monkeypatch.setattr(evaluate, "wait_for_room", lambda: None)
    monkeypatch.setattr(evaluate, "load_cases", lambda split: [{"id": "test-one"}])
    monkeypatch.setattr(evaluate, "Provider", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        evaluate,
        "run_case",
        lambda store, case, *args: {
            "case_id": case["id"],
            "task_success": True,
            "repeated_question_attempts": 0,
        },
    )
    monkeypatch.setattr(evaluate, "summarize", lambda rows: {"tasks": len(rows)})
    monkeypatch.setattr(evaluate, "promotion", lambda rows: {"promoted": False})
    evaluate.main()
    assert loaded == [
        tmp_path / "selected-predecessor",
        tmp_path / "selected-predecessor",
        None,
        tmp_path / "selected-v7",
    ]
