"""CPU protocol fixtures, never inference on the fresh evaluation population."""

import copy
import importlib.util
import json
import math

import pytest

from resolve_desk.resolution_v7 import ResolutionStore, messages
from training.common import ROOT
from training.cuda_eval import evaluation, provider, run
from training.cuda_eval.fixtures import load_cases
from training.cuda_torch.model import lora_config_values
from training.v7.evaluation import run_case


def reference(case_id):
    rows = json.loads((ROOT / "reports/cuda-eval/reference-audit.json").read_text())["records"]
    return next(row["reference"] for row in rows if row["case_id"] == case_id)


class Tokenizer:
    eos_token_id, pad_token_id = 99, 0

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "".join({1: '{"tool":"x"}', 7: "<think>", 99: "<eos>"}[i] for i in ids)


def test_terminal_eos_only_removal_preserves_invalid_special_text_and_empty_output():
    text, info = provider.decode_completion(Tokenizer(), [7, 1, 99], [99])
    assert text == '<think>{"tool":"x"}' and info["output_tokens"] == 3
    assert info["decoded_with_terminal_special_tokens"] == text + "<eos>"
    assert info["generated_token_ids"] == [7, 1, 99] and info["finish_reason"] == "eos"
    assert provider.decode_completion(Tokenizer(), [99], [99])[0] == ""
    assert provider.decode_completion(Tokenizer(), [], [99])[0] == ""
    assert provider.token_hash([1, 99]) != provider.token_hash([1])


def test_generation_settings_explicit_greedy512_kv_cache_and_pinned_stop_ids():
    settings = provider.generation_settings({"eos_token_id": 99, "pad_token_id": 0})
    assert settings["max_new_tokens"] == 512 and not settings["do_sample"] and settings["num_beams"] == 1
    assert settings["use_cache"] and settings["eos_token_id"] == 99 and settings["pad_token_id"] == 0
    assert settings["temperature"] is settings["top_p"] is settings["top_k"] is None
    assert provider.MODEL_KWARGS == {"logits_to_keep": 1}


def test_output_guard_refuses_primary_or_training_report_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    assert evaluation.report_directory(tmp_path / "reports/cuda-eval/new-run") == (
        tmp_path / "reports/cuda-eval/new-run"
    )
    for directory in ("reports/v7", "reports/cuda-torch", "training/v7"):
        with pytest.raises(ValueError, match="reports/cuda-eval"):
            evaluation.report_directory(tmp_path / directory)


class QueueProvider:
    def __init__(self, raw, case):
        self.raw, self.case, self.calls = raw, case, 0

    def generate(self, prompt):
        visible = json.loads(prompt[1]["content"])
        assert not ({"family", "intent_initial", "answers", "expected"} & set(visible))
        assert self.case["id"] not in evaluation.canonical(prompt)
        result = self.raw[self.calls]
        self.calls += 1
        return result, {"provider": "cpu_reference_fixture", "output_tokens": 0}


def test_first_raw_enum_error_retained_before_same_turn_repair(tmp_path):
    case = load_cases("test")[0]
    raws = [o["raw"] for o in reference(case["id"])["trace"]]
    broken = json.loads(raws[0])
    broken["arguments"]["reason"] = "keep the item"
    raws.insert(0, json.dumps(broken))
    journal = evaluation.CallJournal(tmp_path / "calls.json")
    queue = QueueProvider(raws, case)
    wrapped = provider.RecordingProvider(queue, journal, evaluation.ARMS[0], case["id"])
    row = {
        "variant": evaluation.ARMS[0],
        **run_case(ResolutionStore(tmp_path / "world.sqlite"), case, wrapped, messages, max_steps=12),
    }
    assert row["task_success"] and row["model_calls"] == len(raws) == queue.calls
    assert row["initial_enum_errors"] == {"reason": "keep the item"}
    assert not row["intent_field_correct"]["reason"] and row["reason_enum_recovered"]
    assert row["trace"][0]["raw"] == raws[0]
    evaluation.reconcile_journal(journal.entries, [row])


def test_identical_quote_rejections_consume_exact12_calls_not_hidden_retries(tmp_path):
    case = load_cases("test")[0]
    action = json.loads(reference(case["id"])["trace"][0]["raw"])
    action["arguments"]["evidence"] = "not a literal source quotation"
    journal = evaluation.CallJournal(tmp_path / "calls.json")
    queue = QueueProvider([json.dumps(action)] * 12, case)
    wrapped = provider.RecordingProvider(queue, journal, evaluation.ARMS[0], case["id"])
    row = run_case(ResolutionStore(tmp_path / "world.sqlite"), case, wrapped, messages, max_steps=12)
    assert row["model_calls"] == queue.calls == len(journal.entries) == 12
    assert not row["task_success"] and row["identical_action_repeats"] == 11
    assert row["repeated_question_attempts"] == 0
    with pytest.raises(RuntimeError, match="budget"):
        wrapped.generate(messages(ResolutionStore(tmp_path / "another.sqlite").create(case)))
    assert queue.calls == 12 and len(journal.entries) == 12


def test_interrupted_generation_is_durable_and_cannot_be_silently_reissued(tmp_path):
    class Interrupt:
        def generate(self, prompt):
            raise RuntimeError("interrupt after request started")

    journal = evaluation.CallJournal(tmp_path / "calls.json")
    wrapped = provider.RecordingProvider(Interrupt(), journal, evaluation.ARMS[0], "fixture")
    with pytest.raises(RuntimeError, match="interrupt"):
        wrapped.generate([{"role": "user", "content": "fixture"}])
    saved = json.loads(journal.path.read_text())
    assert len(saved) == 1 and saved[0]["status"] == "started" and "raw" not in saved[0]
    with pytest.raises(RuntimeError, match="Orphan"):
        evaluation.reconcile_journal(saved, [])


def test_resume_refuses_pending_case_even_if_no_returned_text(tmp_path):
    evaluation.write_atomic(tmp_path / "analysis-plan.json", {"fixture": "plan"})
    evaluation.write_atomic(tmp_path / "artifact-identity.json", {"fixture": "weights"})
    evaluation.write_atomic(
        tmp_path / "evaluation-partial.json",
        {
            "analysis_plan_sha256": evaluation.sha(tmp_path / "analysis-plan.json"),
            "artifact_identity_sha256": evaluation.sha(tmp_path / "artifact-identity.json"),
            "pending_case": {"variant": evaluation.ARMS[0], "case_id": "fixture"},
        },
    )
    with pytest.raises(RuntimeError, match="refusing to repeat"):
        run.checkpoint_state(tmp_path, {}, {}, resume=True)


def full_adapter_specs():
    shapes = {
        "self_attn.q_proj": (8, 2560, 4096),
        "self_attn.k_proj": (8, 2560, 1024),
        "self_attn.v_proj": (8, 2560, 1024),
        "self_attn.o_proj": (8, 4096, 2560),
        "mlp.gate_proj": (8, 2560, 9728),
        "mlp.up_proj": (8, 2560, 9728),
        "mlp.down_proj": (8, 9728, 2560),
    }
    return {
        f"base_model.model.model.layers.{layer}.{module}.{component}.default.weight": shape
        for layer in range(28, 36)
        for module, (rank, inputs, outputs) in shapes.items()
        for component, shape in [("lora_A", (rank, inputs)), ("lora_B", (outputs, rank))]
    }


def test_final_peft_strict112_shape_dtype_and_namespace_mapping():
    specs = full_adapter_specs()
    header = {
        name.replace(".default.weight", ".weight"): {"dtype": "F32", "shape": shape}
        for name, shape in specs.items()
    }
    mapping = evaluation.peft_mapping(header, specs)
    assert len(mapping) == 112 and sum(math.prod(x["shape"]) for x in mapping) == 3_670_016
    damaged = copy.deepcopy(header)
    damaged[next(iter(damaged))]["dtype"] = "F16"
    with pytest.raises(ValueError, match="mismatch"):
        evaluation.peft_mapping(damaged, specs)
    damaged = copy.deepcopy(header)
    damaged["unexpected.weight"] = {"dtype": "F32", "shape": [1]}
    with pytest.raises(ValueError, match="extra"):
        evaluation.peft_mapping(damaged, specs)


def test_protocol_audit_counts_actual_scoped_yes_and_decline_separately():
    case = next(c for c in load_cases("test") if c["family"] == "exchange_fallback" and c["confirmation"])
    record = reference(case["id"])
    records = [{**copy.deepcopy(record), "variant": arm} for arm in evaluation.ARMS]
    report = evaluation.protocol_report(records, [case])
    for summary in report["summary"].values():
        assert (
            summary["return_proposals"] == summary["committed_returns"] == summary["final_return_orders"] == 1
        )
        assert (
            summary["return_proposals_without_scoped_yes"]
            == summary["committed_returns_without_scoped_yes"]
            == 0
        )
    assert report["records_audited"] == 2 and not report["business_scores_or_release_gate_modified"]


def test_actual_tiny_cpu_peft_generate_uses_same_protocol_without_fresh_cases():
    if any(importlib.util.find_spec(n) is None for n in ("torch", "transformers", "peft")):
        pytest.skip("Optional Torch/Transformers/PEFT unavailable locally")
    import peft
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    class TinyTokenizer:
        eos_token_id = pad_token_id = 0

        def apply_chat_template(self, prompt, **kwargs):
            assert kwargs["enable_thinking"] is False and kwargs["add_generation_prompt"] is True
            return [1, 2, 3]

        def decode(self, ids, **kwargs):
            return "".join("<eos>" if i == 0 else f"T{i}" for i in ids)

    with torch.device("cpu"):
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=128,
            tie_word_embeddings=True,
            eos_token_id=0,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        base = Qwen3ForCausalLM(config)
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.zero_()
        adapted = peft.get_peft_model(base, peft.LoraConfig(**lora_config_values(config.to_dict(), 1)))
        state = peft.get_peft_model_state_dict(adapted)
        specs = {
            name: tuple(parameter.shape)
            for name, parameter in adapted.named_parameters()
            if name.endswith((".lora_A.default.weight", ".lora_B.default.weight"))
        }
        header = {name: {"shape": tuple(value.shape), "dtype": "F32"} for name, value in state.items()}
        mapped = evaluation.peft_mapping(header, specs, expected_tensors=14, expected_parameters=17_408)
        assert {item["target"] for item in mapped} == set(specs)
        head_positions = []
        hook = (
            adapted.get_base_model()
            .get_output_embeddings()
            .register_forward_pre_hook(lambda module, args: head_positions.append(args[0].shape[1]))
        )
        wrapped = provider.TorchProvider(
            adapted, TinyTokenizer(), config.to_dict(), "cpu_fixture", "fixture", "cpu"
        )
        text, timing = wrapped.generate([{"role": "user", "content": "toy protocol fixture"}])
        hook.remove()
        assert text == "" and timing["generated_token_ids"] == [0]
        assert timing["input_token_ids"] == [1, 2, 3] and timing["output_tokens"] == 1
        assert timing["generation_settings"]["max_new_tokens"] == 512
        assert timing["provider"] == "torch_cpu_fixture" and timing["finish_reason"] == "eos"
        assert timing["model_kwargs"] == {"logits_to_keep": 1} and head_positions == [1]
