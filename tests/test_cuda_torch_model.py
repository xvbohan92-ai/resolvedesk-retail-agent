"""Small CPU fixtures only: no MLX import, real model construction, CUDA or downloads."""

import importlib.util
import json
import math
import struct

import numpy as np
import pytest

from training.cuda_torch import model

CONFIG = {
    "num_hidden_layers": 36,
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
}


def adapter_header():
    result = {}
    dimensions = {
        "self_attn.q_proj": (2560, 4096),
        "self_attn.k_proj": (2560, 1024),
        "self_attn.v_proj": (2560, 1024),
        "self_attn.o_proj": (4096, 2560),
        "mlp.gate_proj": (2560, 9728),
        "mlp.up_proj": (2560, 9728),
        "mlp.down_proj": (9728, 2560),
    }
    for layer in range(28, 36):
        for name, (inputs, outputs) in dimensions.items():
            stem = f"model.layers.{layer}.{name}"
            result[stem + ".lora_a"] = {"shape": [inputs, 8], "dtype": "F32"}
            result[stem + ".lora_b"] = {"shape": [8, outputs], "dtype": "F32"}
    return result


def small_base():
    shapes = {
        "model.embed_tokens.weight": (3, 128),
        "model.layers.0.self_attn.q_norm.weight": (128,),
        "model.layers.0.self_attn.k_norm.weight": (128,),
        "lm_head.weight": (3, 128),
    }
    header = {
        "model.embed_tokens.weight": {"shape": [3, 16], "dtype": "U32"},
        "model.embed_tokens.scales": {"shape": [3, 1], "dtype": "BF16"},
        "model.embed_tokens.biases": {"shape": [3, 1], "dtype": "BF16"},
        "model.layers.0.self_attn.q_norm.weight": {"shape": [128], "dtype": "BF16"},
        "model.layers.0.self_attn.k_norm.weight": {"shape": [128], "dtype": "BF16"},
    }
    return header, shapes


def test_decode_u32_little_endian_nibbles_low_first():
    # Physical bytes encode 0..15 twice per two words; repeat to one 128-wide group.
    packed = np.frombuffer(bytes.fromhex("1032547698badcfe") * 8, dtype="<u4").reshape(1, 16)
    fp16, logical = model.decode_affine4bit_numpy(packed, np.ones((1, 1)), np.zeros((1, 1)))
    np.testing.assert_array_equal(logical[0], np.tile(np.arange(16), 8))
    np.testing.assert_array_equal(fp16.astype(np.float32), logical)
    assert fp16.dtype == np.float16


def test_decode_group128_affine_boundaries_and_bf16_rounding_order():
    words = np.full((1, 32), np.uint32(0x11111111))
    fp16, logical = model.decode_affine4bit_numpy(
        words, np.array([[1 / 256, 1 / 128]], dtype=np.float32), np.array([[1.0, -2.0]], dtype=np.float32)
    )
    # 1+2^-8 lies exactly halfway between BF16 values and rounds to even 1.
    assert logical[0, 127] == 1.0
    assert logical[0, 128] == -1.9921875
    assert fp16[0, 127] == 1.0
    rounded = model.round_bf16_numpy(np.array([1 + 1 / 256, 1 + 3 / 256], dtype=np.float32))
    np.testing.assert_array_equal(rounded, np.array([1.0, 1.015625], dtype=np.float32))


def test_decode_rounds_bf16_product_before_adding_bias_not_f32_once():
    packed = np.full((1, 16), np.uint32(0x33333333))
    scale = np.float32(0.10009765625)  # Exactly represented BF16, unlike decimal0.1.
    bias = np.float32(-0.25)
    fp16, logical = model.decode_affine4bit_numpy(packed, np.array([[scale]]), np.array([[bias]]))
    # 3*scale = .30029296875 -> BF16 .30078125, then add -.25 -> .05078125.
    np.testing.assert_array_equal(logical, np.full((1, 128), np.float32(0.05078125)))
    np.testing.assert_array_equal(fp16.astype(np.float32), logical)
    f32_once = model.round_bf16_numpy(np.array([[3 * scale + bias]], dtype=np.float32))
    assert f32_once[0, 0] == np.float32(0.05029296875)
    assert abs(logical[0, 0] - f32_once[0, 0]) == 0.00048828125


@pytest.mark.parametrize("mutate", ["signed", "bad_scale", "bad_bias", "bad_group"])
def test_decode_rejects_incompatible_packing_and_groups(mutate):
    packed, scales, biases, group = np.zeros((1, 16), dtype=np.uint32), np.ones((1, 1)), np.zeros((1, 1)), 128
    if mutate == "signed":
        packed = packed.astype(np.int32)
    elif mutate == "bad_scale":
        scales = np.ones((1, 2))
    elif mutate == "bad_bias":
        biases = np.zeros((2, 1))
    else:
        group = 127
    with pytest.raises(ValueError):
        model.decode_affine4bit_numpy(packed, scales, biases, group)


def test_strict_base_mapping_retains_qk_norm_and_omits_only_tied_head():
    header, shapes = small_base()
    plan = model.base_plan(header, shapes)
    assert [p["name"] for p in plan] == list(shapes)[:-1]
    assert plan[0]["kind"] == "affine4bit" and len(plan[0]["source_keys"]) == 3
    assert all(p["kind"] == "bf16" for p in plan[1:])


@pytest.mark.parametrize("corruption", ["missing_qnorm", "extra_head", "wrong_bias", "bad_tie"])
def test_strict_base_mapping_rejects_missing_extra_or_wrong_shapes(corruption):
    header, shapes = small_base()
    if corruption == "missing_qnorm":
        header.pop("model.layers.0.self_attn.q_norm.weight")
    elif corruption == "extra_head":
        header["lm_head.weight"] = {"shape": [3, 128], "dtype": "BF16"}
    elif corruption == "wrong_bias":
        header["model.embed_tokens.biases"]["shape"] = [3, 2]
    else:
        shapes["lm_head.weight"] = (4, 128)
    with pytest.raises(ValueError):
        model.base_plan(header, shapes)


def test_lora_targets_exactly_last8_layers_and_mlx_scale_not_alpha16():
    values = model.lora_config_values(CONFIG)
    targets = values["target_modules"]
    assert len(targets) == 56 and len(set(targets)) == 56
    assert {int(name.split(".")[2]) for name in targets} == set(range(28, 36))
    assert all(
        sum(name.startswith(f"model.layers.{layer}.") for name in targets) == 7 for layer in range(28, 36)
    )
    assert values["lora_alpha"] == 128 and values["lora_alpha"] / values["r"] == 16
    assert values["bias"] == "none" and not values["use_rslora"]


def test_initializer_plan_has_exact112_shapes_transposes_and_3670016_parameters():
    plan = model.adapter_plan(adapter_header(), CONFIG)
    assert len(plan) == 112
    assert sum(math.prod(p["source_shape"]) for p in plan) == 3_670_016
    for item in plan:
        assert item["target_shape"] == item["source_shape"][::-1]
        assert item["target"].endswith(f"{item['peft_component']}.default.weight")
        assert item["target"].startswith("base_model.model.model.layers.")


@pytest.mark.parametrize("corruption", ["missing", "extra", "layout", "dtype"])
def test_initializer_rejects_missing_extra_wrong_layout_or_dtype(corruption):
    header = adapter_header()
    key = "model.layers.28.self_attn.q_proj.lora_a"
    if corruption == "missing":
        header.pop(key)
    elif corruption == "extra":
        header["model.layers.27.self_attn.q_proj.lora_a"] = {"shape": [2560, 8], "dtype": "F32"}
    elif corruption == "layout":
        header[key]["shape"] = [8, 2560]
    else:
        header[key]["dtype"] = "F16"
    with pytest.raises(ValueError):
        model.adapter_plan(header, CONFIG)


def test_header_reads_only_metadata_and_ignores_metadata_block(tmp_path):
    path = tmp_path / "tiny.safetensors"
    data = json.dumps(
        {"__metadata__": {"kind": "fixture"}, "weight": {"shape": [1], "dtype": "F32"}}
    ).encode()
    path.write_bytes(struct.pack("<Q", len(data)) + data + b"not model values")
    assert model.read_header(path) == {"weight": {"shape": [1], "dtype": "F32"}}
    path.write_bytes(struct.pack("<Q", 17 * 1024**2))
    with pytest.raises(ValueError, match="header size"):
        model.read_header(path)


def test_model_loader_has_no_cpu_fallback_before_framework_import():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        model.load_model("not accessed", "not accessed", device="cpu")


def test_torch_cpu_decoder_matches_numpy_and_records_tiny_underflow():
    torch = pytest.importorskip("torch")
    packed = torch.full((1, 16), 0x11111111, dtype=torch.uint32)
    scales = torch.tensor([[2**-30]], dtype=torch.bfloat16)
    biases = torch.zeros((1, 1), dtype=torch.bfloat16)
    fp16, logical = model.decode_affine4bit_torch(packed, scales, biases)
    expected, expected_logical = model.decode_affine4bit_numpy(
        packed.numpy(), scales.float().numpy(), biases.float().numpy()
    )
    np.testing.assert_array_equal(fp16.numpy(), expected)
    np.testing.assert_array_equal(logical.float().numpy(), expected_logical)
    audit = model.cast_stats(logical, fp16)
    assert audit["fp16_underflow_count"] == 128 and audit["fp16_cast_mismatches"] == 128
    assert audit["fp16_cast_max_abs_error"] == 2**-30 and audit["fp16_nonfinite_count"] == 0


def test_torch_cpu_decoder_preserves_bf16_separate_operation_rounding():
    torch = pytest.importorskip("torch")
    packed = torch.full((1, 16), 0x33333333, dtype=torch.uint32)
    scales = torch.tensor([[0.10009765625]], dtype=torch.bfloat16)
    biases = torch.tensor([[-0.25]], dtype=torch.bfloat16)
    fp16, logical = model.decode_affine4bit_torch(packed, scales, biases)
    expected = (torch.tensor(3, dtype=torch.bfloat16) * scales + biases).expand(1, 128)
    assert torch.equal(logical, expected)
    assert torch.equal(logical.float(), torch.full((1, 128), 0.05078125))
    assert torch.equal(fp16.float(), logical.float())


@pytest.mark.parametrize("source_dtype", ["float32", "bfloat16"])
def test_torch_initializer_transpose_preserves_values_and_effective_scale(source_dtype):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, source_dtype)
    a = torch.arange(24, dtype=torch.float32).reshape(3, 8).to(dtype)
    b = torch.arange(16, dtype=torch.float32).reshape(8, 2).to(dtype)
    pa, pb = model.transpose_lora(a), model.transpose_lora(b)
    assert pa.dtype == pb.dtype == torch.float32 and pa.is_contiguous() and pb.is_contiguous()
    assert torch.equal(pa.T, a.float()) and torch.equal(pb.T, b.float())
    x = torch.tensor([[1.0, 2.0, 3.0]])
    mlx_update = ((x @ a.float()) @ b.float()) * 16
    peft_update = torch.nn.functional.linear(torch.nn.functional.linear(x, pa), pb) * (128 / 8)
    assert torch.equal(mlx_update, peft_update)


def test_tiny_meta_qwen_reinitializes_rope_from_config_on_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    accelerate = pytest.importorskip("accelerate")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
    )
    with accelerate.init_empty_weights(include_buffers=True):
        tiny = Qwen3ForCausalLM(config)
    assert tiny.model.rotary_emb.inv_freq.is_meta
    model.initialize_rotary(tiny, config, "cpu")
    expected = Qwen3RotaryEmbedding(config)
    assert torch.equal(tiny.model.rotary_emb.inv_freq, expected.inv_freq)
    assert torch.equal(tiny.model.rotary_emb.original_inv_freq, expected.original_inv_freq)


def test_tiny_meta_qwen_rejects_unknown_nonpersistent_buffer():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    accelerate = pytest.importorskip("accelerate")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
    )
    with accelerate.init_empty_weights(include_buffers=True):
        tiny = Qwen3ForCausalLM(config)
    tiny.register_buffer("unknown_buffer", torch.zeros(1), persistent=False)
    with pytest.raises(ValueError, match="Unexpected Qwen3 buffers"):
        model.initialize_rotary(tiny, config, "cpu")


def test_tiny_cpu_qwen_actual_peft_injection_layout_scale_and_master_gradients():
    """Exercise actual PEFT dispatch before any large-base load, CUDA or download."""
    if any(importlib.util.find_spec(name) is None for name in ("torch", "transformers", "peft")):
        pytest.skip("Optional Torch/Transformers/PEFT packages absent from this CPU environment")
    # Installed-but-broken optional integrations must fail, rather than being hidden by importorskip.
    import peft
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(7)
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=128,
            tie_word_embeddings=True,
        )
        config._attn_implementation = "eager"  # CPU fixture; real CUDA trainer separately requires SDPA.
        base = Qwen3ForCausalLM(config).to(dtype=torch.float16, device="cpu")
        base.requires_grad_(False)
        values = model.lora_config_values(config.to_dict(), num_layers=1)
        adapted = peft.get_peft_model(base, peft.LoraConfig(**values), autocast_adapter_dtype=True)
        assert len(values["target_modules"]) == 7
        expected = set()
        with torch.no_grad():
            for name in values["target_modules"]:
                layer = adapted.get_base_model().get_submodule(name)
                assert layer.r["default"] == 8 and layer.scaling["default"] == 16
                for suffix, component, shape in (
                    ("lora_a", "lora_A", (layer.in_features, 8)),
                    ("lora_b", "lora_B", (8, layer.out_features)),
                ):
                    source = torch.linspace(-0.02, 0.02, math.prod(shape), dtype=torch.float32).reshape(shape)
                    parameter = getattr(layer, component)["default"].weight
                    assert parameter.dtype == torch.float32 and parameter.device.type == "cpu"
                    parameter.copy_(model.transpose_lora(source))
                    assert torch.equal(parameter.T, source), suffix
                    expected.add(f"base_model.model.{name}.{component}.default.weight")
        parameters = dict(adapted.named_parameters())
        trainable = {name: p for name, p in parameters.items() if p.requires_grad}
        assert set(trainable) == expected and len(trainable) == 14
        assert sum(p.numel() for p in trainable.values()) == 17_408
        assert all(p.dtype == torch.float16 for name, p in parameters.items() if name not in trainable)
        causal = adapted.get_base_model()
        assert causal.get_input_embeddings().weight is causal.get_output_embeddings().weight
        tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device="cpu")
        adapted.train()
        loss = adapted(input_ids=tokens, labels=tokens, use_cache=False).loss
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(
            p.grad is not None and p.grad.dtype == torch.float32 and bool(torch.isfinite(p.grad).all())
            for p in trainable.values()
        )
        assert sum(float(p.grad.abs().sum()) for p in trainable.values()) > 0
