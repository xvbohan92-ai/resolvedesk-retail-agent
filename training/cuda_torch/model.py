"""Stream the pinned MLX snapshot into an FP16 HF base and import exact F32 LoRA.

This is LoRA on decoded-4bit-derived FP16 weights, not bitsandbytes/NF4 QLoRA.
Only load_model imports the GPU/model frameworks. No model downloads are performed.
"""

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

INITIALIZER_SHA256 = "041cde85d1593f27f154da5b4afbec2f0635fb2bcfe4a34940339e092925ce08"
BASE_REPO = "Qwen/Qwen3-4B-MLX-4bit"
BASE_REVISION = "52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495"
RANK, MLX_SCALE, LAST_LAYERS = 8, 16.0, 8
EXPECTED_ADAPTER_TENSORS, EXPECTED_ADAPTER_PARAMETERS = 112, 3_670_016
LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
GROUP_SIZE = 128
CHUNK_ROWS = 256
MINIMUM_FREE_BYTES = 12 * 1024**3


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path):
    with Path(path).open("rb") as source:
        size = struct.unpack("<Q", source.read(8))[0]
        if not 0 < size <= 16 * 1024**2:
            raise ValueError("Unexpected safetensors header size")
        header = json.loads(source.read(size))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def target_modules(config, num_layers=LAST_LAYERS):
    total = int(config["num_hidden_layers"])
    if not 0 < num_layers <= total:
        raise ValueError("Invalid number of final LoRA layers")
    return [f"model.layers.{i}.{name}" for i in range(total - num_layers, total) for name in LINEARS]


def lora_config_values(config, num_layers=LAST_LAYERS):
    return {
        "r": RANK,
        "lora_alpha": int(MLX_SCALE * RANK),
        "lora_dropout": 0.0,
        "target_modules": target_modules(config, num_layers),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "use_rslora": False,
        "init_lora_weights": False,
    }


def adapter_plan(header, config, num_layers=LAST_LAYERS):
    hidden, intermediate = int(config["hidden_size"]), int(config["intermediate_size"])
    head_dim = int(config.get("head_dim", hidden // int(config["num_attention_heads"])))
    query, kv = int(config["num_attention_heads"]) * head_dim, int(config["num_key_value_heads"]) * head_dim
    dimensions = {
        "self_attn.q_proj": (hidden, query),
        "self_attn.k_proj": (hidden, kv),
        "self_attn.v_proj": (hidden, kv),
        "self_attn.o_proj": (query, hidden),
        "mlp.gate_proj": (hidden, intermediate),
        "mlp.up_proj": (hidden, intermediate),
        "mlp.down_proj": (intermediate, hidden),
    }
    plan = []
    for module in target_modules(config, num_layers):
        short = ".".join(module.split(".")[-2:])
        inputs, outputs = dimensions[short]
        for suffix, shape, peft in (
            ("lora_a", (inputs, RANK), "lora_A"),
            ("lora_b", (RANK, outputs), "lora_B"),
        ):
            name = f"{module}.{suffix}"
            if name not in header or tuple(header[name]["shape"]) != shape:
                raise ValueError(f"Missing or mismatched initializer tensor: {name}")
            if header[name]["dtype"] not in {"F32", "BF16"}:
                raise ValueError(f"Unsupported initializer dtype: {name}")
            plan.append(
                {
                    "source": name,
                    "module": module,
                    "peft_component": peft,
                    "source_shape": shape,
                    "target_shape": shape[::-1],
                    "source_dtype": header[name]["dtype"],
                    "target": f"base_model.model.{module}.{peft}.default.weight",
                }
            )
    if set(header) != {p["source"] for p in plan}:
        raise ValueError("Unexpected extra initializer tensors")
    return plan


def base_plan(header, parameter_shapes, tied=True):
    """Strict same-name Qwen3 mapping, including q_norm/k_norm and the tied head."""
    shapes = dict(parameter_shapes)
    if tied:
        if shapes.get("lm_head.weight") != shapes.get("model.embed_tokens.weight"):
            raise ValueError("Tied embedding/head shapes must match")
        shapes.pop("lm_head.weight", None)
    expected, plan = set(), []
    for name, shape in shapes.items():
        shape = tuple(shape)
        if name not in header:
            raise ValueError(f"Missing base tensor: {name}")
        value = header[name]
        if value["dtype"] == "U32":
            if len(shape) != 2 or shape[1] % GROUP_SIZE:
                raise ValueError(f"Invalid affine4bit target dimensions: {name}")
            stem = name.removesuffix(".weight")
            keys = (name, stem + ".scales", stem + ".biases")
            if tuple(value["shape"]) != (shape[0], shape[1] // 8):
                raise ValueError(f"Packed shape mismatch: {name}")
            for key in keys[1:]:
                if (
                    key not in header
                    or header[key]["dtype"] != "BF16"
                    or tuple(header[key]["shape"]) != (shape[0], shape[1] // GROUP_SIZE)
                ):
                    raise ValueError(f"Affine scale/bias mismatch: {key}")
            kind = "affine4bit"
        elif value["dtype"] == "BF16" and tuple(value["shape"]) == shape:
            keys, kind = (name,), "bf16"
        else:
            raise ValueError(f"Base dtype/shape mismatch: {name}")
        expected.update(keys)
        plan.append({"name": name, "shape": shape, "source_keys": keys, "kind": kind})
    if set(header) != expected:
        raise ValueError("Unexpected extra base tensors (including any duplicate tied head)")
    return plan


def round_bf16_numpy(values):
    """IEEE RN-even logical BF16 represented in F32; reference only, no MLX import."""
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return np.where(np.isfinite(values), rounded.view(np.float32), values).astype(np.float32)


def decode_affine4bit_numpy(packed, scales, biases, group_size=GROUP_SIZE):
    """MLX CPU reference: low nibble first, BF16-round multiply, BF16-round add."""
    import numpy as np

    packed, scales, biases = np.asarray(packed), np.asarray(scales), np.asarray(biases)
    if packed.dtype.kind != "u" or packed.dtype.itemsize != 4 or packed.ndim != 2:
        raise ValueError("Expected a two-dimensional U32 packed tensor")
    width = packed.shape[1] * 8
    if group_size <= 0 or width % group_size or scales.shape != (packed.shape[0], width // group_size):
        raise ValueError("Affine group/scale dimensions mismatch")
    if biases.shape != scales.shape:
        raise ValueError("Affine bias dimensions mismatch")
    shifts = np.arange(8, dtype=np.uint32) * 4
    q = ((packed.astype(np.uint32)[..., None] >> shifts) & 15).reshape(packed.shape[0], width)
    product = round_bf16_numpy(
        q.astype(np.float32) * np.repeat(scales.astype(np.float32), group_size, axis=1)
    )
    logical = round_bf16_numpy(product + np.repeat(biases.astype(np.float32), group_size, axis=1))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return logical.astype(np.float16), logical


def decode_affine4bit_torch(packed, scales, biases, group_size=GROUP_SIZE):
    import torch

    if packed.device.type != "cpu" or scales.device.type != "cpu" or biases.device.type != "cpu":
        raise ValueError("Decode source slices on CPU; only the FP16 destination goes to CUDA")
    if packed.dtype != torch.uint32 or packed.ndim != 2:
        raise ValueError("Expected a two-dimensional U32 packed tensor")
    rows, width = packed.shape[0], packed.shape[1] * 8
    if (
        group_size <= 0
        or width % group_size
        or tuple(scales.shape) != (rows, width // group_size)
        or biases.shape != scales.shape
        or scales.dtype != torch.bfloat16
        or biases.dtype != torch.bfloat16
    ):
        raise ValueError("Expected group-aligned BF16 affine scales and biases")
    shifts = torch.arange(8, dtype=torch.int64) * 4
    q = ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & 15).reshape(rows, width).float()
    # Actual MLX CPU dequant rounds each BF16 operation separately, not one F32 FMA.
    product = (q * scales.float().repeat_interleave(group_size, dim=1)).to(torch.bfloat16)
    logical = (product.float() + biases.float().repeat_interleave(group_size, dim=1)).to(torch.bfloat16)
    return logical.to(torch.float16), logical


def cast_stats(logical, converted):
    import torch

    before, after = logical.float(), converted.float()
    difference = (before - after).abs()
    return {
        "elements": logical.numel(),
        "fp16_cast_mismatches": int((before != after).sum()),
        "fp16_cast_max_abs_error": float(difference.max()) if difference.numel() else 0.0,
        "fp16_underflow_count": int(((before != 0) & (after == 0)).sum()),
        "logical_nonfinite_count": int((~torch.isfinite(before)).sum()),
        "fp16_nonfinite_count": int((~torch.isfinite(after)).sum()),
    }


def transpose_lora(values):
    """MLX (in,r)/(r,out) -> PEFT (r,in)/(out,r), with lossless F32 masters."""
    import torch

    if values.ndim != 2 or values.dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("Initializer tensors must be two-dimensional F32 or BF16")
    return values.to(torch.float32).T.contiguous()


def _merge_stats(total, current):
    for key, value in current.items():
        total[key] = (
            max(total.get(key, 0), value) if key.endswith("max_abs_error") else total.get(key, 0) + value
        )


def _assign_parameter(model, name, value):
    import torch

    parent, _, leaf = name.rpartition(".")
    module = model.get_submodule(parent)
    if tuple(module._parameters[leaf].shape) != tuple(value.shape):
        raise ValueError(f"Destination shape mismatch: {name}")
    module._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)


def initialize_rotary(model, config, device):
    """Recreate known nonpersistent RoPE state from config, never from empty storage."""
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    expected = {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"}
    if set(dict(model.named_buffers(remove_duplicate=False))) != expected:
        raise ValueError("Unexpected Qwen3 buffers; refusing to silently materialize unknown state")
    # Transformers5.21 has no device argument and registers BOTH frequency buffers.
    # The device context makes its official factory compute real values on the target.
    with torch.device(device):
        rotary = Qwen3RotaryEmbedding(config=config)
    model.model.rotary_emb = rotary
    for name, value in model.named_buffers(remove_duplicate=False):
        if name not in expected or value.is_meta or value.device != torch.device(device):
            raise ValueError("RoPE buffer was not initialized on the target device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite rotary frequencies")
    original = model.model.rotary_emb.original_inv_freq
    if (
        original.is_meta
        or original.device != torch.device(device)
        or not bool(torch.isfinite(original).all())
    ):
        raise ValueError("Original rotary frequencies were not initialized")


def _verify_sources(base_path, initializer_path):
    spec = json.loads((Path(__file__).resolve().parents[1] / "base-model-spec.json").read_text())
    if (spec["repo"], spec["revision"]) != (BASE_REPO, BASE_REVISION):
        raise ValueError("Unexpected base snapshot specification")
    for item in spec["files"]:
        path = base_path / item["path"]
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"Pinned base file mismatch: {item['path']}")
    initializer = initializer_path / "adapters.safetensors" if initializer_path.is_dir() else initializer_path
    if sha(initializer) != INITIALIZER_SHA256:
        raise ValueError("Initializer is not the exact selected V6.1 adapter")
    adapter_config = json.loads(initializer.with_name("adapter_config.json").read_text())
    if adapter_config.get("num_layers") != LAST_LAYERS or adapter_config.get("lora_parameters") != {
        "rank": RANK,
        "dropout": 0.0,
        "scale": MLX_SCALE,
    }:
        raise ValueError("Initializer LoRA configuration mismatch")
    return spec, initializer


def load_model(base_path, initializer_path, device="cuda"):
    """Return (PEFT model, audit). No forward pass, update, download or input write."""
    if not str(device).startswith("cuda"):
        raise RuntimeError("This separate FP16 loader requires CUDA; no CPU fallback")
    import torch
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, Qwen3Config

    target = torch.device(device)
    if not torch.cuda.is_available():
        raise RuntimeError("No usable NVIDIA CUDA GPU")
    index = target.index if target.index is not None else torch.cuda.current_device()
    target = torch.device("cuda", index)
    properties = torch.cuda.get_device_properties(index)
    if (properties.major, properties.minor) < (7, 5):
        raise RuntimeError("Require NVIDIA compute capability >=7.5")
    free, total = torch.cuda.mem_get_info(index)
    if free < MINIMUM_FREE_BYTES:
        raise MemoryError("Require at least12GiB free before loading the decoded FP16 base")
    torch.cuda.reset_peak_memory_stats(index)
    allocated_before = torch.cuda.memory_allocated(index)
    base_path, initializer_path = Path(base_path), Path(initializer_path)
    spec, initializer = _verify_sources(base_path, initializer_path)
    raw = json.loads((base_path / "config.json").read_text())
    quant = raw.get("quantization")
    if (
        raw.get("model_type") != "qwen3"
        or not raw.get("tie_word_embeddings")
        or quant != {"group_size": GROUP_SIZE, "bits": 4}
        or raw.get("quantization_config") != quant
    ):
        raise ValueError("Only the pinned tied-embedding Qwen3 affine4bit/group128 snapshot is supported")
    clean = {k: v for k, v in raw.items() if k not in {"quantization", "quantization_config", "torch_dtype"}}
    config = Qwen3Config.from_dict(clean)
    config.dtype, config.use_cache = torch.float16, False
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16, attn_implementation="sdpa")
    model.requires_grad_(False)
    model.tie_weights()
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters(remove_duplicate=False)}
    weight_file = base_path / "model.safetensors"
    plan = base_plan(read_header(weight_file), shapes, tied=True)
    adapters = adapter_plan(read_header(initializer), raw)
    if len(adapters) != EXPECTED_ADAPTER_TENSORS or sum(math.prod(p["source_shape"]) for p in adapters) != (
        EXPECTED_ADAPTER_PARAMETERS
    ):
        raise ValueError("The predecessor must contain112 tensors /3,670,016 parameters")
    aggregate, tensor_audit = {}, []
    with torch.no_grad(), safe_open(str(weight_file), framework="pt", device="cpu") as source:
        for item in plan:
            destination = torch.empty(item["shape"], dtype=torch.float16, device=target)
            stats = {}
            if item["kind"] == "affine4bit":
                packed, scales, biases = (source.get_slice(k) for k in item["source_keys"])
                for start in range(0, item["shape"][0], CHUNK_ROWS):
                    end = min(start + CHUNK_ROWS, item["shape"][0])
                    converted, logical = decode_affine4bit_torch(
                        packed[start:end], scales[start:end], biases[start:end]
                    )
                    current = cast_stats(logical, converted)
                    _merge_stats(stats, current)
                    if current["logical_nonfinite_count"] or current["fp16_nonfinite_count"]:
                        raise ValueError(f"Nonfinite decoded/cast weights: {item['name']}: {current}")
                    destination[start:end].copy_(converted)
                    del converted, logical
            else:
                logical = source.get_slice(item["name"])[:]
                converted = logical.to(torch.float16)
                stats = cast_stats(logical, converted)
                if stats["logical_nonfinite_count"] or stats["fp16_nonfinite_count"]:
                    raise ValueError(f"Nonfinite base norm: {item['name']}")
                destination.copy_(converted)
                del converted, logical
            _assign_parameter(model, item["name"], destination)
            _merge_stats(aggregate, stats)
            tensor_audit.append({"name": item["name"], "kind": item["kind"], **stats})
            del destination
    model.tie_weights()
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("Input embedding and output head must share one FP16 allocation")
    initialize_rotary(model, config, target)
    model = get_peft_model(model, LoraConfig(**lora_config_values(raw)), autocast_adapter_dtype=True)
    causal = model.get_base_model()
    expected_names, dtype_conversions = {p["target"] for p in adapters}, []
    with torch.no_grad(), safe_open(str(initializer), framework="pt", device="cpu") as source:
        for item in adapters:
            layer = causal.get_submodule(item["module"])
            if layer.scaling["default"] != MLX_SCALE:
                raise ValueError("PEFT effective alpha/r must equal MLX's direct scale16")
            parameter = getattr(layer, item["peft_component"])["default"].weight
            values = source.get_slice(item["source"])[:]
            reference = transpose_lora(values)
            if tuple(parameter.shape) != item["target_shape"]:
                raise ValueError("LoRA target layout mismatch")
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.to(torch.float32)
            parameter.copy_(reference.to(target))
            if not torch.equal(parameter.detach().cpu(), reference) or not bool(
                torch.isfinite(reference).all()
            ):
                raise ValueError(f"Initializer numeric equality failed: {item['source']}")
            if values.dtype != torch.float32:
                dtype_conversions.append(
                    {
                        "source": item["source"],
                        "from": str(values.dtype),
                        "to": "torch.float32",
                        "numeric_values_exact": True,
                    }
                )
    parameters = dict(model.named_parameters())
    trainable = {name: p for name, p in parameters.items() if p.requires_grad}
    if (
        set(trainable) != expected_names
        or sum(p.numel() for p in trainable.values()) != EXPECTED_ADAPTER_PARAMETERS
    ):
        raise ValueError("Unexpected trainable tensors: only the exact112 imported LoRA tensors may train")
    if any(p.dtype != torch.float32 for p in trainable.values()):
        raise ValueError("LoRA master tensors must be F32")
    if any(
        p.is_meta or p.device.type != "cuda" or p.dtype != torch.float16
        for name, p in parameters.items()
        if name not in trainable
    ):
        raise ValueError("All frozen base parameters must be materialized CUDA FP16")
    if causal.get_input_embeddings().weight is not causal.get_output_embeddings().weight:
        raise ValueError("PEFT must preserve the tied embedding/head allocation")
    model.eval()
    audit = {
        "experiment": "separate_cuda_fp16_lora",
        "base_repo": spec["repo"],
        "base_revision": spec["revision"],
        "base_model_sha256": sha(weight_file),
        "initializer_sha256": sha(initializer),
        "source_tensor_count": len(adapters),
        "source_parameter_count": EXPECTED_ADAPTER_PARAMETERS,
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "exact_tensor_equality": True,
        "initializer_dtype_conversions": dtype_conversions,
        "adapter_master_dtype": "float32",
        "source_dtype_counts": {
            dtype: sum(p["source_dtype"] == dtype for p in adapters)
            for dtype in sorted({p["source_dtype"] for p in adapters})
        },
        "base_storage_dtype": "float16",
        "dequant_logical_dtype": "bfloat16",
        "dequant_arithmetic": "BF16-rounded multiply followed by BF16-rounded add; MLX CPU dequant reference",
        "source_bits": 4,
        "source_group_size": GROUP_SIZE,
        "lora_rank": RANK,
        "mlx_direct_scale": MLX_SCALE,
        "peft_lora_alpha": int(MLX_SCALE * RANK),
        "target_modules": target_modules(raw),
        "tied_embedding_head": True,
        "rotary_reinitialized_from_config": True,
        "rotary_buffer_names": ["model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"],
        "all_parameters_and_buffers_materialized": True,
        "attention_backend": config._attn_implementation,
        "not_qlora_or_nf4": True,
        "mlx_compute_bitwise_equivalence_claimed": False,
        "no_whole_float32_base_materialized": True,
        "stream_chunk_rows": CHUNK_ROWS,
        "base_tensor_count": len(plan),
        "base_unique_parameters": sum(math.prod(p["shape"]) for p in plan),
        **aggregate,
        "base_cast_audit": tensor_audit,
        "gpu": {
            "name": properties.name,
            "free_bytes_before": free,
            "total_bytes": total,
            "allocated_bytes_before": allocated_before,
            "allocated_bytes_after": torch.cuda.memory_allocated(index),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
        },
        "model_source_sha256": sha(__file__),
        "python": sys.version.split()[0],
    }
    return model, audit
