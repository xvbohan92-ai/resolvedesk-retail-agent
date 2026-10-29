"""V8 fixed-final CUDA LoRA continuation; no historical result or weight is overwritten."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from training.common import ROOT
from training.v7.tokenization import encode_row
from training.v7.train import balanced_sample_indices, projection_batch, sampling_buckets
from training.v8.common import CONFIG, DATA, read_rows, verify_frozen

PROJECTION_WINDOW = 192
EXPECTED_PARAMETERS = 3_670_016
EXPECTED_TENSORS = 112
REPORT_ROOT = ROOT / "reports/v8/cuda"
PACKAGE_PINS = {
    "torch": "2.11.0+cu128",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "transformers": "5.21.0",
    "safetensors": "0.8.0",
    "numpy": "2.5.4",
}
OPTIONAL_PACKAGES = (
    "torchao",
    "bitsandbytes",
    "triton",
    "xformers",
    "flash-attn",
    "torchvision",
    "torchaudio",
    "mlx",
    "mlx-cuda-12",
    "mlx-lm",
    "huggingface-hub",
    "tokenizers",
    "packaging",
)
SCOPE = (
    "New V8 fixed-final continuation from the completed prior CUDA1344 adapter. "
    "Same decoded-4bit-derived FP16 base, F32 rank8 LoRA, weighted next-token objective; "
    "not NF4 QLoRA, not base-versus-finetune and not an accuracy result. "
    "One discarded four-update smoke and one fresh1344-update formal run; no resume or test selection."
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    from training.cuda_eval.evaluation import write_atomic

    write_atomic(path, value)


def dependency_snapshot():
    """Read metadata and module discovery without importing Torch or optional GPU packages."""

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    probe_error, spec = None, None
    try:
        spec = importlib.util.find_spec("torchao")
    except (ImportError, ValueError) as error:
        probe_error = f"{type(error).__name__}: {error}"
    return {
        "python": platform.python_version(),
        "sys_version": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "required_packages": {name: version(name) for name in PACKAGE_PINS},
        "optional_packages": {name: version(name) for name in OPTIONAL_PACKAGES},
        "torchao": {
            "distribution_version": version("torchao"),
            "module_discoverable": spec is not None,
            "module_origin": getattr(spec, "origin", None),
            "module_probe_error": probe_error,
            "in_sys_modules": "torchao" in sys.modules,
        },
        "torchao_policy": "Must be genuinely uninstalled; no import masking or module stubbing.",
    }


def dependency_precheck(expected=None):
    snapshot = dependency_snapshot()
    if expected is not None and snapshot != expected:
        raise RuntimeError("Dependency environment changed after the run plan was recorded")
    optional = snapshot["torchao"]
    if (
        optional["distribution_version"] is not None
        or optional["module_discoverable"]
        or optional["module_probe_error"]
        or optional["in_sys_modules"]
    ):
        raise RuntimeError(
            "Unused torchao must be uninstalled in this isolated Colab runtime before model allocation; "
            "both its import discovery and distribution metadata must be absent. Do not mask the import."
        )
    if snapshot["required_packages"] != PACKAGE_PINS:
        raise RuntimeError("Use the observed Colab package versions recorded in package_pins")
    return snapshot


def weighted_mean(token_losses, weights):
    """Works with Torch tensors and NumPy CPU test arrays; prompt/pad weights are zero."""
    return (token_losses * weights).sum() / weights.sum()


def tail_batch(example, max_seq_length):
    return projection_batch(example, PROJECTION_WINDOW, max_seq_length)


def token_audit(encoded, max_seq_length):
    required_tails, paddings = [], []
    for example in encoded:
        tokens, weights = tail_batch(example, max_seq_length)
        original = example["weights"]
        assert sum(weights) == sum(original) and sum(w > 0 for w in weights) == sum(w > 0 for w in original)
        assert not any(original[: example["prompt_length"] - 1])
        padded_weights = original + [0.0] * (len(tokens) - 1 - len(original))
        first = next(i for i, weight in enumerate(padded_weights) if weight)
        required_tails.append(len(padded_weights) - first)
        paddings.append(len(tokens) - len(example["tokens"]))
    return {
        "all_physical_rows_checked": len(encoded),
        "all_targets_covered": True,
        "full_input_context_retained": True,
        "longest_sequence": max(len(e["tokens"]) for e in encoded),
        "longest_completion": max(len(e["tokens"]) - e["prompt_length"] for e in encoded),
        "maximum_tail_including_padding": max(required_tails),
        "maximum_padding_tokens": max(paddings),
        "projection_window": PROJECTION_WINDOW,
        "mlx_projection_window": 512,
        "prompt_and_padding_weight": 0,
        "ordinary_completion_weight": 1,
        "permission_value_weight": 8,
        "reason_value_weight": 4,
        "permission_value_tokens": sum(e["permission_tokens"] for e in encoded),
        "reason_value_tokens": sum(e["reason_tokens"] for e in encoded),
        "projection_difference": (
            "Only zero-weight logits omitted; no input token or supervised target dropped."
        ),
    }


def schedule_audit(rows, schedule):
    indices = [index for _, index in schedule]
    return {
        "physical_rows": len(rows),
        "unique_rows_sampled": len(set(indices)),
        "unique_cases_sampled": len({rows[index]["case_id"] for index in indices}),
        "all_physical_rows_used": len(set(indices)) == len(rows),
        "bucket_rows": {key: len(value) for key, value in sampling_buckets(rows).items()},
        "bucket_updates": dict(Counter(bucket for bucket, _ in schedule)),
    }


def prepare_plan(destination, kind, steps):
    """Bind immutable inputs before model framework imports or GPU allocation."""
    if kind not in {"smoke", "train"} or not (steps == 4 if kind == "smoke" else steps == 1344):
        raise ValueError("Only four discarded smoke updates or the fixed1344 final run are permitted")
    destination = Path(destination).resolve()
    if not destination.is_relative_to(REPORT_ROOT.resolve()) or destination == REPORT_ROOT.resolve():
        raise ValueError("Keep new runs below reports/v8/cuda/<unique-run>")
    if destination.exists():
        raise FileExistsError("Never overwrite or resume an issued training run")
    manifest = verify_frozen()
    from training.v8.resource_guard import verify_resource_receipt

    verify_resource_receipt()
    config = json.loads(CONFIG.read_text())
    if (config["seed"], config["batch_size"], config["iters"], config["learning_rate"]) != (
        202610288,
        1,
        1344,
        2e-6,
    ):
        raise ValueError("V8 fixed training configuration changed")
    rows = read_rows("train")
    if not rows or len(rows) != len({json.dumps(r["messages"], sort_keys=True) for r in rows}):
        raise ValueError("Training rows must be physically deduplicated")
    base = ROOT / config["model"]
    spec = json.loads((ROOT / "training/base-model-spec.json").read_text())
    if (config["model_repo"], config["model_revision"]) != (spec["repo"], spec["revision"]):
        raise ValueError("Pinned base identity changed")
    for item in spec["files"]:
        if (base / item["path"]).stat().st_size != item["bytes"] or digest(base / item["path"]) != item[
            "sha256"
        ]:
            raise ValueError("Pinned base file changed: " + item["path"])
    initializer = ROOT / config["initialization"] / "adapter_model.safetensors"
    if digest(initializer) != config["initialization_sha256"]:
        raise ValueError("V8 must continue the actual prior final1344 weights")
    schedule = balanced_sample_indices(rows, config["seed"], steps)
    plan = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": SCOPE,
        "kind": kind,
        "planned_true_optimizer_updates": steps,
        "smoke_weights_discarded": True,
        "fresh_predecessor_initialization": True,
        "smoke_weights_reused": False,
        "source_inputs_sha256": dict(manifest["files"]),
        "manifest_sha256": digest(DATA / "manifest.json"),
        "frozen_manifest": manifest,
        "initializer_sha256": digest(initializer),
        "base_spec": spec,
        "frozen_config": config,
        "compute_config": {
            "backend": "pytorch_cuda_peft",
            "base_dtype": "float16",
            "lora_master_dtype": "float32",
            "autocast_dtype": "float16",
            "projection_window": PROJECTION_WINDOW,
            "attention": "sdpa_causal",
            "gradient_checkpointing": "nonreentrant",
            "force_embedding_output_gradients": False,
            "use_cache": False,
            "optimizer": "Adam",
            "learning_rate": 2e-6,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "gradient_clipping": False,
            "grad_scaler_initial_scale": 128.0,
            "batch_size": 1,
            "seed": config["seed"],
            "generation_or_evaluation": False,
        },
        "package_pins": PACKAGE_PINS,
        "dependency_environment": dependency_snapshot(),
        "planned_schedule": [
            {"iteration": i, "bucket": b, "sample_index": n} for i, (b, n) in enumerate(schedule, 1)
        ],
        "planned_sampling": schedule_audit(rows, schedule),
        "checkpoints": [],
        "selection_rule": (
            "Fixed final1344 only, no alternative checkpoint; validation does not select weights"
        ),
        "cloud_compute_used": True,
        "test_used_for_selection": False,
        "new_accuracy_claim": False,
    }
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "plan.json", plan)
    return plan, config, rows, schedule


def check_smoke(plan, smoke_path):
    smoke_path = Path(smoke_path)
    smoke = json.loads(smoke_path.read_text())
    assert smoke["status"] == "completed" and smoke["kind"] == "smoke"
    assert smoke["completed_true_optimizer_updates"] == 4 and smoke["adapter_l2_change"] > 0
    assert smoke["smoke_weights_discarded"] and not smoke["smoke_weights_reused"]
    assert smoke["initializer_audit"]["exact_tensor_equality"] and smoke["all_gradients_finite"]
    assert smoke["initializer_sha256"] == plan["initializer_sha256"]
    assert smoke["source_inputs_sha256"] == plan["source_inputs_sha256"]
    return {
        "report_sha256": digest(smoke_path),
        "path": str(smoke_path.relative_to(ROOT)),
        "weights_reused": False,
    }


def tail_loss(model, tokens, weights, torch):
    causal = model.get_base_model()
    hidden = causal.model(input_ids=tokens[:, :-1], use_cache=False, return_dict=True).last_hidden_state
    width = weights.shape[1]
    logits = causal.get_output_embeddings()(hidden[:, -width:, :])
    ce = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), tokens[:, -width:].reshape(-1), reduction="none"
    ).reshape_as(weights)
    return weighted_mean(ce, weights)


def train(destination, plan, config, rows, schedule, report):
    """Imports GPU libraries only after the result-blind on-disk plan exists."""
    report["dependency_environment"] = dependency_precheck(plan["dependency_environment"])
    write_json(destination / "report.json", report)
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from training.cuda_torch.model import load_model

    if platform.system() != "Linux" or sys.version_info[:2] != (3, 12) or not torch.cuda.is_available():
        raise RuntimeError("This separate experiment requires Linux Python 3.12 and an actual CUDA GPU")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0).strip().casefold() not in {
        "tesla t4",
        "nvidia t4",
        "t4",
    }:
        raise RuntimeError("V8 permits one T4 only; no accelerator fallback")
    assert torch.cuda.get_device_capability(0) >= (7, 5)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    report["environment"] = {
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability(0)),
        "versions": {name: importlib.metadata.version(name) for name in PACKAGE_PINS},
    }
    assert report["environment"]["versions"] == PACKAGE_PINS, "Use the observed Colab package versions"
    write_json(destination / "report.json", report)
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / config["model"], local_files_only=True, trust_remote_code=False
    )
    encoded = [encode_row(r, tokenizer, 8.0, 4.0) for r in rows]
    report["token_audit"] = token_audit(encoded, config["max_seq_length"])
    assert report["token_audit"]["all_physical_rows_checked"] == len(rows)
    write_json(destination / "report.json", report)
    model, initializer_audit = load_model(
        ROOT / config["model"], ROOT / config["legacy_loader_initialization"], device="cuda"
    )
    report["transitional_legacy_loader_audit"] = initializer_audit
    from training.cuda_eval.evaluation import install_final_adapter

    initializer_audit = install_final_adapter(
        model, ROOT / config["initialization"], {"after_adapter_sha256": config["initialization_sha256"]}
    )
    initializer_audit.update(
        initializer_sha256=config["initialization_sha256"],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    report["initializer_audit"] = initializer_audit
    assert initializer_audit["exact_tensor_equality"]
    assert initializer_audit["source_tensor_count"] == EXPECTED_TENSORS
    assert (
        initializer_audit["source_parameter_count"]
        == initializer_audit["trainable_parameters"]
        == EXPECTED_PARAMETERS
    )
    assert initializer_audit["initializer_sha256"] == plan["initializer_sha256"]
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert (
        len(trainable) == EXPECTED_TENSORS
        and sum(p.numel() for p in trainable.values()) == EXPECTED_PARAMETERS
    )
    assert all(p.dtype == torch.float32 and p.is_cuda for p in trainable.values())
    initial = {name: parameter.detach().cpu().contiguous().clone() for name, parameter in trainable.items()}
    initial_cuda = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    assert all(bool(torch.isfinite(p).all()) for p in initial.values())
    save_file(initial, str(destination / "initial-peft.safetensors"))
    report["initial_peft_sha256"] = digest(destination / "initial-peft.safetensors")
    causal = model.get_base_model()
    assert causal.config._attn_implementation == "sdpa"
    causal.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Nonreentrant checkpoints support trainable LoRA parameters with constant inputs.
    # Forcing embedding-output gradients would backpropagate through the frozen prefix.
    model.train()
    optimizer = torch.optim.Adam(
        list(trainable.values()),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )
    true_steps = [0]

    def step_completed(*_):
        true_steps[0] += 1

    step_hook = optimizer.register_step_post_hook(step_completed)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=2000)

    def batch(index):
        tokens, weights = tail_batch(encoded[index], config["max_seq_length"])
        return torch.tensor([tokens], dtype=torch.long, device="cuda"), torch.tensor(
            [weights], dtype=torch.float32, device="cuda"
        )

    def save_adapter(folder):
        folder.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(folder, safe_serialization=True, save_embedding_layers=False)
        assert (folder / "adapter_model.safetensors").is_file() and (folder / "adapter_config.json").is_file()
        return {"path": str(folder.relative_to(ROOT)), "sha256": digest(folder / "adapter_model.safetensors")}

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    report["status"] = "training"
    write_json(destination / "report.json", report)
    try:
        for step, (bucket, index) in enumerate(schedule, 1):
            if digest(ROOT / "reports/v8/resource-authorization.json") != report["resource_receipt_sha256"]:
                raise PermissionError("Resource receipt changed after this training phase started")
            report["attempt_in_progress"] = {
                "iteration": step,
                "sample_index": index,
                "case_id": rows[index]["case_id"],
                "sampling_bucket": bucket,
                "phase": "forward",
            }
            write_json(destination / "report.json", report)
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            tokens, weights = batch(index)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                value = tail_loss(model, tokens, weights, torch)
            loss_value = float(value.detach())
            report["attempt_in_progress"].update(loss=loss_value, phase="backward")
            assert math.isfinite(loss_value) and loss_value >= 0
            scaler.scale(value).backward()
            scaler.unscale_(optimizer)
            gradients = [p.grad for p in trainable.values()]
            assert all(g is not None for g in gradients), "Every LoRA parameter must receive its gradient"
            finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
            report["attempt_in_progress"].update(gradients_finite=finite, phase="optimizer")
            report["all_gradients_finite"] = report["all_gradients_finite"] and finite
            assert finite, (
                "Nonfinite unscaled gradient: abort, do not silently skip or retry an optimizer update"
            )
            gradient_norm = float(torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt())
            assert math.isfinite(gradient_norm) and gradient_norm > 0
            scale_before = scaler.get_scale()
            before = true_steps[0]
            scaler.step(optimizer)
            scaler.update()
            assert true_steps[0] == before + 1 == step, "GradScaler skipped a true Adam update"
            assert scaler.get_scale() >= scale_before, (
                "Overflow scaling changed; the run is not four/formal true updates"
            )
            assert bool(torch.stack([torch.isfinite(p).all() for p in trainable.values()]).all())
            parameter_delta = float(
                torch.stack(
                    [
                        (parameter.detach() - initial_cuda[name]).square().sum()
                        for name, parameter in trainable.items()
                    ]
                )
                .sum()
                .sqrt()
            )
            assert parameter_delta > 0 and math.isfinite(parameter_delta)
            torch.cuda.synchronize()
            entry = {
                "iteration": step,
                "sample_index": index,
                "case_id": rows[index]["case_id"],
                "tool": rows[index]["origin"],
                "sampling_bucket": bucket,
                "loss": loss_value,
                "unscaled_gradient_norm": gradient_norm,
                "gradients_finite": finite,
                "grad_scaler_scale": scale_before,
                "true_optimizer_updates": true_steps[0],
                "adapter_l2_change_from_initializer": parameter_delta,
                "elapsed_gpu_sync_seconds": round(time.perf_counter() - step_started, 6),
                "elapsed_loop_gpu_sync_seconds": round(time.perf_counter() - started, 6),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            report["metrics"].append(entry)
            report["completed_true_optimizer_updates"] = true_steps[0]
            report.pop("attempt_in_progress")
            write_json(destination / "report.json", report)
            print(json.dumps(entry), flush=True)
            if step in plan["checkpoints"]:
                report["candidates"].append(
                    {"iteration": step, **save_adapter(destination / "candidates" / str(step))}
                )
                write_json(destination / "report.json", report)
            del tokens, weights, value, gradients
        report["final_adapter"] = save_adapter(destination / "adapter")
        final = {name: parameter.detach().cpu() for name, parameter in trainable.items()}
        delta = math.sqrt(
            sum(float((final[name] - initial[name]).double().square().sum()) for name in initial)
        )
        assert delta > 0 and math.isfinite(delta)
        report["adapter_l2_change"] = delta
        report["sampling"] = schedule_audit(rows, schedule)
        if plan["kind"] == "train":
            assert true_steps[0] == 1344
            # Coverage is measured, never inferred from the number of optimizer updates.
        report["elapsed_loop_gpu_sync_seconds"] = round(time.perf_counter() - started, 6)
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["timing_scope"] = (
            "Synchronized run wall time includes report/checkpoint/export/audit overhead; "
            "excludes model loading/tokenization and is not kernel-only time."
        )
    finally:
        report["completed_true_optimizer_updates"] = true_steps[0]
        step_hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--kind", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-report", type=Path, default=REPORT_ROOT / "smoke-4/report.json")
    args = parser.parse_args()
    from training.v8.resource_guard import arm_process_deadline, verify_resource_receipt

    receipt = verify_resource_receipt()
    stop_timer = arm_process_deadline(receipt, "training")
    destination = args.output or REPORT_ROOT / f"{args.kind}-{args.steps}"
    plan, config, rows, schedule = prepare_plan(destination, args.kind, args.steps)
    destination = destination.resolve()
    report = {
        "status": "planned",
        "kind": args.kind,
        "scope": SCOPE,
        "plan_sha256": digest(destination / "plan.json"),
        "resource_receipt_sha256": digest(ROOT / "reports/v8/resource-authorization.json"),
        "resource_approval_scope": receipt["permitted_work"],
        "source_inputs_sha256": plan["source_inputs_sha256"],
        "dependency_environment": plan["dependency_environment"],
        "initializer_sha256": plan["initializer_sha256"],
        "smoke_weights_discarded": True,
        "smoke_weights_reused": False,
        "completed_true_optimizer_updates": 0,
        "metrics": [],
        "candidates": [],
        "all_gradients_finite": True,
        "cloud_compute_used": True,
        "test_used_for_selection": False,
        "new_accuracy_claim": False,
    }
    write_json(destination / "report.json", report)
    try:
        if args.kind == "train":
            report["discarded_smoke_receipt"] = check_smoke(plan, args.smoke_report)
        train(destination, plan, config, rows, schedule, report)
        if digest(ROOT / "reports/v8/resource-authorization.json") != report["resource_receipt_sha256"]:
            raise PermissionError("Resource receipt drifted during training")
        for name, wanted in plan["source_inputs_sha256"].items():
            assert digest(ROOT / name) == wanted, name
        report.update(status="completed", completed_at_utc=datetime.now(UTC).isoformat())
    except BaseException as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            failed_at_utc=datetime.now(UTC).isoformat(),
        )
        raise
    finally:
        write_json(destination / "report.json", report)
        stop_timer.cancel()
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "status",
                    "kind",
                    "completed_true_optimizer_updates",
                    "adapter_l2_change",
                    "scope",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
