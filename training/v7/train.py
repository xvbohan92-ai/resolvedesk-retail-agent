"""V6.1 continuation: balanced intent/recovery SFT with audited field-value weights."""

import argparse
import json
import math
import random
import shutil
import time
from functools import partial

from training.common import ROOT, sha
from training.v7.common import CONFIG, REPORTS, read_rows, verify_frozen, verify_local_base
from training.v7.resources import wait_for_room
from training.v7.tokenization import encode_row


def sampling_buckets(rows):
    buckets = {
        "intent_and_record_repair": [i for i, r in enumerate(rows) if r["origin"] == "record_request"],
        "execution_and_reply": [i for i, r in enumerate(rows) if r["origin"] != "record_request"],
    }
    assert all(buckets.values())
    return buckets


def balanced_sample_indices(rows, seed, iterations):
    buckets, rng = sampling_buckets(rows), random.Random(seed)
    orders = {name: [] for name in buckets}
    sampled = []
    for step in range(1, iterations + 1):
        name = "intent_and_record_repair" if step % 2 else "execution_and_reply"
        if not orders[name]:
            orders[name] = list(buckets[name])
            rng.shuffle(orders[name])
        sampled.append((name, orders[name].pop()))
    return sampled


def projection_batch(example, window, max_seq_length):
    size = len(example["tokens"])
    assert 2 <= size <= max_seq_length
    assert len(example["weights"]) == size - 1
    padded = 1 + 32 * ((size + 31) // 32)
    w = min(window, padded - 1)
    tokens = example["tokens"] + [0] * (padded - size)
    weights = example["weights"] + [0.0] * (padded - 1 - len(example["weights"]))
    assert not any(weights[:-w]), "A supervised completion token would be dropped"
    return tokens, weights[-w:]


def main():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = verify_frozen()
    config = json.loads(CONFIG.read_text())
    verify_local_base(config)
    assert sha(ROOT / config["initialization"] / "adapters.safetensors") == config["initialization_sha256"]
    destination = ROOT / (config["adapter_path"] + ("-preflight" if args.preflight else ""))
    destination.mkdir(parents=True, exist_ok=False)
    wait_for_room()
    mx.set_wired_limit(6 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    mx.random.seed(config["seed"])
    model, tokenizer = load(ROOT / config["model"], trust_remote_code=False)
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    model.load_weights(str(ROOT / config["initialization"] / "adapters.safetensors"), strict=False)
    grad_checkpoint(model.layers[0])
    model.train()
    initial = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
    mx.eval(initial)
    mx.save_safetensors(str(destination / "initial.safetensors"), initial)
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": config["num_layers"],
        "lora_parameters": config["lora_parameters"],
    }
    (destination / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n")
    hf = AutoTokenizer.from_pretrained(ROOT / config["model"], local_files_only=True, trust_remote_code=False)
    rows = read_rows("train")
    encoded = [
        encode_row(r, hf, config["permission_slot_loss_weight"], config["reason_slot_loss_weight"])
        for r in rows
    ]
    window = config["projection_window"]

    def batch(example):
        tokens, weights = projection_batch(example, window, config["max_seq_length"])
        return mx.array([tokens]), mx.array([weights], dtype=mx.float32)

    for e in encoded:
        batch(e)
    for i in [0, len(rows) - 1]:
        assert (
            tokenizer.apply_chat_template(rows[i]["messages"], enable_thinking=False, return_dict=False)
            == encoded[i]["tokens"]
        )

    def loss(model, tokens, weights):
        hidden = model.model(tokens[:, :-1])[:, -weights.shape[1] :]
        logits = model.model.embed_tokens.as_linear(hidden)
        ce = nn.losses.cross_entropy(logits, tokens[:, -weights.shape[1] :]).astype(mx.float32)
        return (ce * weights).sum() / weights.sum()

    grad = nn.value_and_grad(model, loss)

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def compiled(tokens, weights):
        return grad(model, tokens, weights)

    optimizer = optim.Adam(learning_rate=config["learning_rate"])
    buckets = sampling_buckets(rows)
    schedule = balanced_sample_indices(rows, config["seed"], 1 if args.preflight else config["iters"])
    sampled = []
    metrics, candidates = [], []
    iterations = 1 if args.preflight else config["iters"]
    preflight_index = next(
        i
        for i, r in enumerate(rows)
        if r["origin"] == "record_request"
        and r.get("sample_role") == "initial_record"
        and "exchange_fallback-00" in r["case_id"]
    )
    audit = {
        "all_targets_covered": True,
        "longest_train_sequence": max(len(e["tokens"]) for e in encoded),
        "longest_train_completion": max(len(e["tokens"]) - e["prompt_length"] for e in encoded),
        "ordinary_completion_weight": 1.0,
        "permission_value_weight": config["permission_slot_loss_weight"],
        "permission_value_tokens": sum(e["permission_tokens"] for e in encoded),
        "reason_value_weight": config["reason_slot_loss_weight"],
        "reason_value_tokens": sum(e["reason_tokens"] for e in encoded),
        "prompt_and_padding_weight": 0.0,
        "projection_window": window,
    }
    if args.preflight:
        tokens, weights = batch(encoded[preflight_index])
        w = weights.shape[1]
        full_logits = model(tokens[:, :-1])
        full_ce = nn.losses.cross_entropy(full_logits[:, -w:], tokens[:, -w:]).astype(mx.float32)
        full_loss = (full_ce * weights).sum() / weights.sum()
        selected_loss = loss(model, tokens, weights)
        mx.eval(full_loss, selected_loss)
        audit["loss_error_vs_full_output_projection"] = abs(float(full_loss) - float(selected_loss))
        assert audit["loss_error_vs_full_output_projection"] < 0.001
        del full_logits, full_ce
        mx.clear_cache()
    started = time.perf_counter()
    for step in range(1, iterations + 1):
        bucket, index = schedule[step - 1]
        if args.preflight:
            index = preflight_index
        sampled.append(index)
        resource_wait = wait_for_room()
        tokens, weights = batch(encoded[index])
        value, gradients = compiled(tokens, weights)
        mx.eval(value, gradients)
        assert math.isfinite(float(value)) and float(value) >= 0
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        entry = {
            "iteration": step,
            "loss": float(value),
            "case_id": rows[index]["case_id"],
            "tool": rows[index]["origin"],
            "sample_index": index,
            "sample_role": rows[index]["sample_role"],
            "trajectory": rows[index]["trajectory"],
            "sampling_bucket": bucket,
            "resource_wait_seconds": resource_wait,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if args.preflight:
            post = loss(model, tokens, weights)
            mx.eval(post)
            entry["post_update_same_sample_loss"] = float(post)
            assert float(post) < float(value)
        metrics.append(entry)
        if step % 16 == 0 or args.preflight:
            (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(entry), flush=True)
        if not args.preflight and step in config["checkpoint_candidates_completed_updates"]:
            path = destination / "candidates" / str(step)
            path.mkdir(parents=True, exist_ok=False)
            mx.save_safetensors(
                str(path / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters()))
            )
            shutil.copyfile(destination / "adapter_config.json", path / "adapter_config.json")
            candidates.append(
                {
                    "iteration": step,
                    "path": str(path.relative_to(ROOT)),
                    "adapter_sha256": sha(path / "adapters.safetensors"),
                }
            )
        del gradients, value
        mx.clear_cache()
    final = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(destination / "adapters.safetensors"), final)
    delta = (
        sum(
            float(mx.sum((final[k].astype(mx.float32) - initial[k].astype(mx.float32)) ** 2)) for k in initial
        )
        ** 0.5
    )
    assert delta > 0
    report = {
        "config": config,
        "manifest_sha256": sha(ROOT / "training/v7/datasets/manifest.json"),
        "dataset": manifest,
        "initialization_sha256": config["initialization_sha256"],
        "completed_updates": iterations,
        "metrics": metrics,
        "candidates": candidates,
        "preflight_only": args.preflight,
        "preflight_weights_reused": False,
        "adapter_sha256": sha(destination / "adapters.safetensors"),
        "adapter_l2_change": delta,
        "trainable_parameters": sum(v.size for v in final.values()),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_active_memory_bytes": mx.get_peak_memory(),
        "token_audit": audit,
        "test_used_for_selection": False,
        "cloud_compute_used": False,
        "sampling": {
            "policy": (
                "Alternating 50% record/repair targets and 50% execution/reply targets; "
                "shuffled cycles within each pool."
            ),
            "bucket_rows": {k: len(v) for k, v in buckets.items()},
            "bucket_updates": {k: sum(m["sampling_bucket"] == k for m in metrics) for k in buckets},
            "unique_rows_sampled": len(set(sampled)),
            "physical_rows": len(rows),
            "unique_cases_sampled": len({rows[i]["case_id"] for i in sampled}),
            "all_physical_rows_used": len(set(sampled)) == len(rows),
        },
    }
    (REPORTS / ("preflight.json" if args.preflight else "updates.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"metrics", "dataset", "config", "token_audit"}}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
