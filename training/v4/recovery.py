"""Recovery-state action preference with observed affordability facts."""

import argparse
import json
import math
import random
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

from resolve_desk.decision_facts import augment_messages
from resolve_desk.progress import ProgressStore
from training.common import ROOT, sha
from training.v4.common import REPORTS
from training.v4.fixtures import load_cases
from training.v4.prepare import apply, prefix
from training.v4.prepare import row as plain_row

PAIR_PATH = ROOT / "training/v4/datasets/recovery.jsonl"
PROTOCOL = ROOT / "training/v4/recovery-protocol.json"
MANIFEST = ROOT / "reports/v4/recovery-manifest.json"


def row(state, action, origin):
    example = plain_row(state, action, origin)
    example["messages"] = augment_messages(example["messages"][:-1]) + example["messages"][-1:]
    return example


def prepare():
    from training.v4.action_preference import MANIFEST as previous_manifest
    from training.v4.action_preference import verify_protocol as verify_previous

    assert not MANIFEST.exists()
    assert not (REPORTS / "evaluation-partial.json").exists()
    assert not (REPORTS / "evaluation.json").exists()
    verify_previous()
    previous = json.loads((REPORTS / "action-updates.json").read_text())
    assert len(json.loads((REPORTS / "validation-616.json").read_text())["records"]) == 32
    pairs = []

    def add(state, chosen, rejected, category, case):
        pairs.append(
            {
                "chosen": row(state, chosen, category),
                "rejected": row(state, rejected, category),
                "category": category,
                "case_id": case["id"],
            }
        )

    def exchange(case, variant, payment=None):
        return {
            "tool": "propose_exchange",
            "arguments": {
                "order_id": case["answers"]["order_id"],
                "item_id": case["answers"]["item_id"],
                "new_item_id": variant,
                "payment_method": payment or case["answers"]["payment_method"],
            },
        }

    wrong_handoff = {"tool": "handoff", "arguments": {"reason": "Unable to proceed."}}
    with tempfile.TemporaryDirectory() as tmp:
        store = ProgressStore(Path(tmp) / "recovery.sqlite")
        for case in load_cases("train"):
            terminal, action, upstream, _ = prefix(store, case)
            field = case["required_clarification"]
            if field:
                asked = next(
                    json.loads(r["messages"][-1]["content"])
                    for r in upstream
                    if json.loads(r["messages"][-1]["content"])["tool"] == "ask_customer"
                )
                add(terminal, action, asked, "preserve_answered_continuation", case)
                state = store.create(case)
                if field == "email":
                    state = apply(
                        store,
                        state,
                        {"tool": "identify_customer", "arguments": {"email": "unknown@retail.example"}},
                    )
                    rejected = wrong_handoff
                else:
                    assert case["answers"]["email"] in case["initial_message"]
                    assert case["answers"]["order_id"] in case["initial_message"]
                    state = apply(
                        store,
                        state,
                        {"tool": "identify_customer", "arguments": {"email": case["answers"]["email"]}},
                    )
                    state = apply(
                        store,
                        state,
                        {"tool": "get_order", "arguments": {"order_id": case["answers"]["order_id"]}},
                    )
                    if case["family"].startswith("exchange"):
                        state = apply(
                            store,
                            state,
                            {
                                "tool": "get_product",
                                "arguments": {"product_id": case["orders"][0]["items"][0]["product_id"]},
                            },
                        )
                        variants = next(iter(case["products"].values()))["variants"]
                        target = next(
                            v
                            for v in variants
                            if v["available"]
                            and v["item_id"] != case["answers"]["item_id"]
                            and (field == "new_option" or v["option"] == case["answers"]["new_option"])
                        )
                        rejected = exchange(case, target["item_id"], "original")
                    elif field == "reason":
                        rejected = {
                            "tool": "propose_cancel",
                            "arguments": {
                                "order_id": case["answers"]["order_id"],
                                "reason": "ordered by mistake",
                            },
                        }
                    else:
                        rejected = {
                            "tool": "propose_return",
                            "arguments": {
                                "order_id": case["answers"]["order_id"],
                                "item_id": case["answers"]["item_id"],
                                "payment_method": "original",
                            },
                        }
                add(state, asked, rejected, "clarify_after_reads_instead_of_guessing", case)
            if case["family"] in {"exchange_unavailable", "exchange_shortfall"}:
                variants = next(iter(case["products"].values()))["variants"]
                alt = next(
                    v for v in variants if v["available"] and v["item_id"] != case["answers"]["item_id"]
                )
                add(
                    terminal,
                    action,
                    exchange(case, alt["item_id"], "original"),
                    "handoff_vs_unsafe_exchange",
                    case,
                )
                bad_return = {
                    "tool": "propose_return",
                    "arguments": {
                        "order_id": case["answers"]["order_id"],
                        "item_id": case["answers"]["item_id"],
                        "payment_method": case["answers"]["payment_method"],
                    },
                }
                add(terminal, action, bad_return, "handoff_vs_unauthorized_return", case)
            if case["family"] == "exchange_fallback":
                add(terminal, action, wrong_handoff, "authorized_stockout_return", case)
            if case["family"] in {"exchange_gift", "exchange_fallback_available"}:
                rejected = (
                    wrong_handoff
                    if case["family"] == "exchange_gift"
                    else {
                        "tool": "propose_return",
                        "arguments": {k: v for k, v in action["arguments"].items() if k != "new_item_id"},
                    }
                )
                add(terminal, action, rejected, "preserve_available_exchange", case)
            if case["family"] == "return_reject_exchange":
                variants = next(iter(case["products"].values()))["variants"]
                alt = next(
                    v for v in variants if v["available"] and v["item_id"] != case["answers"]["item_id"]
                )
                add(terminal, action, exchange(case, alt["item_id"]), "preserve_return", case)
            if case["family"] in {"forbidden_status", "foreign_order"}:
                add(
                    terminal,
                    action,
                    {
                        "tool": "ask_customer",
                        "arguments": {"field": "payment_method", "message": "Which payment method?"},
                    },
                    "preserve_policy_handoff",
                    case,
                )
    assert len(pairs) == 80
    assert all(p["case_id"].startswith("v4-train-") for p in pairs)
    PAIR_PATH.write_text("".join(json.dumps(p) + "\n" for p in pairs))
    config = {
        "method": "Recovery-state action-span preference plus chosen-completion NLL",
        "beta": 2.0,
        "margin": 0.5,
        "sft_weight": 1.0,
        "learning_rate": 0.000003,
        "seed": 2026102743,
        "updates": 80,
        "batch_size": 1,
        "projection_window": 256,
        "max_seq_length": 3072,
        "initialization": previous["protocol"]["output"],
        "output": "data/finetuning/retail-qlora-v4/recovery",
        "best_path": "data/finetuning/retail-qlora-v4/best-recovery",
        "selection": "Fixed 80-update recovery checkpoint before any new test inference.",
        "training_pairs": dict(Counter(p["category"] for p in pairs)),
        "scope": (
            "Original training worlds only. "
            "New states omit missing answers even after real order/catalog reads. "
            "No validation/test examples enter training. New facts are arithmetic over actual completed "
            "tool results for all candidates, never selected answers or intent routing."
        ),
    }
    PROTOCOL.write_text(json.dumps(config, indent=2) + "\n")
    files = [
        Path(__file__),
        PROTOCOL,
        PAIR_PATH,
        ROOT / "training/v4/recovery_select.py",
        ROOT / "training/v4/facts_evaluate.py",
        ROOT / "training/v4/facts_provider.py",
        ROOT / "resolve_desk/decision_facts.py",
        ROOT / "resolve_desk/agent.py",
    ]
    manifest = {
        "original_action_manifest_sha256": sha(previous_manifest),
        "initial_weights_sha256": previous["adapter_sha256"],
        "previous_validation_sha256": sha(REPORTS / "validation-616.json"),
        "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "pair_count": 80,
        "test_inference_started_at_freeze": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pairs": 80, "categories": config["training_pairs"]}))


def verify_protocol():
    from training.v4.action_preference import MANIFEST as previous_manifest
    from training.v4.action_preference import verify_protocol as verify_previous

    verify_previous()
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["original_action_manifest_sha256"] == sha(previous_manifest)
    assert manifest["previous_validation_sha256"] == sha(REPORTS / "validation-616.json")
    for name, digest in manifest["files"].items():
        assert sha(ROOT / name) == digest, name
    return manifest


def train(preflight=False):
    from functools import partial

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from transformers import AutoTokenizer

    from training.v4.tokenization import encode_row

    manifest = verify_protocol()
    config = json.loads(PROTOCOL.read_text())
    sft = json.loads((ROOT / "training/v4/config.json").read_text())
    origin = ROOT / config["initialization"]
    assert sha(origin / "adapters.safetensors") == manifest["initial_weights_sha256"]
    destination = ROOT / (config["output"] + ("-preflight" if preflight else ""))
    destination.mkdir(parents=True, exist_ok=False)
    random.seed(config["seed"])
    mx.random.seed(config["seed"])
    wired_limit = min(mx.device_info()["max_recommended_working_set_size"], 6 * 1024**3)
    previous_wired_limit = mx.set_wired_limit(wired_limit)
    previous_cache_limit = mx.set_cache_limit(256 * 1024**2)
    model, _ = load(ROOT / sft["model"], trust_remote_code=False)
    model.freeze()
    linear_to_lora_layers(model, sft["num_layers"], sft["lora_parameters"])
    model.load_weights(str(origin / "adapters.safetensors"), strict=False)
    grad_checkpoint(model.layers[0])
    model.train()
    initial = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
    mx.eval(initial)
    hf = AutoTokenizer.from_pretrained(ROOT / sft["model"], local_files_only=True, trust_remote_code=False)
    pairs = [json.loads(line) for line in PAIR_PATH.read_text().splitlines()]
    random.shuffle(pairs)
    encoded = [(encode_row(p["chosen"], hf, 2), encode_row(p["rejected"], hf, 2), p) for p in pairs]
    assert max(len(e["tokens"]) for pair in encoded for e in pair[:2]) <= config["max_seq_length"]
    window = config["projection_window"]

    def batch(example):
        size = len(example["tokens"])
        padded = 1 + 32 * ((size + 31) // 32)
        tokens = example["tokens"] + [0] * (padded - size)
        weights = example["weights"] + [0.0] * (padded - size)
        assert not any(weights[:-window]), "A supervised target would be dropped"
        full_mask = [float(w > 0) for w in weights[-window:]]
        action_mask = [float(w == 2) for w in weights[-window:]]
        assert sum(action_mask) and sum(full_mask)
        return mx.array([tokens]), mx.array([full_mask]), mx.array([action_mask])

    # Audit every train pair before any updates; only output projection is shortened.
    for chosen, rejected, _ in encoded:
        batch(chosen)
        batch(rejected)

    def stats(model, tokens, full_mask, action_mask):
        hidden = model.model(tokens[:, :-1])[:, -window:]
        assert model.args.tie_word_embeddings
        logits = model.model.embed_tokens.as_linear(hidden)
        ce = nn.losses.cross_entropy(logits, tokens[:, -window:]).astype(mx.float32)
        return -(ce * action_mask).sum() / action_mask.sum(), -(ce * full_mask).sum() / full_mask.sum()

    def rejected_logp(model, tokens, full_mask, action_mask):
        return stats(model, tokens, full_mask, action_mask)[0]

    def chosen_loss(model, tokens, full_mask, action_mask, rejected_value):
        action_lp, full_lp = stats(model, tokens, full_mask, action_mask)
        z = config["beta"] * (action_lp - mx.stop_gradient(rejected_value)) - config["margin"]
        loss = mx.logaddexp(mx.array(0.0), -z) - config["sft_weight"] * full_lp
        return loss, (action_lp, full_lp, z)

    reject_grad = nn.value_and_grad(model, rejected_logp)
    choose_grad = nn.value_and_grad(model, chosen_loss)

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def compiled_rejected(tokens, full_mask, action_mask):
        return reject_grad(model, tokens, full_mask, action_mask)

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def compiled_chosen(tokens, full_mask, action_mask, rejected_value):
        return choose_grad(model, tokens, full_mask, action_mask, rejected_value)

    gradient_audit = None
    projection_audit = None
    if preflight:
        theta = mx.array([0.2, -0.7, 0.9])

        def fa(t):
            return nn.log_softmax(t)[0]

        def fr(t):
            return nn.log_softmax(2 * t)[1]

        def fp(t):
            return (nn.log_softmax(t)[0] + nn.log_softmax(t)[2]) / 2

        def objective(t):
            z = config["beta"] * (fa(t) - fr(t)) - config["margin"]
            return mx.logaddexp(mx.array(0.0), -z) - config["sft_weight"] * fp(t)

        rv = fr(theta)

        def chosen_only(t):
            z = config["beta"] * (fa(t) - rv) - config["margin"]
            return mx.logaddexp(mx.array(0.0), -z) - config["sft_weight"] * fp(t)

        z = config["beta"] * (fa(theta) - rv) - config["margin"]
        combined = mx.grad(chosen_only)(theta) + config["beta"] * mx.sigmoid(-z) * mx.grad(fr)(theta)
        exact = mx.grad(objective)(theta)
        error = float(mx.max(mx.abs(combined - exact)))
        eps = mx.array([0.001, 0.0, 0.0])
        finite = float((objective(theta + eps) - objective(theta - eps)) / 0.002)
        assert error < 1e-6 and abs(finite - float(exact[0])) < 1e-3
        gradient_audit = {
            "chain_rule_max_absolute_error": error,
            "finite_difference_error": abs(finite - float(exact[0])),
        }
        tokens, fm, am = batch(encoded[0][0])
        full_logits = model(tokens[:, :-1])[:, -window:]
        ce = nn.losses.cross_entropy(full_logits, tokens[:, -window:]).astype(mx.float32)
        full_stats = (-(ce * am).sum() / am.sum(), -(ce * fm).sum() / fm.sum())
        tail_stats = stats(model, tokens, fm, am)
        mx.eval(full_stats, tail_stats)
        projection_error = max(abs(float(a) - float(b)) for a, b in zip(full_stats, tail_stats, strict=True))
        assert projection_error < 0.01
        projection_audit = {
            "window": window,
            "all_train_targets_covered": True,
            "mean_logp_max_error_vs_full_projection": projection_error,
        }
        del full_logits, ce, full_stats, tail_stats
        mx.clear_cache()

    optimizer = optim.Adam(learning_rate=config["learning_rate"])
    metrics, started = [], time.perf_counter()
    for iteration, (chosen, rejected, pair) in enumerate(encoded[:1] if preflight else encoded, 1):
        ca, cm, cam = batch(chosen)
        ra, rm, ram = batch(rejected)
        lr, gr = compiled_rejected(ra, rm, ram)
        mx.eval(lr, gr)
        (loss, (lc, lp, z)), gc = compiled_chosen(ca, cm, cam, lr)
        mx.eval(loss, lc, lp, z, gc)
        coefficient = config["beta"] / (1 + math.exp(max(-60, min(60, float(z)))))
        combined = tree_map(lambda a, b, coefficient=coefficient: a + coefficient * b, gc, gr)
        optimizer.update(model, combined)
        mx.eval(model.parameters(), optimizer.state)
        record = {
            "iteration": iteration,
            "category": pair["category"],
            "loss": float(loss),
            "chosen_action_logp": float(lc),
            "rejected_action_logp": float(lr),
            "chosen_completion_nll": -float(lp),
            "pre_update_mean_logp_gap": float(lc) - float(lr),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if preflight:
            a, p = stats(model, ca, cm, cam)
            b = rejected_logp(model, ra, rm, ram)
            mx.eval(a, p, b)
            record["post_update_mean_logp_gap"] = float(a) - float(b)
            record["post_update_chosen_completion_nll"] = -float(p)
            assert record["post_update_mean_logp_gap"] > record["pre_update_mean_logp_gap"]
            assert record["post_update_chosen_completion_nll"] < record["chosen_completion_nll"]
        metrics.append(record)
        if iteration == 1 or iteration % 10 == 0 or preflight:
            print(json.dumps(record), flush=True)
        (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        del gc, gr, combined
        mx.clear_cache()
    final = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(destination / "adapters.safetensors"), final)
    shutil.copyfile(origin / "adapter_config.json", destination / "adapter_config.json")
    delta = (
        sum(
            float(mx.sum((final[k].astype(mx.float32) - initial[k].astype(mx.float32)) ** 2)) for k in initial
        )
        ** 0.5
    )
    assert delta > 0
    report = {
        "protocol": config,
        "manifest": manifest,
        "metrics": metrics,
        "completed_updates": len(metrics),
        "trainable_parameters": sum(int(v.size) for v in final.values()),
        "adapter_l2_change_from_action136": delta,
        "adapter_sha256": sha(destination / "adapters.safetensors"),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_memory_bytes": mx.get_peak_memory(),
        "preflight_only": preflight,
        "gradient_audit": gradient_audit,
        "projection_audit": projection_audit,
        "previous_wired_limit": previous_wired_limit,
        "wired_limit_bytes": wired_limit,
        "cache_limit_bytes": 256 * 1024**2,
        "previous_cache_limit": previous_cache_limit,
        "execution": (
            "Compiled gradient with mutable input/output state capture; masked 32-token padding. "
            "Wired memory capped at 6 GiB, cache at 256 MiB, cleared after each pair. "
            "Action136 initialization; no failed unregularized or preflight weights reused."
        ),
        "test_used_for_selection": False,
        "cloud_compute_used": False,
    }
    (REPORTS / ("recovery-preflight.json" if preflight else "recovery-updates.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k not in {"metrics", "manifest", "protocol"}}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "preflight", "train"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else train(args.mode == "preflight")
