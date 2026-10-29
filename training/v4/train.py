"""Export exact-update candidates; validate only after this GPU process exits."""

import argparse
import importlib.metadata
import json
import shutil
import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers
from transformers import AutoTokenizer

from training.common import ROOT, sha
from training.v4.common import REPORTS, read_rows, verify_frozen
from training.v4.objective import iterate_batches, weighted_loss
from training.v4.tokenization import encode_row


class Recorder(TrainingCallback):
    def __init__(self, model, config, destination, preflight):
        self.model, self.config, self.destination = model, config, destination
        self.preflight, self.metrics, self.candidates = preflight, [], []
        self.started = time.perf_counter()

    def on_train_loss_report(self, values):
        self.metrics.append({"elapsed_seconds": round(time.perf_counter() - self.started, 3), **values})
        (self.destination / "metrics.json").write_text(json.dumps(self.metrics, indent=2) + "\n")
        print(json.dumps(self.metrics[-1]), flush=True)
        iteration = values["iteration"]
        if not self.preflight and iteration in self.config["checkpoint_candidates_completed_updates"]:
            target = self.destination / "candidates" / str(iteration)
            target.mkdir(parents=True, exist_ok=False)
            mx.save_safetensors(
                str(target / "adapters.safetensors"), dict(tree_flatten(self.model.trainable_parameters()))
            )
            shutil.copyfile(self.destination / "adapter_config.json", target / "adapter_config.json")
            self.candidates.append(
                {
                    "completed_updates": iteration,
                    "path": str(target.relative_to(ROOT)),
                    "adapter_sha256": sha(target / "adapters.safetensors"),
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = verify_frozen()
    config = json.loads((ROOT / "training/v4/config.json").read_text())
    audit = json.loads((REPORTS / "token-audit.json").read_text())
    assert audit["dataset"] == manifest
    destination = ROOT / (
        "data/finetuning/retail-qlora-v4/preflight" if args.preflight else config["adapter_path"]
    )
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Training run already exists")
    destination.mkdir(parents=True, exist_ok=True)
    np.random.seed(config["seed"])
    mx.random.seed(config["seed"])
    model, tokenizer = load(ROOT / config["model"], trust_remote_code=False)
    hf = AutoTokenizer.from_pretrained(ROOT / config["model"], local_files_only=True, trust_remote_code=False)
    examples = read_rows("train")[:4] if args.preflight else read_rows("train")
    dataset = [encode_row(r, hf, config["tool_name_loss_weight"]) for r in examples]
    assert all(
        tokenizer.apply_chat_template(r["messages"], enable_thinking=False, return_dict=False) == e["tokens"]
        for r, e in zip(examples, dataset, strict=True)
    )
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    parameters = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(destination / "initial.safetensors"), parameters)
    ac = {
        "fine_tune_type": "lora",
        "num_layers": config["num_layers"],
        "lora_parameters": config["lora_parameters"],
    }
    (destination / "adapter_config.json").write_text(json.dumps(ac, indent=2) + "\n")
    recorder = Recorder(model, config, destination, args.preflight)
    iters = 1 if args.preflight else config["iters"]
    train(
        model,
        optim.Adam(learning_rate=config["learning_rate"]),
        dataset,
        val_dataset=None,
        args=TrainingArgs(
            batch_size=1,
            iters=iters,
            steps_per_report=1 if args.preflight else config["steps_per_report"],
            steps_per_save=iters,
            adapter_file=str(destination / "adapters.safetensors"),
            max_seq_length=config["max_seq_length"],
            grad_checkpoint=True,
        ),
        loss=weighted_loss,
        iterate_batches=iterate_batches,
        training_callback=recorder,
    )
    initial = mx.load(str(destination / "initial.safetensors"))
    final = mx.load(str(destination / "adapters.safetensors"))
    delta = (
        sum(
            float(mx.sum((final[k].astype(mx.float32) - initial[k].astype(mx.float32)) ** 2)) for k in initial
        )
        ** 0.5
    )
    assert delta > 0
    report = {
        "status": "updates_complete",
        "config": config,
        "dataset": manifest,
        "metrics": recorder.metrics,
        "candidates": recorder.candidates,
        "completed_updates": iters,
        "initial_adapter_sha256": sha(destination / "initial.safetensors"),
        "final_adapter_sha256": sha(destination / "adapters.safetensors"),
        "adapter_l2_change_final": delta,
        "trainable_parameters": sum(int(p.size) for p in final.values()),
        "elapsed_seconds": round(time.perf_counter() - recorder.started, 2),
        "peak_memory_bytes": mx.get_peak_memory(),
        "preflight_only": args.preflight,
        "initialization": "Pinned base; no previous or preflight adapter reused",
        "versions": {
            n: importlib.metadata.version(n) for n in ["mlx", "mlx-lm", "transformers", "tokenizers"]
        },
        "test_used_for_selection": False,
        "cloud_compute_used": False,
    }
    (REPORTS / ("preflight.json" if args.preflight else "updates.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "completed_updates": iters,
                "adapter_l2_change_final": delta,
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
