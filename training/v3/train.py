"""Actual action-weighted updates; choose by validation business outcomes, never test."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
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
from training.v3.common import REPORTS, read_rows, verify_frozen
from training.v3.evaluation import selection_key
from training.v3.objective import iterate_batches, weighted_loss
from training.v3.tokenization import encode_row


class Recorder(TrainingCallback):
    def __init__(self, model, tokenizer, config, destination):
        self.model, self.tokenizer, self.config, self.destination = model, tokenizer, config, destination
        self.metrics, self.validations, self.best, self.key = [], [], None, None
        self.started = time.perf_counter()

    def record(self, kind, values):
        row = {"kind": kind, "elapsed_seconds": round(time.perf_counter() - self.started, 3), **values}
        self.metrics.append(row)
        (self.destination / "metrics.json").write_text(json.dumps(self.metrics, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    def on_train_loss_report(self, values):
        self.record("train", values)

    def on_val_loss_report(self, values):
        iteration = values["iteration"]
        self.record("weighted_valid_loss", values)
        if iteration == 0:
            return
        assert iteration in self.config["checkpoint_candidates_completed_updates"]
        candidate = self.destination / "candidates" / str(iteration)
        candidate.mkdir(parents=True, exist_ok=False)
        mx.save_safetensors(
            str(candidate / "adapters.safetensors"), dict(tree_flatten(self.model.trainable_parameters()))
        )
        shutil.copyfile(self.destination / "adapter_config.json", candidate / "adapter_config.json")
        mx.clear_cache()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "training.v3.validation",
                "--adapter",
                str(candidate),
                "--iteration",
                str(iteration),
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            check=True,
        )
        result = json.loads((REPORTS / f"validation-{iteration}.json").read_text())
        assert not result["preflight_only"] and len(result["records"]) == 16
        assert result["adapter_sha256"] == sha(candidate / "adapters.safetensors")
        summary = result["summary"]
        key = selection_key(summary, values["val_loss"])
        result.update(selection_key=list(key), weighted_valid_loss=values["val_loss"])
        self.validations.append(result)
        (REPORTS / f"validation-{iteration}.json").write_text(json.dumps(result, indent=2) + "\n")
        if self.key is None or key > self.key:
            self.key = key
            self.best = {
                "iteration": iteration,
                "weighted_valid_loss": values["val_loss"],
                "validation_summary": summary,
                "selection_key": list(key),
            }
            shutil.copyfile(candidate / "adapters.safetensors", self.destination / "best.safetensors")
        print(
            json.dumps({"validation_summary": summary, "selected_updates": self.best["iteration"]}),
            flush=True,
        )
        self.model.train()
        mx.clear_cache()


def main():
    manifest = verify_frozen()
    config = json.loads((ROOT / "training/v3/config.json").read_text())
    audit = json.loads((REPORTS / "token-audit.json").read_text())
    assert audit["dataset"] == manifest and audit["config_sha256"] == sha(ROOT / "training/v3/config.json")
    destination, best = ROOT / config["adapter_path"], ROOT / config["best_path"]
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("V3 run already exists")
    destination.mkdir(parents=True, exist_ok=True)
    np.random.seed(config["seed"])
    mx.random.seed(config["seed"])
    model, tokenizer = load(ROOT / config["model"], trust_remote_code=False)
    hf = AutoTokenizer.from_pretrained(ROOT / config["model"], trust_remote_code=False, local_files_only=True)
    datasets = {
        s: [encode_row(r, hf, config["tool_name_loss_weight"]) for r in read_rows(s)]
        for s in ("train", "valid")
    }
    for s in datasets:
        for row, encoded in zip(read_rows(s), datasets[s], strict=True):
            assert (
                tokenizer.apply_chat_template(row["messages"], enable_thinking=False, return_dict=False)
                == encoded["tokens"]
            )
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    parameters = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(destination / "initial.safetensors"), parameters)
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": config["num_layers"],
        "lora_parameters": config["lora_parameters"],
    }
    (destination / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n")
    protocol = {
        "config": config,
        "config_sha256": sha(ROOT / "training/v3/config.json"),
        "dataset": manifest,
        "runner_sha256": sha(__file__),
        "objective_sha256": sha(ROOT / "training/v3/objective.py"),
        "tokenization_sha256": sha(ROOT / "training/v3/tokenization.py"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "mlx_version": importlib.metadata.version("mlx"),
        "test_used_for_selection": False,
        "initialization": "Pinned base, no v1/v2 adapter",
    }
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    recorder = Recorder(model, tokenizer, config, destination)
    args = TrainingArgs(
        batch_size=1,
        iters=config["iters"],
        val_batches=-1,
        steps_per_report=config["steps_per_report"],
        steps_per_eval=config["steps_per_eval"],
        steps_per_save=config["save_every"],
        adapter_file=str(destination / "adapters.safetensors"),
        max_seq_length=config["max_seq_length"],
        grad_checkpoint=True,
    )
    train(
        model,
        optim.Adam(learning_rate=config["learning_rate"]),
        datasets["train"],
        datasets["valid"],
        args=args,
        loss=weighted_loss,
        iterate_batches=iterate_batches,
        training_callback=recorder,
    )
    assert recorder.best is not None
    best.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(destination / "best.safetensors", best / "adapters.safetensors")
    shutil.copyfile(destination / "adapter_config.json", best / "adapter_config.json")
    initial, selected = (
        mx.load(str(destination / "initial.safetensors")),
        mx.load(str(best / "adapters.safetensors")),
    )
    delta = (
        sum(
            float(mx.sum((selected[k].astype(mx.float32) - initial[k].astype(mx.float32)) ** 2))
            for k in initial
        )
        ** 0.5
    )
    assert delta > 0
    report = {
        "status": "trained",
        "config": config,
        "dataset": manifest,
        "best_checkpoint": recorder.best,
        "metrics": recorder.metrics,
        "validation_candidates": [
            {k: v for k, v in r.items() if k != "records"} for r in recorder.validations
        ],
        "initial_adapter_sha256": sha(destination / "initial.safetensors"),
        "selected_adapter_sha256": sha(best / "adapters.safetensors"),
        "final_adapter_sha256": sha(destination / "adapters.safetensors"),
        "adapter_l2_change": delta,
        "trainable_parameters": sum(int(p.size) for p in selected.values()),
        "elapsed_seconds": round(time.perf_counter() - recorder.started, 2),
        "peak_memory_bytes": mx.get_peak_memory(),
        "test_used_for_selection": False,
        "cloud_compute_used": False,
    }
    (REPORTS / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    (REPORTS / "training-protocol.json").write_bytes((destination / "protocol.json").read_bytes())
    print(
        json.dumps(
            {k: report[k] for k in ("status", "best_checkpoint", "adapter_l2_change", "elapsed_seconds")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
