import importlib.metadata
import json

import mlx.core as mx
from transformers import AutoTokenizer

from training.common import ROOT, sha
from training.v4.common import REPORTS, read_rows, verify_frozen
from training.v4.objective import weighted_loss
from training.v4.tokenization import encode_row


def main():
    manifest = verify_frozen()
    config = json.loads((ROOT / "training/v4/config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / config["model"], trust_remote_code=False, local_files_only=True
    )
    counts = {}
    for split in ("train", "valid", "test"):
        encoded = [encode_row(r, tokenizer, config["tool_name_loss_weight"]) for r in read_rows(split)]
        assert all(len(r["tokens"]) <= config["max_seq_length"] for r in encoded)
        assert all(sum(w > 0 for w in r["weights"]) == len(r["tokens"]) - r["prompt_length"] for r in encoded)
        counts[split] = {
            "rows": len(encoded),
            "max_tokens": max(len(r["tokens"]) for r in encoded),
            "tool_tokens_upweighted": sum(sum(w == 2 for w in r["weights"]) for r in encoded),
            "all_prompt_masks_and_offsets_valid": True,
            "truncated": 0,
        }

    class Model:
        def __init__(self, logits):
            self.logits = mx.array([logits])

        def __call__(self, _):
            return self.logits

    batch, lengths, weights = mx.array([[0, 0, 1, 0]]), mx.array([[2, 4]]), mx.array([[0.0, 2.0, 1.0]])
    wrong_tool, _ = weighted_loss(Model([[6.0, -6.0], [6.0, -6.0], [6.0, -6.0]]), batch, lengths, weights)
    wrong_argument, _ = weighted_loss(Model([[6.0, -6.0], [-6.0, 6.0], [-6.0, 6.0]]), batch, lengths, weights)
    ratio = float(wrong_tool / wrong_argument)
    assert 1.99 < ratio < 2.01, ratio
    spec = json.loads((ROOT / "training/base-model-spec.json").read_text())
    for f in spec["files"]:
        p = ROOT / config["model"] / f["path"]
        assert p.stat().st_size == f["bytes"] and sha(p) == f["sha256"]
    report = {
        "base_model_spec_sha256": sha(ROOT / "training/base-model-spec.json"),
        "base_model_files": spec["files"],
        "dataset": manifest,
        "config_sha256": sha(ROOT / "training/v4/config.json"),
        "counts": counts,
        "numerical_objective_audit": {
            "equal_single_token_error_tool_vs_argument_loss_ratio": ratio,
            "expected_ratio": 2.0,
            "prompt_loss_masked": True,
        },
        "versions": {
            n: importlib.metadata.version(n) for n in ("mlx", "mlx-lm", "transformers", "tokenizers")
        },
        "test_model_inference_performed": False,
    }
    (REPORTS / "token-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": counts, "objective_ratio": ratio}), flush=True)


if __name__ == "__main__":
    main()
