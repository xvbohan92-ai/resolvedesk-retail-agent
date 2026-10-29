import importlib.metadata
import json

from mlx_lm import load

from training.common import ROOT, NoThinkingTokenizer, read_rows, sha, verify_frozen


def main():
    manifest = verify_frozen()
    config = json.loads((ROOT / "training/config.json").read_text())
    _, tokenizer = load(ROOT / config["model"], trust_remote_code=False)
    tokenizer = NoThinkingTokenizer(tokenizer)
    counts = {}
    for split in ("train", "valid", "test"):
        lengths = []
        for row in read_rows(split):
            conversation = row["messages"]
            all_tokens = tokenizer.apply_chat_template(conversation, tokenize=True, return_dict=False)
            prompt = tokenizer.apply_chat_template(
                conversation[:-1], add_generation_prompt=True, tokenize=True, return_dict=False
            )
            assert all_tokens[: len(prompt)] == prompt, "Prompt mask must cover exactly the prefix"
            assert len(all_tokens) <= config["max_seq_length"], (split, len(all_tokens))
            lengths.append(len(all_tokens))
        counts[split] = {
            "rows": len(lengths),
            "max_tokens": max(lengths),
            "all_completion_masks_valid": True,
            "truncated": 0,
        }
    report = {
        "dataset": manifest,
        "training_config_sha256": sha(ROOT / "training/config.json"),
        "counts": counts,
        "versions": {
            n: importlib.metadata.version(n) for n in ("mlx-lm", "mlx", "transformers", "tokenizers")
        },
        "test_inference_performed": False,
        "scope": "Tokenization and label audit only",
    }
    (ROOT / "reports/token-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
