import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(split):
    return [json.loads(s) for s in (ROOT / f"training/datasets/{split}.jsonl").read_text().splitlines()]


def verify_frozen():
    m = json.loads((ROOT / "training/datasets/manifest.json").read_text())
    for name, digest in m["files"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen experiment input changed: {name}")
    return m


class NoThinkingTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(*args, **kwargs)
