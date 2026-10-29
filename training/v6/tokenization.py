"""Offset-audited permission-value weighting, preserving every completion token."""

import json
import re

from training.v4.tokenization import encode_row as base_encode


def encode_row(row, tokenizer, multiplier):
    result = base_encode(row, tokenizer, 1.0)
    result["permission_tokens"] = 0
    content = row["messages"][-1]["content"]
    if json.loads(content)["tool"] != "record_request":
        return result
    rendered = tokenizer.apply_chat_template(row["messages"], enable_thinking=False, tokenize=False)
    offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    match = re.search(r'"fallback_return":"(yes|no|unknown)"', content)
    assert match
    start = rendered.rfind(content) + match.start(1)
    end = rendered.rfind(content) + match.end(1)
    for i, (a, b) in enumerate(offsets[1:]):
        if result["weights"][i] and a < end and b > start:
            result["weights"][i] = multiplier
            result["permission_tokens"] += 1
    assert result["permission_tokens"] and all(
        w == 0 for w in result["weights"][: result["prompt_length"] - 1]
    )
    return result
