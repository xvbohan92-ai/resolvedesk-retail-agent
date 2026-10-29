"""Audit exact field-value target spans without supervising prompts or padding."""

import json
import re

from training.v4.tokenization import encode_row as base_encode


def encode_row(row, tokenizer, permission_weight, reason_weight):
    result = base_encode(row, tokenizer, 1.0)
    result.update(permission_tokens=0, reason_tokens=0)
    content = row["messages"][-1]["content"]
    if json.loads(content)["tool"] != "record_request":
        return result
    rendered = tokenizer.apply_chat_template(row["messages"], enable_thinking=False, tokenize=False)
    offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    for field, multiplier, count_key in [
        ("fallback_return", permission_weight, "permission_tokens"),
        ("reason", reason_weight, "reason_tokens"),
    ]:
        match = re.search(r'"' + field + r'":"([^"\\]*)"', content)
        assert match, field
        start, end = rendered.rfind(content) + match.start(1), rendered.rfind(content) + match.end(1)
        for i, (a, b) in enumerate(offsets[1:]):
            if result["weights"][i] and a < end and b > start:
                result["weights"][i] = multiplier
                result[count_key] += 1
        assert result[count_key], field
    assert not any(result["weights"][: result["prompt_length"] - 1])
    return result
