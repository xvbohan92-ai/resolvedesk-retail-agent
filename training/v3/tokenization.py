"""Precise character-to-target-token weights; no heuristic prefix-length mask."""

import json


def target_weights(offsets, prompt_length, name_start, name_end, multiplier):
    weights = []
    for token_index, (start, end) in enumerate(offsets[1:], 1):
        weight = 0.0 if token_index < prompt_length else 1.0
        if weight and start < name_end and end > name_start:
            weight = multiplier
        weights.append(weight)
    assert any(w == multiplier for w in weights)
    assert all(w == 0 for w in weights[: prompt_length - 1])
    return weights


def encode_row(row, tokenizer, multiplier):
    conversation = row["messages"]
    rendered = tokenizer.apply_chat_template(conversation, enable_thinking=False, tokenize=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoded["input_ids"]
    expected = tokenizer.apply_chat_template(
        conversation, enable_thinking=False, tokenize=True, return_dict=False
    )
    assert tokens == expected
    prompt = tokenizer.apply_chat_template(
        conversation[:-1], enable_thinking=False, add_generation_prompt=True, tokenize=True, return_dict=False
    )
    assert tokens[: len(prompt)] == prompt
    content = conversation[-1]["content"]
    tool = json.loads(content)["tool"]
    assert content.startswith('{"tool":"' + tool + '"')
    start = rendered.rfind(content) + len('{"tool":"')
    assert rendered[start : start + len(tool)] == tool
    weights = target_weights(encoded["offset_mapping"], len(prompt), start, start + len(tool), multiplier)
    return {
        "tokens": tokens,
        "prompt_length": len(prompt),
        "weights": weights,
        "tool": tool,
        "origin": row.get("origin", "reference"),
    }
