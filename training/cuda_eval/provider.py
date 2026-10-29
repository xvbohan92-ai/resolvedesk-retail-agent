"""Unconstrained matched FP16 greedy generation; no inference occurs at import."""

import hashlib
import json
import time

MAX_NEW_TOKENS = 512
MODEL_KWARGS = {"logits_to_keep": 1}


def token_hash(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


def decode_completion(tokenizer, generated, eos_ids):
    """Preserve invalid/empty/special output; remove only one actual terminal stop token."""
    ids = list(generated)
    terminal = bool(ids and ids[-1] in eos_ids)
    text_ids = ids[:-1] if terminal else ids
    raw = tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    full = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return raw, {
        "generated_token_ids": ids,
        "generated_token_ids_sha256": token_hash(ids),
        "decoded_with_terminal_special_tokens": full,
        "terminal_eos_removed_for_text": terminal,
        "finish_reason": "eos" if terminal else "length",
        "output_tokens": len(ids),
    }


def generation_settings(config):
    eos = config["eos_token_id"]
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    if not eos_ids or not all(isinstance(v, int) for v in eos_ids):
        raise ValueError("Pinned EOS IDs must be explicit")
    pad = config["pad_token_id"]
    if not isinstance(pad, int):
        raise ValueError("Pinned pad ID must be explicit")
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "eos_token_id": eos,
        "pad_token_id": pad,
        "bos_token_id": config.get("bos_token_id"),
        "repetition_penalty": 1.0,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "return_dict_in_generate": False,
        "output_scores": False,
    }


class TorchProvider:
    def __init__(self, model, tokenizer, config, variant, adapter_sha256, device="cuda"):
        import torch
        from transformers import GenerationConfig

        self.torch, self.model, self.tokenizer = torch, model, tokenizer
        self.variant, self.adapter_sha256, self.device = variant, adapter_sha256, torch.device(device)
        self.settings = generation_settings(config)
        eos = self.settings["eos_token_id"]
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos)
        if (
            tokenizer.eos_token_id not in self.eos_ids
            or tokenizer.pad_token_id != self.settings["pad_token_id"]
        ):
            raise ValueError("Tokenizer EOS/pad differs from the pinned model configuration")
        self.generation_config = GenerationConfig(**self.settings)
        model.eval()
        model.gradient_checkpointing_disable()
        model.get_base_model().config.use_cache = True

    def generate(self, prompt):
        torch = self.torch
        ids = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False
        )
        if not isinstance(ids, list) or not ids or not all(isinstance(v, int) for v in ids):
            raise ValueError("Expected the complete untruncated chat-template token IDs")
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention = torch.ones_like(tokens)
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            output = self.model.generate(
                input_ids=tokens,
                attention_mask=attention,
                generation_config=self.generation_config,
                **MODEL_KWARGS,
            )
        if cuda:
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        generated = output[0, len(ids) :].detach().cpu().tolist()
        raw, decoded = decode_completion(self.tokenizer, generated, self.eos_ids)
        timing = {
            "provider": "torch_cuda" if cuda else "torch_cpu_fixture",
            "variant": self.variant,
            "adapter_sha256": self.adapter_sha256,
            "workflow": "v7",
            "input_tokens": len(ids),
            "input_token_ids": ids,
            "input_token_ids_sha256": token_hash(ids),
            "latency_ms": round(elapsed * 1000, 3),
            "generation_settings": self.settings,
            "model_kwargs": dict(MODEL_KWARGS),
            "gradient_checkpointing": False,
            "json_schema_constrained": False,
            "autocast": False,
            "cloud_compute_used": cuda,
            **decoded,
        }
        if cuda:
            timing.update(
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
            )
        return raw, timing


class RecordingProvider:
    """A durable start precedes each call; interruption never authorizes replaying that call."""

    def __init__(self, provider, journal, variant, case_id):
        self.provider, self.journal, self.variant, self.case_id = provider, journal, variant, case_id
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if self.calls > 12:
            raise RuntimeError("Total model-call budget exceeded")
        entry = self.journal.start(self.variant, self.case_id, self.calls, prompt)
        raw, timing = self.provider.generate(prompt)
        timing = {**timing, "workflow": "v7"}
        self.journal.finish(entry, raw, timing)
        return raw, timing
