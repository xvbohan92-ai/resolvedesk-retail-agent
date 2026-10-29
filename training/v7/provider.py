"""Every real generation is counted; resource waiting never unloads other models."""

from training.v7.resources import wait_for_room


class Provider:
    def __init__(self, model, tokenizer, config, variant, legacy=False):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.variant, self.legacy = variant, legacy

    def generate(self, prompt):
        import mlx.core as mx

        from training.runtime import generate_action

        waited = wait_for_room()
        if self.legacy:
            from resolve_desk.decision_facts import augment_messages

            prompt = augment_messages(prompt)
        raw, timing = generate_action(self.model, self.tokenizer, prompt, self.config)
        mx.clear_cache()
        return raw, {**timing, "provider": "mlx", "variant": self.variant, "resource_wait_seconds": waited}
