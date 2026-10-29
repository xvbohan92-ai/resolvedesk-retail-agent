from resolve_desk.decision_facts import augment_messages
from training.evaluate import Provider


class FactsProvider(Provider):
    def __init__(self, model, tokenizer, config, variant, enabled=True):
        super().__init__(model, tokenizer, config, variant)
        self.enabled = enabled

    def generate(self, prompt):
        return super().generate(augment_messages(prompt) if self.enabled else prompt)
