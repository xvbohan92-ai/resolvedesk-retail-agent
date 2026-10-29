> Archived round 1 experiment. For the current controller, adapter service and new evaluation, use [round 2](v2/README.md). The complete original app is preserved at Git commit `06fabd6`; running its adapter with the current controller is a different system.

# ResolveDesk local Qwen3 QLoRA experiment

The experiment trains a retail **next-tool controller**, not a chat-style customer simulator or an OCR model. The model reads the current customer messages and observed tool results, then emits one application JSON action. The server validates and dispatches it. Business writes remain behind explicit customer confirmation and a SQLite transaction.

## Fixed experiment

- Original base: official `Qwen/Qwen3-4B-MLX-4bit`, revision `52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495`. Nine local model/tokenizer/license files are verified against [base-model-spec.json](base-model-spec.json). The original base is reused; no Gridnote-trained adapter was copied.
- Apple Silicon M5, 16 GB unified memory. MLX-LM 0.32.2 and MLX 0.32.4. Exact training dependencies are in [uv.lock](uv.lock), runtime versions in [token-audit.json](../reports/token-audit.json).
- 4-bit QLoRA, last 8 layers, rank 8, scale 16, dropout 0; 3,670,016 trainable parameters. Adam, learning rate 1e-4, batch 1, gradient checkpointing, seed 20261026, 160 optimizer updates. Only completion tokens contribute to training loss.
- 72 training cases produce 256 decision snapshots; 18 validation cases produce 64; 27 test cases have 96 reference snapshots. Training uses only train snapshots. Validation-only completion loss selects a positive-step checkpoint; initial zero-step loss is diagnostic, and the final step is explicitly evaluated.
- The selected checkpoint is at reported iteration 119 with validation loss 0.0515066. The library evaluates before the next optimizer update, so this represents 119 completed updates. The final 160-update checkpoint has higher validation loss. The selected adapter L2 change from initialization is about 9.01913, proving that weight tensors changed.
- Actual training plus validation took about 439.72 seconds; MLX peak allocated memory was 3.559 GB. These are single-machine training statistics, not end-to-end app latency or total system RAM.
- Thinking is disabled consistently through the Qwen chat template for SFT and paired MLX inference. All training, validation and reference-test examples fit without truncation; the maximum encoded length is 900 tokens, under the fixed 2048-token limit.

The dataset, prompt, labels, task builder and initial outcome grader were frozen before training. All cases and labels are self-authored/programmatic. IDs, numeric values and request-template sets differ across splits, but policy, schema, task families, product vocabulary and author are shared. This is a small in-domain pilot, not unseen-retailer, unseen-policy or independent human-annotated generalization.

## Paired evaluation

Each test task runs an actual multi-step model/tool loop, up to eight model decisions. A deterministic user script supplies a requested field only after the model asks for it. The same predefined confirmation decision is used in both variants. For accepted operations, grading checks the exact operation, item IDs, payment method, amount and final order state; declined operations must leave the order unchanged. Handoffs must leave orders unchanged.

The raw model comparison uses the same MLX base, tokenizer, policy, tools, greedy decoding and 160-token generation limit. Each variant gets one validation prompt warm-up. The base runs first, then the selected adapter is loaded. No JSON grammar, JSON repair, oracle action fallback or hidden retry is used in this comparison. Tool errors remain visible, consume the shared eight-call budget and are retained in the report.

**Grading correction:** During the first base-model pass, inspection showed that the initial final-state grader could reward a guessed missing cancellation reason or payment method. The already-frozen policy required asking the customer. [grade.py](grade.py) therefore regrades both groups' saved trajectories uniformly for required clarification and identity/order inspection, retaining the original scores. No model outputs are regenerated; no prompt, checkpoint, training data, test case or tool behavior is changed. [evaluation.json](../reports/evaluation.json) preserves the original run, and [evaluation-graded.json](../reports/evaluation-graded.json) is the primary corrected evaluation. This evaluator repair is disclosed separately from weight-training gains.

Task success requires the correct outcome **and** necessary information acquisition. Schema validity and reference-next-action agreement are separate metrics; an alternative valid question can differ from the demonstrator without necessarily failing the business task. Confirmation preservation and transactional safety are enforced by the shared server code, not attributed to learned model behavior.

One greedy trial is run per task. No pass^3 reliability estimate, statistical significance, real support-ticket reduction, customer satisfaction, production SLA or production cost saving is claimed. End-to-end task timing includes model/tool execution and scripted replies; phase order, compilation and output lengths can affect it. All failures are retained. Further tuning on these failures requires a new held-out test set.

## Ollama is a separate integration path

The existing Ollama `qwen3:4b` GGUF remains unchanged. A dedicated `resolvedesk-qwen3:4b` profile closes its empty thinking prefix and reuses the same weight blobs; this is a template setting, not training. Free-form generation still failed to produce JSON within budget. The app uses Ollama's native `/api/chat` **JSON Schema** decoding with application-level action dispatch; this is not native Ollama `tool_calls`.

Both failed attempts and the successful validation smoke are saved. The successful smoke reached a return proposal without changing the order; a browser exchange case reached a correct $5 proposal and committed only after explicit confirmation. Ollama's constrained GGUF integration is not substituted for the unconstrained MLX base in the paired training comparison.

The trained adapter is deployed in a separate local MLX service on port 8772, selectable in the same UI. Exporting or merging the adapter into Ollama/GGUF has not been verified.

## Reproduce in a clean experiment copy

The frozen labels are already supplied. Do not rerun `prepare.py` to overwrite them. Keep the reference reports, then use a fresh checkout/directory with no `data/finetuning` run folder:

```bash
uv sync --frozen --group viz
uv sync --frozen --project training
mkdir reference
mv reports/training.json reports/training-protocol.json reports/evaluation*.json reports/token-audit.json reports/release-gate.json reference/
training/.venv/bin/python -m training.provision --download
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
training/.venv/bin/python -m training.audit
training/.venv/bin/python -m training.train
cp data/finetuning/retail-qlora-v1/run/protocol.json reports/training-protocol.json
training/.venv/bin/python -m training.evaluate
uv run python -m training.grade
uv run python -m training.release_gate
MPLCONFIGDIR="$PWD/.mpl-cache" uv run --group viz python -m training.visualize
```

Downloading uses the fixed public Hugging Face revision without an HF token. Alternatively pass `--from-local /path/to/original/base` to `training.provision`. It rejects file drift instead of rewriting expected digests. Frozen inputs, complete evaluation reports and existing training directories are protected against silent reuse/overwrite. Interrupted raw evaluation can resume only when its recorded source/report identities still match. Different hardware/library versions may produce different floating-point outputs, weights or timings; retain new reports rather than replacing reference results.

Start trained inference after completing evaluation:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.serve
```

Then run the app in another terminal: `uv run resolvedesk serve --port 8771`. All listeners bind to loopback. The app defaults to Ollama and verifies available local models without automatic downloads or cloud fallback.

Official references: [MLX-LM LoRA/QLoRA](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md), [Qwen3 MLX base](https://huggingface.co/Qwen/Qwen3-4B-MLX-4bit), [Ollama chat API](https://docs.ollama.com/api/chat), [τ-bench policy attribution](../third_party/tau-bench/README.md).
