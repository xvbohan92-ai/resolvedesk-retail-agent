# ResolveDesk round 2: repetition recovery and exchange decisions

This is a separate experiment, starting again from the pinned official Qwen3-4B 4-bit MLX base. It does not continue the rejected v1 adapter. All v1 inputs, weights, raw traces, corrected scores and figures remain intact.

## Engineering changes

`resolve_desk/workflow.py` adds a progress-aware controller without importing task answers. Model input contains plain customer messages, de-duplicated successful read results, recent errors and available tool names. Allowed tools depend on inspected identity, order status and catalog evidence; they do not select the customer's operation or target variant.

- Repeating a successful, unchanged identity/order/catalog lookup is blocked before the backend tool executes. The observed cached result and a recovery instruction remain visible. A second blocked repetition closes the model loop with a handoff. Failed lookups can be corrected and retried. A changed order revision permits an actual fresh order read.
- Catalog inspection returns all different same-product variants, their availability and exact price differences against the inspected item. The model must still match the requested option and copy the correct item ID. The controller does not select the target, replace exchange with return, or generate a successful proposal.
- Proposed exchanges require catalog inspection, and status gates reject operations unavailable at the current stage. The original ownership, payment, inventory, confirmation, idempotency and transaction validators still apply.

These changes alter the system prompt, visible state representation and tool behavior. Their effect is measured separately from the subsequent parameter updates. The read guard is engineering behavior, not learned model compliance. The prototype has a frozen catalog per sandbox session; this is not a production inventory-cache design.

## New data and predeclared method

The first experiment's test failures informed this redesign, so its 27 tasks are treated as development evidence. Round 2 freezes **112 training tasks, 28 validation tasks and 42 new test tasks** across 14 families before any v2 model test output is generated. It includes 864/108/162 reference decision snapshots. The training set contains clean and injected-error recovery trajectories; only corrective/reference decisions are supervised, not the deliberately injected bad actions.

The original nine families are extended with missing exchange option, missing exchange payment, gift-card exchanges, unavailable variants without return authorization, and insufficient gift-card balance. Catalog ordering is shuffled, IDs and price draws differ between splits, opening templates are disjoint, and requested colors are split-specific. However, task families, author, policy, schema and per-family request cores are shared. All cases are synthetic, with one order, one product and three variants. Demonstrations come from a programmatic reference, not human annotations or real retail transcripts.

There are **21 exchange-related test tasks**, including authorized fallback and handoff, of which **12 require fulfilling an available exchange**. The tests use both positive and negative price differences. Ten actionable test cases are configured to decline confirmation; handoff tasks never reach confirmation. A correct proposed outcome and all required information acquisition are graded; a correct guess cannot replace a missing customer clarification. An allowed but unrequested variant fails even if the transaction itself succeeds.

Training uses 4-bit QLoRA in the last eight layers, rank 8, scale 16, dropout 0, batch 2, learning rate 0.00005, seed 20261027 and 280 optimizer updates. The base model/revision and all nine local model/tokenizer/license hashes are unchanged. Completion-only loss and no-thinking chat prefixes are audited. The longest training snapshot is 1,550 tokens, below the 3,072-token limit; no example is truncated.

Checkpoint selection uses the minimum full validation completion loss among positive-update checkpoints, including an explicit final evaluation. The MLX-LM callbacks evaluate before the next update; callback numbers record the number of completed updates. Neither test outputs nor end-task test success select the checkpoint.

## Three-arm evaluation

| Arm | Weights | Workflow |
| --- | --- | --- |
| `legacy_base` | Pinned MLX base, no adapter | Original v1 prompt, tool controller and validator |
| `workflow_base` | Same MLX base, no adapter | V2 progress prompt, read guard and enriched catalog |
| `workflow_qlora_v2` | Same base + validation-selected v2 adapter | Exactly the same V2 workflow |

The first comparison measures the combined workflow/prompt/tool changes. The second compares trained and untrained weights with the workflow fixed. All arms use the same tokenizer, greedy unconstrained generation, 180-token limit, eight-model-call budget, fixed customer replies and confirmation decisions. No JSON repair, oracle action fallback or cloud calls are used. Successful backend reads repeated within the task are counted separately from blocked repeated attempts. The base arms are interleaved by case; the trained arm runs after them. Local serial latency is descriptive, with no production-speed or cost-savings claim.

Customer replies are keyed to the model's requested tool field, rather than generated by a natural-language user simulator. Clarification grading checks acquisition of the required field; it does not independently judge the wording of every question. The suite does not measure human customer satisfaction, multi-item orders or live inventory changes.

The supplementary analysis also counts every identical valid action after its first occurrence in a task, including repeated rejected actions and clarification requests. It is broader than the completed-read cache metric. Both are exposed so zero repeated backend reads cannot be mistaken for zero model repetition. This derived diagnostic does not change task scoring or the predeclared promotion rule.

The promotion rule is frozen in `config.json` before training and test inference: the candidate must improve overall and exchange-related task success over `workflow_base`, preserve confirmation on every case, introduce no paired non-exchange regressions and not increase rejected writes. This is a conservative prototype rule, not a statistical test or proof of production readiness. A candidate failing the rule remains an explicit local experiment. Ollama has different GGUF weights/JSON-Schema decoding and is verified as a separate integration, never substituted as the causal MLX baseline.

## Reproduce and inspect

The published `datasets/manifest.json` binds the controller, grading, runner, config and data to their hashes. `reports/v2/` holds actual training, raw per-case model/tool trajectories and failures. `docs/figures/v2/` holds source-backed export figures, a per-case CSV and report/figure hashes. No model weights, databases or credentials are committed.

Use the existing project/training lockfiles and the [base model provisioning instructions](../README.md). In a fresh source checkout, preserve the supplied measurements before reproducing:

```bash
mv reports/v2 reports/v2-reference
mkdir reports/v2
mv docs/figures/v2 docs/figures/v2-reference
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v2.audit
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v2.train
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v2.evaluate
MPLCONFIGDIR=.mpl-cache uv run --group viz python -m training.v2.visualize
```

Training refuses to replace an existing run directory; move it aside when reproducing on a machine with a previous run. Evaluation refuses to overwrite a completed run and verifies identities when resuming a partial run. Do not rerun data preparation over frozen inputs or tune again on these test results. Create a new experiment/test set for a further optimization cycle.

To serve the actual selected adapter on loopback after training:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false training/.venv/bin/python -m training.v2.serve
uv run resolvedesk serve --port 8771
```

The adapter service binds to port 8772 and includes the selected weights' hash in every response. Select **Experimental QLoRA v2 · MLX**. Ollama remains the default separate integration; the MLX adapter is not merged into GGUF. All payments, customer identities and confirmations remain simulated.
