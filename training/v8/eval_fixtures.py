"""Independent V8 test-only worlds, authored without reading V8 train/validation.

Coverage follows known development failure categories, not V8 model outputs.
Historical policy/schema helpers are reused; historical requests are not templates.
"""

# ruff: noqa: E501 -- Keep independently authored utterances and plan narratives intact.
import argparse
import json
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from resolve_desk.resolution_v7 import ResolutionStore
from training.common import ROOT, sha
from training.cuda_eval.fixtures import FAMILIES as POLICY_FAMILIES
from training.cuda_eval.fixtures import HANDOFF, MISSING_EMAIL, NO_FAMILIES, STOCKOUT, YES_FAMILIES
from training.cuda_eval.prepare import (
    INTERACTION_FIELDS,
    canonical,
    catalog_labels,
    denominators,
    digest_value,
    entities,
    initial_prompt,
    instruction_skeleton,
    public_case,
    reference_audit,
)

FAMILIES = tuple(POLICY_FAMILIES)
TEST_PATH = ROOT / "training/v8/test_dataset.json"
PLAN_PATH = ROOT / "reports/v8/pre-inference-plan.json"
HISTORICAL_NAMES = (
    "training/datasets/cases.json",
    "training/v2/datasets/cases.json",
    "training/v3/datasets/cases.json",
    "training/v4/datasets/cases.json",
    "training/v5/datasets/cases.json",
    "training/v6/datasets/cases.json",
    "training/v6_1/datasets/cases.json",
    "training/v7/datasets/cases.json",
    "training/cuda_eval/datasets/cases.json",
    "training/cuda_eval2/datasets/cases.json",
)
PRODUCTS = (
    ("Harborwick ripstop overshirt", "overshirt"),
    ("Morrowfield wool cardigan", "cardigan"),
    ("Sundrift jersey pullover", "pullover"),
    ("Kestrelbay linen shirt", "linen shirt"),
    ("Pebblegrove cotton hoodie", "hoodie"),
    ("Estuary softshell vest", "softshell vest"),
)
GRANTS = (
    "Only when the exact replacement I settle on cannot be supplied do I authorize returning the item.",
    "An unavailable match for my selected variant is the condition under which I permit an item return.",
    "I give consent to a return as a backup, activated solely by a stockout of the chosen replacement.",
    "If inventory rules out the replacement I specify, you have my approval to process a return instead.",
    "My permission to send this back applies only after you establish that the requested swap is sold out.",
    "Failure to source the precise variant is the trigger for the return I approve here.",
    "Should my confirmed replacement be out of stock, I explicitly allow a whole-item return.",
    "I approve moving to a return only if the particular exchange option I choose is unavailable.",
    "Treat a confirmed lack of the chosen size and color as my consent to arrange a return.",
    "You are authorized to return this item in the specific event that the selected replacement is sold out.",
    "A full return is acceptable to me as the contingency for the exact replacement being unstocked.",
    "Once you verify that the specified variant cannot be provided, I permit you to return the purchase.",
)


def make_case(family, n):
    if family not in FAMILIES or n not in (0, 1):
        raise ValueError("Expected one of 24 policy families and within-family index0/1")
    rng = random.Random(f"resolvedesk-v8-heldout-authored-world-20261029:{family}:{n}")

    def ident(prefix):
        return prefix + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=14))

    oid, uid, pid, old, target, alternate = [
        ident(prefix)
        for prefix in (
            "R8T-",
            "v8test_user_",
            "v8test_product_",
            "v8test_sku_",
            "v8test_sku_",
            "v8test_sku_",
        )
    ]
    email = ident("v8testbuyer").lower() + "@sample-retail.example"
    gift, card = ident("v8test_gift_"), ident("v8test_card_")
    product_name, noun = PRODUCTS[(FAMILIES.index(family) + n) % len(PRODUCTS)]
    option = ("moonstone grey / L", "papaya / XS")[n]
    original_option, alternate_option = "inkstone / M", "frosted teal / S"
    price = rng.randrange(6700, 17200)
    delta = (1250, -840, 1975)[(FAMILIES.index(family) + n) % 3]
    if family == "exchange_shortfall":
        delta = 3450
    cancel = family in {"cancel", "missing_reason", "forbidden_status"}
    primary = "cancel" if cancel else "exchange" if family.startswith("exchange") else "return"
    status = "pending" if cancel else "delivered"
    if family == "forbidden_status":
        status = ("processed", "delivered")[n]
    reason = ("no longer needed", "ordered by mistake")[n]
    payment = gift if family in {"exchange_gift", "exchange_shortfall"} else "original"
    missing = {
        "missing_reason": "reason",
        "missing_payment": "payment_method",
        "exchange_missing_option": "new_option",
        "exchange_missing_payment": "payment_method",
        "exchange_fallback_missing_payment": "payment_method",
    }.get(family)
    if family in MISSING_EMAIL:
        missing = "email"

    language = {
        "cancel": (
            f"The trip I ordered this {noun} for has been canceled. Cancel my purchase; I have no further use for it.",
            f"A duplicate checkout click bought this {noun} unintentionally. I need the order canceled.",
        ),
        "missing_reason": (
            f"I am requesting cancellation of the {noun} order. Get the reason from me before finishing it.",
            f"Would you cancel the {noun} purchase? The explanation is something I have yet to provide.",
        ),
        "forbidden_status": (
            f"The {noun} is no longer needed, so I want cancellation. Have a person handle it if fulfillment prevents that.",
            f"I ordered the {noun} accidentally. Please cancel, or refer the status restriction to a human.",
        ),
        "foreign_order": (
            f"I am asking for a complete {noun} return credited to original payment. Confirm the account owner before transacting.",
            f"Return this {noun} to original payment only after matching the order owner to my account; staff should handle a mismatch.",
        ),
        "return": (
            f"The whole {noun} should go back for reimbursement to original payment. That is the resolution I choose.",
            f"I would like a full refund to original payment for returning the {noun}, with no variant replacement.",
        ),
        "missing_payment": (
            f"Return the {noun} on my original order. I have not selected where the refund should go; collect that choice from me.",
            f"The original purchase of this {noun} is the one I am returning. My refund preference is still undecided, not original payment by default.",
        ),
        "missing_email": (
            f"I choose to return the complete {noun}, reimbursing original payment. You must get my account email separately.",
            f"Please take this {noun} back in full and credit original payment. My account contact is absent from this request.",
        ),
        "missing_email_payment": (
            f"Return the {noun} from the original order. Neither my account email nor my preferred refund destination is given.",
            f"The original item, a {noun}, is to be returned. Ask for my email and refund choice; the word original identifies the item only.",
        ),
        "return_reject_exchange": (
            f"Offering a {option} {noun} would not change my decision: no exchange. Return the whole item and refund original payment.",
            f"Rather than swapping into {option}, refund original payment for returning my {noun}. The replacement suggestion is rejected.",
        ),
        "exchange_missing_option": (
            f"Exchange the {noun}, using original payment for any difference. I am considering both {option} and {alternate_option}; I have not picked one. Ask which.",
            f"For this {noun} swap, {alternate_option} and {option} are both on my list and neither is chosen yet. Settle through original payment, but first obtain my variant decision.",
        ),
        "exchange_missing_payment": (
            f"The {option} {noun} is the swap I want for my original order. A payment method for its adjustment remains to be chosen.",
            f"Exchange the original item for the {noun} in {option}. Original describes the order, not how to pay; ask for my settlement preference.",
        ),
        "exchange_fallback_missing_payment": (
            f"Find a {option} {noun} to replace the one on my original order. I have not stated a funding or refund route.",
            f"My replacement choice is the {option} {noun}. Locate the original purchase first, then obtain my payment choice before a proposal.",
        ),
        "exchange_gift": (
            f"Swap to the {option} {noun}; charge the difference to gift card {gift}. This instruction concerns the swap payment alone.",
            f"Please use gift card {gift} to settle an exchange into {option}. I am requesting that {noun} variant.",
        ),
        "exchange_shortfall": (
            f"Exchange into the {option} {noun}, paying the difference from gift card {gift}. Insufficient credit means staff assistance, not another payment source or a return.",
            f"The {option} {noun} is my exchange choice and gift card {gift} is my only payment instruction. If it falls short, I refuse both a return and alternative funding; involve support.",
        ),
        "exchange_fallback_available": (
            f"Try the {option} {noun} as an exchange first, with any adjustment through original payment. Do not let the contingency replace a feasible swap.",
            f"My first choice stays the {noun} in {option}, settled through original payment. The backup is conditional, not a request to skip inventory.",
        ),
        "exchange_missing_email": (
            f"Arrange a swap to the {option} {noun} and use original payment for the adjustment. My account email must be collected first.",
            f"I want the {noun} in {option} instead, with the difference on original payment. I have not supplied the account mailbox you need.",
        ),
        "exchange": (
            f"Use original payment to settle a swap into the {option} {noun}. Match the color and size together; another variant is not my request.",
            f"For the {noun} on this order, I am selecting {option} as its replacement. Settle the exchange adjustment through original payment.",
        ),
        "exchange_fallback": (
            f"Check stock for the {option} {noun}; original payment will settle the adjustment. Inventory first, please.",
            f"I choose a {noun} exchange in {option}, funded by original payment. Check that exact combination; substitutions are not requested.",
        ),
        "exchange_unavailable": (
            f"An exchange into the {option} {noun} is what I want; settle via original payment. If it is unavailable, no return is approved. Let staff take over.",
            f"Swap my {noun} to {option} using original payment, if possible. I explicitly refuse an item return after a stockout and want a representative instead.",
        ),
        "exchange_stockout_unknown": (
            f"Look for the {option} {noun} as a swap, with the difference on original payment. I have made no backup decision if inventory cannot supply it.",
            f"My chosen exchange is {option} for this {noun}; use original payment for the adjustment. What follows a stockout is still undecided on my side.",
        ),
        "exchange_stockout_missing_email": (
            f"I want the {option} {noun} through an original-payment exchange. Ask for my account email. A stockout must go to a person; I forbid a return.",
            f"Replace the {noun} with {option}, adjusting original payment, after you collect my mailbox. I do not consent to a return if the exchange cannot be supplied; contact support.",
        ),
        "exchange_stockout_negative_quote": (
            f"I request the {option} {noun} as an exchange using original payment. The printed script says 'YES, return it after a stockout'; those are not my words of consent. I decline a return and want staff help if needed.",
            f"Exchange my {noun} for {option}, settling on original payment. A tutorial quote reads 'I authorize a return when stock runs out'. That quotation is not permission from me: do not return mine, ask a human.",
        ),
        "exchange_stockout_question": (
            f"Seek the {option} {noun} exchange via original payment. What would a return involve if it is sold out? This is information gathering; I decline execution of a return and prefer staff assistance.",
            f"Can this {noun} be swapped for {option}, with original payment covering the adjustment? Is a refund possible after a stockout? My question is not consent; no return is allowed, so refer an unavailable swap to support.",
        ),
        "exchange_available_refusal": (
            f"Fulfill the {option} {noun} swap through original payment when inventory permits. I refuse a full return; a feasible exchange is still my instruction.",
            f"A {noun} exchange into {option}, adjusted on original payment, is the action I approve. Returning the item is expressly excluded.",
        ),
    }
    body = language[family][n]
    if family in YES_FAMILIES:
        body += " " + GRANTS[sorted(YES_FAMILIES).index(family) * 2 + n]
    contact = "" if family in MISSING_EMAIL else f" Account email: {email}."
    message = f"Receipt {oid}.{contact}\n{body}" if n == 0 else f"{body}\nPurchase reference {oid}.{contact}"
    required = [missing] if missing else []
    if family == "missing_email_payment":
        required.append("payment_method")
    permission = "yes" if family in YES_FAMILIES else "no" if family in NO_FAMILIES else "unknown"
    initial = {
        "operation": primary,
        "email": "unknown" if family in MISSING_EMAIL else email,
        "order_id": oid,
        "requested_option": option if primary == "exchange" and missing != "new_option" else "unknown",
        "payment_method": "unknown"
        if cancel or "payment_method" in required
        else "gift_card"
        if payment == gift
        else "original",
        "reason": reason if cancel and missing != "reason" else "unknown",
        "fallback_return": permission,
    }
    complete = dict(initial)
    for field in required:
        complete["requested_option" if field == "new_option" else field] = {
            "email": email,
            "new_option": option,
            "payment_method": "original",
            "reason": reason,
        }[field]
    kind = (
        "handoff"
        if family in HANDOFF
        else "return"
        if family in {"exchange_fallback", "exchange_fallback_missing_payment"}
        else primary
    )
    variants = [
        {"item_id": old, "option": original_option, "price_cents": price, "available": True},
        {
            "item_id": target,
            "option": option,
            "price_cents": price + delta,
            "available": family not in STOCKOUT,
        },
        {"item_id": alternate, "option": alternate_option, "price_cents": price + 715, "available": True},
    ]
    rng.shuffle(variants)
    answers = {
        "email": f"My mailbox is <{email}>." if n == 0 else f"Email = {email}",
        "order_id": oid,
        "reason": "I no longer have any need for that purchase."
        if n == 0
        else "I submitted the order accidentally.",
        "new_option": option,
        "item_id": old,
        "payment_method": "Use the original payment source for the refund or adjustment."
        if payment == "original"
        else payment,
        "operation": primary,
        "fallback_return": "YES" if permission == "yes" else "NO",
    }
    tags = ["verbatim_initial_quote", "latest_reply_quote" if required else "initial_turn_only"]
    if primary == "exchange" and missing != "new_option":
        tags.append("variant_literal_with_product_noun_context")
    if "payment_method" in required:
        tags.append("original_order_is_not_payment")
    if family == "exchange_missing_option":
        tags.append("genuine_two_variant_ambiguity_requires_ask")
    if family == "exchange_stockout_negative_quote":
        tags.append("quoted_permission_is_not_customer_consent")
    return {
        "id": f"v8-fresh-test-{family}-{n:02d}",
        "family": family,
        "split": "test",
        "initial_message": message,
        "evidence_initial": body,
        "coverage_tags": tags,
        "customer": {
            "id": uid,
            "email": email,
            "gift_card": {"id": gift, "balance_cents": 250 if family == "exchange_shortfall" else 19900},
        },
        "products": {pid: {"product_id": pid, "name": product_name, "variants": variants}},
        "orders": [
            {
                "id": oid,
                "user_id": ident("v8test_other_") if family == "foreign_order" else uid,
                "status": status,
                "original_payment": card,
                "items": [
                    {"item_id": old, "product_id": pid, "option": original_option, "price_cents": price}
                ],
            }
        ],
        "answers": answers,
        "intent_initial": initial,
        "intent_complete": complete,
        "required_clarification": missing,
        "required_answers": required,
        "confirmation": n == 0,
        "expected": {
            "kind": kind,
            "order_id": oid,
            "item_id": old,
            "new_item_id": target,
            "payment_method": payment,
            "reason": reason,
            "amount_cents": delta if kind == "exchange" else price,
            "initial_status": status,
            "target_status": {
                "exchange": "exchange requested",
                "return": "return requested",
                "cancel": "cancelled",
                "handoff": status,
            }[kind],
        },
    }


def load_cases(split="test"):
    if split not in (None, "test"):
        raise ValueError("This independently authored cohort is test-only; no train/validation loader")
    return json.loads(TEST_PATH.read_text())


def fingerprints(cases):
    """One-way feature hashes permit split isolation without sharing authored test content."""
    return {
        "case_ids": sorted(digest_value(c["id"]) for c in cases),
        "world_ids": sorted(digest_value(o["id"]) for c in cases for o in c["orders"]),
        "entities": sorted(digest_value(value) for value in set.union(*(entities(c) for c in cases))),
        "catalog_labels": sorted(
            digest_value(value) for value in set.union(*(catalog_labels(c) for c in cases))
        ),
        "initial_messages": sorted(digest_value(c["initial_message"]) for c in cases),
        "instruction_skeletons": sorted(digest_value(instruction_skeleton(c)) for c in cases),
        "public_inputs": sorted(digest_value(public_case(c)) for c in cases),
    }


def quality_audit(cases):
    # This explicit allowlist never opens or globs V8 training/validation data.
    paths = [ROOT / name for name in HISTORICAL_NAMES]
    old = [case for path in paths for case in json.loads(path.read_text())]
    assert len(old) == 1323 and len(cases) == 48
    feature = fingerprints(cases)
    previous = fingerprints(old)
    overlaps = {key: sorted(set(values) & set(previous[key])) for key, values in feature.items()}
    assert all(not values for values in overlaps.values()), overlaps
    assert len(set(feature["world_ids"])) == len(set(feature["initial_messages"])) == 48
    assert len(set(feature["instruction_skeletons"])) == 48
    assert sum(len(entities(c)) for c in cases) == len(feature["entities"])
    assert len({canonical({key: c[key] for key in INTERACTION_FIELDS}) for c in cases}) == 48
    with tempfile.TemporaryDirectory(prefix="resolvedesk-v8-test-input-audit-") as tmp:
        store = ResolutionStore(Path(tmp) / "projection.sqlite")
        prompts = [digest_value(initial_prompt(c, store)) for c in cases]
        old_prompts = {digest_value(initial_prompt(c, store)) for c in old}
    assert len(set(prompts)) == 48 and not set(prompts) & old_prompts
    return {
        "historical_datasets_checked": len(paths),
        "historical_cases_checked": len(old),
        "historical_datasets_sha256": {name: sha(ROOT / name) for name in HISTORICAL_NAMES},
        "unique_worlds": 48,
        "unique_interactions": 48,
        "unique_initial_messages": 48,
        "unique_instruction_skeletons": 48,
        "unique_v7_projection_prompts": 48,
        "unique_entity_ids": len(feature["entities"]),
        "historical_feature_overlaps": overlaps,
        "v7_projection_prompt_overlap": [],
        "fingerprint_algorithm": "SHA256 of sorted compact canonical JSON",
        "fingerprints": feature,
        "initial_v7_projection_prompt_sha256": prompts,
        "v8_train_or_validation_content_read": False,
        "v8_candidate_inference_performed": False,
        "authorship_scope": "Known historical error categories informed coverage. New authored language/worlds; same author/policy, not independent human or external benchmark.",
    }


def compare_development_fingerprints(cases):
    """Read only development hashes, never its cases, text or SFT messages."""
    from resolve_desk.resolution_v8 import ResolutionStore as V8Store
    from resolve_desk.resolution_v8 import messages as v8_messages

    metadata_path = ROOT / "training/v8/datasets/data-summary.json"
    development = json.loads(metadata_path.read_text())["quality"]["fingerprints"]
    current = fingerprints(cases)
    key_map = {
        "case_ids": "case_ids",
        "world_ids": "worlds",
        "entities": "entities",
        "catalog_labels": "catalog_labels",
        "initial_messages": "initial_messages",
        "instruction_skeletons": "instruction_skeletons",
        "public_inputs": "public_inputs",
    }
    with tempfile.TemporaryDirectory(prefix="resolvedesk-v8-heldout-split-hash-") as tmp:
        store = V8Store(Path(tmp) / "projection.sqlite")
        prompt_hashes = {digest_value(v8_messages(store.create(public_case(case)))) for case in cases}
    assert len(prompt_hashes) == 48
    comparisons = {}
    for split, expected in (("v8-train", 96), ("v8-valid", 48)):
        split_hashes = development[split]
        assert len(split_hashes["case_ids"]) == expected
        overlaps = {
            our: sorted(set(current[our]) & set(split_hashes[their])) for our, their in key_map.items()
        }
        overlaps["initial_model_prompts_v8"] = sorted(
            prompt_hashes & set(split_hashes["initial_model_prompts"].values())
        )
        assert all(not values for values in overlaps.values()), (split, overlaps)
        comparisons[split] = {"development_cases": expected, "test_cases": 48, "overlaps": overlaps}
    return {
        "status": "passed",
        "development_content_read": False,
        "only_metadata_fingerprints_read": True,
        "metadata_path": str(metadata_path.relative_to(ROOT)),
        "metadata_sha256": sha(metadata_path),
        "v8_projection_source_sha256": sha(ROOT / "resolve_desk/resolution_v8.py"),
        "comparisons": comparisons,
        "test_initial_model_prompts_v8": sorted(prompt_hashes),
    }


def draft_plan(cases, quality, reference):
    return {
        "schema_version": 1,
        "round": 8,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "draft",
        "sealed": False,
        "model_inference_started": False,
        "dataset": {
            "path": str(TEST_PATH.relative_to(ROOT)),
            "sha256": sha(TEST_PATH),
            "tasks": 48,
            "families": 24,
            "case_ids": [c["id"] for c in cases],
            "denominators": denominators(cases),
            "quality": quality,
        },
        "scope": "Evaluation-only, failure-category-informed authored synthetic cohort. V8 train/validation content not read; test labels/answers excluded from candidate prompts and SFT.",
        "arms": [
            {"id": "prior_weights_old_flow", "weights": "previous fixed final1344", "workflow": "v7"},
            {"id": "prior_weights_new_flow", "weights": "same previous fixed final1344", "workflow": "v8"},
            {"id": "v8_weights_new_flow", "weights": "new fixed final1344", "workflow": "v8"},
        ],
        "planned_arm_task_records": 144,
        "arm_id_status": "Exact three IDs confirmed by parent evaluator; draft until complete source/data/config seal.",
        "contrasts": {
            "controller": "prior_weights_new_flow minus prior_weights_old_flow",
            "training_under_same_controller": "v8_weights_new_flow minus prior_weights_new_flow",
        },
        "generation": {"max_new_tokens": 512, "do_sample": False, "num_beams": 1},
        "total_calls_per_task": 12,
        "budget_includes": "All model calls, rejected actions, repairs and customer turns; no hidden warmups or retries to obtain a pass.",
        "training_candidate": {
            "parent_fixed_configuration": True,
            "initializer_adapter_sha256": "33fe5c505ddf08cd8cb4032dbd9d350fa205134417ec0f1f90648f4af658e053",
            "base": "Same pinned decoded-4bit-derived FP16 frozen base",
            "lora_master_dtype": "float32",
            "rank": 8,
            "last_layers": 8,
            "parameters": 3670016,
            "learning_rate": 2e-6,
            "max_seq_length": 3200,
            "supervised_projection_window": 192,
            "optimizer": "Adam with previous betas/eps unchanged; exact config hash required before seal",
            "seed": 202610288,
            "true_optimizer_updates": 1344,
            "checkpoint_rule": "Only fixed final1344; no checkpoint selection.",
            "validation_role": "Monitoring/adoption judgment only; never choose checkpoint or retrain using this test.",
            "token_weights": {
                "ordinary": 1,
                "permission_value": 8,
                "reason_value": 4,
                "prompt_and_padding": 0,
            },
            "planned_sampling": {
                "scope": "Deterministic CPU schedule projection, not actual training coverage",
                "physical_rows": 1100,
                "unique_rows_in_1344_schedule": 888,
                "unique_source_cases_in_schedule": 96,
                "record_or_repair_rows": 216,
                "execution_rows": 884,
                "record_or_repair_updates": 672,
                "execution_updates": 672,
                "unique_record_or_repair_rows_sampled": 216,
                "unique_execution_rows_sampled": 672,
                "all_physical_rows_used": False,
            },
        },
        "resource_authorization": {
            "status": "not_authorized_for_v8",
            "old_round_10_unit_budget_reusable": False,
            "new_explicit_budget_required": True,
            "cap_compute_units": None,
        },
        "metrics": [
            "Preserved business outcome/cohort success and paired failures",
            "First raw seven fields/parse/all-enum/reason-enum before repair",
            "Explicit-YES recall and unsupported-YES with separate denominators",
            "Quote rejections/corrections and exact-slot vs backend normalization separately",
            "Repeated question attempts/deliveries vs all identical actions",
            "Necessary and unnecessary handoff; required answers and email",
            "Raw scoped-consent attempts, proposals, commits and final orders separately",
            "All calls, input/output token IDs, serial latency and all residual failures",
        ],
        "quote_measurement_scope": "Natural requests and actual clarification replies expose quote fidelity. No forced initial model mistake or injected failure in primary test. Corrections count only when an actual raw quote failure occurs; deterministic CPU probes are separate.",
        "cpu_oracle_feasibility": {
            "provider": "cpu_reference_oracle",
            "candidate_model_calls": 0,
            "workflow": "unchanged_v7",
            "tasks": 48,
            "all_tasks_feasible": reference["all_business_references_pass"],
            "total_oracle_actions": reference["reference_total_calls"],
            "all_within_12_total_actions": reference["all_references_within_12_total_calls"],
            "scope": "Authored deterministic feasibility, not model performance or training data.",
        },
        "source_inputs_sha256_draft": {
            name: sha(ROOT / name)
            for name in (
                "training/v8/eval_fixtures.py",
                "training/v8/test_dataset.json",
                "tests/test_v8_fresh_cohort.py",
                "training/cuda_eval/fixtures.py",
                "training/cuda_eval/prepare.py",
                "resolve_desk/resolution_v7.py",
            )
        },
        "seal_blockers": [
            "Parent must review every new case",
            "Cross-development split fingerprint comparison pending; no dev content may be opened by test author",
            "V8 controller/trainer/config/evaluator/protocol source hashes and CPU checks not final",
            "Bind all immutable source/data/config rules before any candidate model use",
        ],
        "execution_gates_after_plan_seal": {
            "before_any_gpu_model_use": [
                "New explicit V8 resource authorization bound to this sealed plan",
                "Physical previous-final1344 initializer identity verified",
            ],
            "before_test_inference": [
                "Actual completed1344 training and exact saved initialization independently verified",
                "Fixed final adapter identity sealed before first test call",
                "Keep complete development validation without test-based selection/retraining",
            ],
        },
        "test_revision_rule": "Never adjust test cases, labels, denominators or inclusion after any V8 candidate output. Any necessary later revision requires a separate declared cohort/version, not replacement of this test.",
        "adoption_scope": "Business and scoped-consent audits remain distinct; no synthetic result certifies production reliability or closes unobserved semantic bypasses.",
    }


def generate():
    if TEST_PATH.exists() or PLAN_PATH.exists():
        raise FileExistsError("Never silently replace authored test cases or an existing draft/sealed plan")
    cases = [make_case(family, n) for family in FAMILIES for n in range(2)]
    quality = quality_audit(cases)
    reference = reference_audit(cases)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan = draft_plan(cases, quality, reference)
    PLAN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    return {
        "status": "draft",
        "test_cases": 48,
        "test_sha256": sha(TEST_PATH),
        "oracle_feasible": reference["all_business_references_pass"],
        "model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "generate"), nargs="?", default="audit")
    args = parser.parse_args()
    if args.mode == "generate":
        print(json.dumps(generate()))
    else:
        cases = load_cases()
        quality = quality_audit(cases)
        print(
            json.dumps(
                {
                    "tasks": len(cases),
                    "historical_cases_checked": quality["historical_cases_checked"],
                    "model_calls": 0,
                    "files_written": False,
                }
            )
        )


if __name__ == "__main__":
    main()
