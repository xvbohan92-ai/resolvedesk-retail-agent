"""Explicit new-round resource authorization; importing this module allocates nothing."""

import json
import math
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from training.common import ROOT, sha

RECEIPT = ROOT / "reports/v8/resource-authorization.json"


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("An explicit UTC timestamp is required")
    return parsed


def verify_resource_receipt(path=None, *, active=True, now=None):
    path = Path(path or RECEIPT)
    if not path.is_file() or path.is_symlink():
        raise PermissionError("New V8 cloud execution needs its own explicit resource approval")
    receipt = json.loads(path.read_text())
    if receipt.get("round") != "v8" or receipt.get("explicit_human_approval") is not True:
        raise PermissionError("The completed run002 allowance cannot authorize this new round")
    cap = receipt.get("cap_compute_units")
    if (
        isinstance(cap, bool)
        or not isinstance(cap, (int, float))
        or not math.isfinite(cap)
        or not 2 < cap <= 10
    ):
        raise PermissionError("Use an expressly approved ceiling above the reserve and at most10 units")
    if receipt.get("purchases_allowed") is not False or receipt.get("renewals_allowed") is not False:
        raise PermissionError("No purchase or renewal is authorized")
    if receipt.get("gpu_type") != "T4" or receipt.get("google_drive_mount_allowed") is not False:
        raise PermissionError("This run permits only its T4 runtime with no Drive mount")
    if not receipt.get("approval_text", "").strip():
        raise PermissionError("Missing direct human approval text")
    scope = receipt.get("permitted_work")
    if scope != ["discarded_smoke4", "fixed_final1344", "validation48", "three_arm_test144"]:
        raise PermissionError("The approved work scope differs from the fixed new-round plan")
    plan = ROOT / "reports/v8/pre-inference-plan.json"
    if not plan.is_file() or receipt.get("analysis_plan_sha256") != sha(plan):
        raise ValueError("Resource receipt does not bind the actual new-round plan")
    start, deadline, release = map(
        utc,
        [
            receipt["allocated_at_utc"],
            receipt["process_deadline_utc"],
            receipt["runtime_release_deadline_utc"],
        ],
    )
    if not start < deadline < release or (release - deadline).total_seconds() > 600:
        raise ValueError("The runtime release must follow the hard process deadline within10 minutes")
    balance, reserve, rate = (
        receipt.get("starting_balance_units"),
        receipt.get("reserve_units"),
        receipt.get("observed_units_per_hour"),
    )
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in (balance, reserve, rate)
    ):
        raise ValueError("Actual initial UI observations are required")
    if balance < cap or reserve < 2 or reserve >= cap or rate <= 0:
        raise ValueError("Insufficient initial balance, reserve or observed rate")
    projected = (release - start).total_seconds() / 3600 * rate
    if projected > cap - reserve:
        raise ValueError("Observed-rate projection exceeds the budget after reserve")
    if receipt.get("operator_stop_balance_units") != balance - cap + reserve:
        raise ValueError("Operator early-stop threshold changed")
    if active and not start <= (now or datetime.now(UTC)) < deadline:
        raise PermissionError("New-round execution window has not started or has expired")
    return receipt


def arm_process_deadline(receipt, phase, *, stop=os._exit):
    """Stopping a Python process does NOT release a billed VM; the notebook must arm that too."""
    delay = (utc(receipt["process_deadline_utc"]) - datetime.now(UTC)).total_seconds()
    if delay <= 0:
        raise PermissionError("Process deadline has passed")
    destination = ROOT / "reports/v8/cuda" / f"{phase}-deadline-stop.json"

    def expired():
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(
                    {
                        "status": "deadline_exhausted",
                        "phase": phase,
                        "stopped_at_utc": datetime.now(UTC).isoformat(),
                        "resource_receipt_sha256": sha(RECEIPT),
                        "automatic_retry_allowed": False,
                        "billing_stopped": False,
                        "instruction": (
                            "Save all raw evidence and release only this Colab runtime immediately."
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            # A full disk or failed audit write must never disable the hard stop.
            stop(124)

    timer = threading.Timer(delay, expired)
    timer.daemon = True
    timer.start()
    return timer
