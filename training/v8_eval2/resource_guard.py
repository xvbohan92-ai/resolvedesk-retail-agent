"""A new explicitly approved eval-only window within the original cumulative cap.

These are observed-rate/time guards, not an automatic Colab billing meter. Exiting
Python does not release a billed VM; the notebook must independently release it.
"""

import math
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha
from training.cuda_eval.evaluation import write_atomic
from training.v8_eval2.evaluation import REPORTS, RUN_ID, read, verify_identity

RECEIPT = REPORTS / "resource-authorization.json"


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("Explicit UTC required")
    return result


def numeric(value):
    return not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value)


def verify_resource_receipt(path=None, *, active=True, now=None):
    path = Path(path or RECEIPT)
    if not path.is_file() or path.is_symlink():
        raise PermissionError("The ready package is not approval; a new explicit window is required")
    receipt = read(path)
    plan, identity = verify_identity()
    if receipt.get("run_id") != RUN_ID or receipt.get("explicit_human_approval") is not True:
        raise PermissionError("Missing new direct human approval for this saved-weight-only run")
    if not receipt.get("approval_text", "").strip():
        raise PermissionError("The actual specific user approval must be recorded")
    if receipt.get("permitted_work") != ["three_arm_test144_saved_weights_only"]:
        raise PermissionError("No training, gradient smoke, validation rerun or old-call replay is permitted")
    for field in (
        "purchases_allowed",
        "renewals_allowed",
        "google_drive_mount_allowed",
        "automatic_compute_unit_metering",
        "billing_stops_when_process_exits",
    ):
        if receipt.get(field) is not False:
            raise PermissionError("No purchase/renewal/Drive mount or false billing guarantee")
    if receipt.get("gpu_type") != "T4" or receipt.get("official_colab_image") != "2026.07":
        raise PermissionError("Only the reviewed official2026.07 T4 runtime is permitted")
    if receipt.get("analysis_plan_sha256") != sha(REPORTS / "analysis-plan.json"):
        raise ValueError("Window approval binds a different evaluation plan")
    if receipt.get("artifact_identity_sha256") != sha(REPORTS / "artifact-identity.json"):
        raise ValueError("Window approval binds different saved weights")
    if receipt.get("after_adapter_sha256") != identity["after_adapter_sha256"]:
        raise ValueError("The already saved candidate changed")
    policy = plan["resource_policy"]
    for field in (
        "original_resource_receipt_sha256",
        "original_round_initial_balance_units",
        "original_round_cap_compute_units",
        "original_operator_stop_balance_units",
    ):
        if receipt.get(field) != policy[field]:
            raise PermissionError("A new time window cannot reset the original round budget")
    cap = receipt.get("new_window_cap_compute_units")
    balance = receipt.get("starting_balance_units")
    rate = receipt.get("observed_units_per_hour")
    buffer = receipt.get("transfer_release_buffer_compute_units")
    if not all(numeric(v) for v in (cap, balance, rate, buffer)):
        raise ValueError("Finite actual UI balance/rate and approved cap/buffer are required")
    if not 0 < cap <= policy["maximum_new_window_compute_units"] or buffer < 0.25 or buffer >= cap:
        raise ValueError(
            "Window cap is at most3 existing units with at least0.25 reserved for transfer/release"
        )
    floor = policy["original_operator_stop_balance_units"]
    if not floor < balance <= policy["original_round_initial_balance_units"] or rate <= 0:
        raise PermissionError("Actual available original-round headroom is insufficient")
    expected_stop = max(floor, balance - cap + buffer)
    if not numeric(receipt.get("operator_stop_balance_units")) or not math.isclose(
        receipt["operator_stop_balance_units"], expected_stop, abs_tol=1e-9
    ):
        raise ValueError("Operator stop is max(original stop, actual new balance minus cap plus buffer)")
    start, deadline, release = [
        utc(receipt[field])
        for field in ("allocated_at_utc", "process_deadline_utc", "runtime_release_deadline_utc")
    ]
    if not start < deadline < release:
        raise ValueError("An observed allocation and explicit future stop/release deadlines are required")
    if not 300 <= (release - deadline).total_seconds() <= 600:
        raise ValueError("Reserve five to ten minutes between process stop and VM release")
    if (release - start).total_seconds() > policy["maximum_window_hours"] * 3600:
        raise ValueError("The newly approved window cannot exceed3 hours")
    available = min(cap, balance - floor) - buffer
    projected = (release - start).total_seconds() / 3600 * rate
    if available <= 0 or projected > available + 1e-9:
        raise PermissionError(
            "Observed-rate projection exceeds the smaller new/cumulative budget after buffer"
        )
    if not numeric(receipt.get("projected_window_units")) or not math.isclose(
        receipt["projected_window_units"], projected, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError("Projection must be recomputed from the actual observed rate and full VM window")
    if receipt.get("projection_is_actual_job_cost") is not False:
        raise ValueError("Rounded account observations are not exact per-job billing")
    if active and not start <= (now or datetime.now(UTC)) < deadline:
        raise PermissionError("The specifically approved new window is not active")
    return receipt


def arm_process_deadline(receipt, *, stop=os._exit):
    delay = (utc(receipt["process_deadline_utc"]) - datetime.now(UTC)).total_seconds()
    if delay <= 0:
        raise PermissionError("No inference after the approved process deadline")

    def expire():
        try:
            write_atomic(
                REPORTS / "deadline-stop.json",
                {
                    "run_id": RUN_ID,
                    "status": "deadline_exhausted",
                    "stopped_at_utc": datetime.now(UTC).isoformat(),
                    "resource_receipt_sha256": sha(RECEIPT),
                    "automatic_retry_allowed": False,
                    "billing_stopped": False,
                    "instruction": "Preserve raw evidence and release only this runtime immediately.",
                },
            )
        finally:
            stop(124)

    timer = threading.Timer(delay, expire)
    timer.daemon = True
    timer.start()
    return timer
