"""One issued V8 job: smoke, fresh continuation, CPU weight proof, validation and test."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime

from training.common import ROOT, sha
from training.cuda_eval.evaluation import write_atomic
from training.v8.common import verify_frozen
from training.v8.resource_guard import arm_process_deadline, verify_resource_receipt

PHASES = (
    (
        "discarded_smoke4",
        ["training.v8.cuda_train", "--kind", "smoke", "--steps", "4", "--output", "reports/v8/cuda/smoke-4"],
    ),
    (
        "fixed_final1344",
        [
            "training.v8.cuda_train",
            "--kind",
            "train",
            "--steps",
            "1344",
            "--output",
            "reports/v8/cuda/train-1344",
            "--smoke-report",
            "reports/v8/cuda/smoke-4/report.json",
        ],
    ),
    ("cpu_weight_proof", ["training.v8.verify_training"]),
    ("validation48_then_three_arm_test144", ["training.v8.cuda_evaluate", "--execute"]),
)


def main():
    verify_frozen()
    receipt = verify_resource_receipt()
    destination = ROOT / "reports/v8"
    marker = destination / "cloud-launch.json"
    if marker.exists() or any(
        (destination / name).exists()
        for name in ("cuda", "artifact-identity.json", "validation", "evaluation")
    ):
        raise FileExistsError(
            "A V8 job has already been issued. Never replay an ambiguous phase or model call"
        )
    report = {
        "round": "v8",
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "resource_receipt_sha256": sha(destination / "resource-authorization.json"),
        "analysis_plan_sha256": sha(destination / "pre-inference-plan.json"),
        "phases": [],
        "automatic_retry_allowed": False,
        "resume_count": 0,
        "old_round_outputs_modified": False,
    }
    # Reserve launch identity atomically so racing cells cannot replace provenance.
    with marker.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    timer = arm_process_deadline(receipt, "orchestrator")
    try:
        for phase, command in PHASES:
            verify_resource_receipt()
            if sha(destination / "resource-authorization.json") != report["resource_receipt_sha256"]:
                raise PermissionError("The original new-round approval changed during execution")
            row = {
                "phase": phase,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "status": "issued",
                "command": [sys.executable, "-u", "-B", "-m", *command],
            }
            report["phases"].append(row)
            write_atomic(marker, report)
            log = destination / (phase + ".log")
            with log.open("x") as stream:
                env = dict(os.environ)
                if phase == "cpu_weight_proof":
                    env.update(USE_TORCH="0", USE_TF="0", HF_HUB_OFFLINE="1")
                completed = subprocess.run(
                    row["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env
                )
            row.update(
                return_code=completed.returncode,
                completed_at_utc=datetime.now(UTC).isoformat(),
                status="completed" if completed.returncode == 0 else "failed",
                log_sha256=sha(log),
            )
            write_atomic(marker, report)
            if completed.returncode:
                raise RuntimeError(f"V8 phase failed: {phase}; all evidence retained, no automatic retry")
        report.update(status="complete", completed_at_utc=datetime.now(UTC).isoformat())
    except BaseException as error:
        report.update(
            status="failed",
            error={"type": type(error).__name__, "detail": str(error)},
            failed_at_utc=datetime.now(UTC).isoformat(),
        )
        raise
    finally:
        write_atomic(marker, report)
        timer.cancel()
    print(
        json.dumps({"round": "v8", "status": report["status"], "phases_completed": len(PHASES)}), flush=True
    )


if __name__ == "__main__":
    main()
