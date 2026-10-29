"""Three-arm attribution using the preserved independent business and first-raw grader."""

from training.v7.evaluation import run_case, summarize

ARMS = ("prior_weights_old_flow", "prior_weights_new_flow", "v8_weights_new_flow")
MAX_CALLS = 12

__all__ = ["ARMS", "MAX_CALLS", "run_case", "summarize"]
