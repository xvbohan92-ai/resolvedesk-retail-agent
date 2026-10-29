"""Validation-only regression guard, frozen before calibration updates/test."""


def eligible(candidate, baseline):
    return (
        candidate["exchange_as_return"] == 0
        and candidate["task_success"] >= baseline["task_success"]
        and candidate["fulfillable_exchange_success"] >= baseline["fulfillable_exchange_success"]
        and candidate["handoff_success"] >= baseline["handoff_success"]
        and candidate["clarifications_obtained"] >= baseline["clarifications_obtained"]
        and candidate["direct_return_success"] >= baseline["direct_return_success"]
        and candidate["repeated_question_attempts"] <= baseline["repeated_question_attempts"]
    )


def prefer_calibration(candidate, baseline):
    def key(s):
        return s["fallback_success"], s["task_success"], -s["repeated_question_attempts"], -s["model_calls"]

    return eligible(candidate, baseline) and key(candidate) > key(baseline)
