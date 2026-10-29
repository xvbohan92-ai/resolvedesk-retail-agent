"""Preserved business grader, additionally requiring every authored missing field."""

from training.v2.scoring import score as previous_score


def score(case, state, after, before, questions):
    result = previous_score(case, state, after, before, questions)
    obtained = all(f in questions for f in case["required_answers"])
    return {**result, "clarification_obtained": obtained, "task_success": result["task_success"] and obtained}
