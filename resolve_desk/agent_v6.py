"""Opt-in V6 sessions; historical sessions retain their original controller."""

from .agent_v5 import advance as previous_advance
from .resolution_v6 import messages


def advance(store, sid, provider, max_steps=12):
    if store.get(sid).get("workflow_version") != "v6":
        return previous_advance(store, sid, provider, max_steps=max_steps)
    for _ in range(max_steps):
        state = store.get(sid)
        if state["status"] != "running":
            return state
        raw, timing = provider.generate(messages(state))
        store.apply(sid, raw, timing, state["revision"])
    return store.get(sid)
