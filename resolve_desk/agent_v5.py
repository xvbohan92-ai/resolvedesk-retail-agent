"""New sessions use typed intent; preserved sessions retain their v4 controller."""

from .agent import advance as legacy_advance
from .resolution import messages


def advance(store, sid, provider, max_steps=12):
    if store.get(sid).get("workflow_version") != "v5":
        return legacy_advance(store, sid, provider, max_steps=max_steps)
    for _ in range(max_steps):
        state = store.get(sid)
        if state["status"] != "running":
            return state
        raw, timing = provider.generate(messages(state))
        store.apply(sid, raw, timing, state["revision"])
    return store.get(sid)
