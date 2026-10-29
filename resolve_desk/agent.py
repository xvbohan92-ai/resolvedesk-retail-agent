from .decision_facts import augment_messages
from .progress import messages


def step(store, sid, provider):
    state = store.get(sid)
    text, timing = provider.generate(augment_messages(messages(state)))
    return store.apply(sid, text, timing, state["revision"])


def advance(store, sid, provider, max_steps=8):
    for _ in range(max_steps):
        state = store.get(sid)
        if state["status"] != "running":
            return state
        state = step(store, sid, provider)
    # Budget is visible and resumable; the server never fabricates a successful result.
    return store.get(sid)
