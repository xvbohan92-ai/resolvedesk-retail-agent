"""Wait for competing model caches; never unload another task's model."""

import json
import time
import urllib.error
import urllib.request


def wait_for_room():
    started, previous = time.monotonic(), None
    while True:
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5) as response:
                models = json.load(response).get("models", [])
        except (urllib.error.URLError, TimeoutError):
            models = []
        occupied = sum(m.get("size_vram", m.get("size", 0)) for m in models)
        if occupied <= 6 * 1024**3:
            return round(time.monotonic() - started, 3)
        current = tuple(m["name"] for m in models)
        if current != previous:
            print(
                json.dumps({"waiting_for_natural_cache_release": current, "loaded_vram_bytes": occupied}),
                flush=True,
            )
            previous = current
        time.sleep(15)
