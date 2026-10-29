"""Bound MLX allocator resources without changing frozen experiment sources."""

import json
import runpy
import sys


def main():
    import mlx.core as mx

    module = sys.argv[1]
    if module not in {"training.v4.calibration_select", "training.v4.facts_evaluate"}:
        raise ValueError("Only the frozen final validation and evaluation entry points are allowed")
    mx.set_wired_limit(6 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    mx.clear_cache()
    print(
        json.dumps(
            {
                "module": module,
                "wired_limit_bytes": 6 * 1024**3,
                "cache_limit_bytes": 256 * 1024**2,
                "generation_and_selection_unchanged": True,
            }
        ),
        flush=True,
    )
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
