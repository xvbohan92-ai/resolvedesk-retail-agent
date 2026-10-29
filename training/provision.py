import argparse
import json
import shutil
from pathlib import Path

from training.common import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--from-local", type=Path)
    args = parser.parse_args()
    spec = json.loads((ROOT / "training/base-model-spec.json").read_text())
    destination = ROOT / "data/models/qwen3-4b-mlx"
    destination.mkdir(parents=True, exist_ok=True)
    if args.from_local:
        for record in spec["files"]:
            source = args.from_local / record["path"]
            assert sha(source) == record["sha256"], record["path"]
            if not (destination / record["path"]).exists():
                shutil.copyfile(source, destination / record["path"])
    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=spec["repo"],
            revision=spec["revision"],
            local_dir=destination,
            allow_patterns=[r["path"] for r in spec["files"]],
            token=False,
        )
    for record in spec["files"]:
        path = destination / record["path"]
        if not path.exists() or sha(path) != record["sha256"]:
            raise ValueError(f"Missing or changed base model file: {record['path']}")
    (ROOT / "data/models/qwen3-4b-mlx-source.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(
        json.dumps({"repo": spec["repo"], "revision": spec["revision"], "verified_files": len(spec["files"])})
    )


if __name__ == "__main__":
    main()
