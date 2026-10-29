"""Install the built wheel offline in a temporary environment, outside the source tree."""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from training.common import ROOT, sha
from training.v5.common import REPORTS

PROBE = """
import importlib.metadata
import json
import os
from pathlib import Path
from fastapi.testclient import TestClient
import resolve_desk
from resolve_desk.api import app

assert importlib.metadata.version("resolvedesk") == "0.5.0"
assert Path(resolve_desk.__file__).is_relative_to(Path(os.environ["VIRTUAL_ENV"]))
client = TestClient(app)
cases = client.get("/api/cases").json()
assert len(cases) == 126
assert sum(c["id"].startswith("v5-valid-") for c in cases) == 32
assert all(set(c) == {"id", "family", "message"} for c in cases)
assert client.post("/api/sessions", json={"case_id": "v5-test-exchange-00"}).status_code == 400
for experiment in ["v1", "v2", "v3", "v4", "v5"]:
    assert client.get("/api/evidence", params={"experiment":experiment}).json()["available"] is False
assert client.get("/").status_code == 200
print(json.dumps({"installed_version":"0.5.0", "packaged_worlds":len(cases),
                  "source_tree_imported":False, "absent_reports_handled":True,
                  "test_worlds_blocked":True}))
"""


def main():
    wheel = ROOT / "dist/resolvedesk-0.5.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert "resolve_desk/fixtures/validation-v5.json" in names
    assert not any(
        name.startswith(("training/", "data/", "reports/", ".env"))
        or name.endswith((".safetensors", ".gguf", ".sqlite"))
        for name in names
    )
    with tarfile.open(ROOT / "dist/resolvedesk-0.5.0.tar.gz") as archive:
        source_names = ["/".join(n.split("/")[1:]) for n in archive.getnames()]
    assert "training/v5/evaluate.py" in source_names
    assert not any(
        n.startswith(("data/", ".venv/", "training/.venv/"))
        or n.endswith((".safetensors", ".gguf", ".sqlite"))
        or Path(n).name.startswith(".env")
        and Path(n).name != ".env.example"
        for n in source_names
    )
    assert "reports/v5/package-verification.json" not in source_names
    with tempfile.TemporaryDirectory(prefix="resolvedesk-v5-package-") as tmp:
        environment = Path(tmp) / "venv"
        env = {
            **os.environ,
            "UV_CACHE_DIR": str(ROOT.parent / ".uv-cache"),
            "VIRTUAL_ENV": str(environment),
            "RESOLVEDESK_DATA": str(Path(tmp) / "sandbox"),
        }
        subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], env=env, check=True)
        python = environment / "bin/python"
        subprocess.run(
            ["uv", "pip", "install", "--offline", "--python", str(python), str(wheel)], env=env, check=True
        )
        result = subprocess.run(
            [str(python), "-c", PROBE], cwd=tmp, env=env, check=True, capture_output=True, text=True
        )
    report = {
        **json.loads(result.stdout),
        "wheel_sha256": sha(wheel),
        "sdist_sha256": sha(ROOT / "dist/resolvedesk-0.5.0.tar.gz"),
        "source_sha256": sha(__file__),
        "offline_install": True,
        "weights_and_credentials_excluded": True,
        "build_receipt_excluded_from_own_archive": True,
    }
    (REPORTS / "package-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
