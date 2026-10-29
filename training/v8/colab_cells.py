"""Inspect-only notebook cell sources. No allocation, upload or execution on import."""

# ruff: noqa: E501 -- Generated notebook source is parsed and reviewed separately.

CONFIG_CODE = '''import json, threading
from datetime import UTC, datetime
from pathlib import Path
from google.colab import runtime

OPERATOR_EXECUTE_V8 = False
# Filled only after new human approval AND the allocated T4's observed UI balance/rate.
V8_RESOURCE_RECEIPT = {}
if not OPERATOR_EXECUTE_V8:
    raise RuntimeError("Inspect only. New direct approval and actual resource observations are required.")
if V8_RESOURCE_RECEIPT.get("round") != "v8" or not V8_RESOURCE_RECEIPT.get("explicit_human_approval"):
    raise RuntimeError("The old96-record allowance cannot authorize this new round")
release_at = datetime.fromisoformat(V8_RESOURCE_RECEIPT["runtime_release_deadline_utc"])
seconds_to_release = (release_at - datetime.now(UTC)).total_seconds()
if not 0 < seconds_to_release <= 8*3600:
    raise RuntimeError("Invalid fixed runtime-release window")
v8_release_timer = threading.Timer(seconds_to_release, runtime.unassign)
v8_release_timer.daemon = True
v8_release_timer.start()
print(json.dumps({"runtime_release_timer_alive":v8_release_timer.is_alive(),
                  "release_at_utc":release_at.isoformat(),
                  "billing_warning":"Process exit alone does not release a billed VM"}))
'''

SETUP_CODE = '''import hashlib, importlib.metadata, os, stat, subprocess, sys, zipfile
from pathlib import PurePosixPath
from google.colab import files

EXPECTED = {
    "resolvedesk-v8-source.zip":"SOURCE_SHA_PLACEHOLDER",
    "resolvedesk-v8-initializers.zip":"WEIGHTS_SHA_PLACEHOLDER",
}
EXPECTED_PLAN_SHA = "PLAN_SHA_PLACEHOLDER"
if not OPERATOR_EXECUTE_V8 or not v8_release_timer.is_alive():
    raise RuntimeError("Explicit operator approval and the independent VM-release timer are required")
if V8_RESOURCE_RECEIPT.get("analysis_plan_sha256") != EXPECTED_PLAN_SHA:
    raise RuntimeError("Resource approval is for another plan")
uploaded = files.upload()
if set(uploaded) != set(EXPECTED):
    raise RuntimeError("Upload exactly the two approved V8 packages")
workspace = Path("/content/resolvedesk-v8")
workspace.mkdir(exist_ok=False)
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
def file_digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            value.update(chunk)
    return value.hexdigest()
for archive_name, wanted in EXPECTED.items():
    raw = uploaded[archive_name]
    if digest(raw) != wanted:
        raise RuntimeError("Package SHA mismatch: "+archive_name)
    manifest_name = "v8-source-manifest.json" if "source" in archive_name else "v8-initializer-manifest.json"
    with zipfile.ZipFile(archive_name) as archive:
        entries = archive.infolist()
        if len({e.filename for e in entries}) != len(entries) or sum(e.file_size for e in entries)>512*1024**2:
            raise RuntimeError("Duplicate names or oversized package")
        manifest = json.loads(archive.read(manifest_name))
        if set(e.filename for e in entries) != set(manifest["files"])|{manifest_name}:
            raise RuntimeError("Archive membership differs from its manifest")
        for entry in entries:
            name = entry.filename
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\\\" in name or stat.S_ISLNK(entry.external_attr>>16):
                raise RuntimeError("Unsafe archive entry")
            content = archive.read(name)
            if name != manifest_name and digest(content) != manifest["files"][name]:
                raise RuntimeError("Member SHA mismatch: "+name)
            target = workspace/name
            if target.exists():
                raise RuntimeError("Never overwrite workspace inputs")
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
if digest((workspace/"reports/v8/pre-inference-plan.json").read_bytes()) != EXPECTED_PLAN_SHA:
    raise RuntimeError("Frozen plan mismatch")
sys.path.insert(0,str(workspace))
os.chdir(workspace)
os.environ["USE_TF"]="0"
os.environ["TOKENIZERS_PARALLELISM"]="false"
receipt_path=workspace/"reports/v8/resource-authorization.json"
with receipt_path.open("x") as stream:
    stream.write(json.dumps(V8_RESOURCE_RECEIPT,indent=2)+"\\n")
from training.v8.common import verify_frozen
from training.v8.resource_guard import verify_resource_receipt
verify_frozen(); verify_resource_receipt()
device=subprocess.check_output(["nvidia-smi","--id=0","--query-gpu=name","--format=csv,noheader,nounits"],text=True).strip()
if device.casefold() not in {"tesla t4","nvidia t4","t4"}:
    raise RuntimeError("This plan permits only one T4")
if importlib.metadata.version("torch") != "2.11.0+cu128":
    raise RuntimeError("Unexpected Colab Torch runtime: stop, preserve evidence, release; no automatic framework upgrade")
subprocess.run([sys.executable,"-m","pip","uninstall","-y","torchao"],check=True)
subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check",
               "peft==0.19.1","accelerate==1.14.0","transformers==5.21.0",
               "safetensors==0.8.0","numpy==2.5.4","fastapi>=0.115,<1","httpx>=0.28,<1"],check=True)
from huggingface_hub import snapshot_download
spec=json.loads((workspace/"training/base-model-spec.json").read_text())
base=workspace/"data/models/qwen3-4b-mlx"
snapshot_download(spec["repo"],revision=spec["revision"],allow_patterns=[f["path"] for f in spec["files"]],local_dir=base,token=False)
for item in spec["files"]:
    path=base/item["path"]
    if path.stat().st_size != item["bytes"] or file_digest(path) != item["sha256"]:
        raise RuntimeError("Pinned public base mismatch: "+item["path"])
os.environ["HF_HUB_OFFLINE"]="1"
from training.v8.cuda_train import dependency_precheck
dependency_precheck()
print(json.dumps({"ready":True,"workspace":str(workspace),"gpu":device,
                  "source_and_weights_verified":True,"plan_sha256":EXPECTED_PLAN_SHA,
                  "runtime_release_timer_alive":v8_release_timer.is_alive()}))
'''

LAUNCH_CODE = '''import subprocess,sys
from training.v8.common import verify_frozen
from training.v8.resource_guard import verify_resource_receipt
verify_frozen(); verify_resource_receipt()
if not v8_release_timer.is_alive():
    raise RuntimeError("Independent runtime release timer must remain armed")
if (workspace/"reports/v8/cloud-launch.json").exists() or "v8_process" in locals():
    raise RuntimeError("Never start a second V8 process or replay an issued phase")
v8_orchestrator_log=(workspace/"reports/v8/orchestrator-console.log").open("x")
v8_process=subprocess.Popen([sys.executable,"-u","-B","-m","training.v8.cloud_run"],
                            cwd=workspace,stdout=v8_orchestrator_log,stderr=subprocess.STDOUT)
print(json.dumps({"issued_once":True,"pid":v8_process.pid,"process_return_code":v8_process.poll(),
                  "scope":"smoke4/fresh1344/CPU proof/validation48/three-arm test144"}))
'''

STATUS_CODE = '''import base64
from datetime import UTC,datetime
from IPython.display import HTML,display
from training.v8.snapshot import collect_snapshot
proc=locals().get("v8_process")
raw,receipt=collect_snapshot(workspace,proc.poll() if proc is not None else None)
name="resolvedesk-v8-"+datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")+".zip"
print(json.dumps(receipt["progress"],indent=2),flush=True)
print(json.dumps({"snapshot":name,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),
                  "complete_artifacts_present":receipt["complete_artifacts_present"],
                  "scope":"Artifact completeness only; independent local token/business/weight verification still required"}))
display(HTML('<a download="'+name+'" href="data:application/zip;base64,'+base64.b64encode(raw).decode()+'">Download V8 raw evidence and available verified-weight snapshot</a>'))
'''
