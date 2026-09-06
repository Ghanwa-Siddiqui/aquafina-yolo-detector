"""Explicit Colab setup. Never fetch datasets or weights during installation."""
import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

from . import UPSTREAM_COMMIT, UPSTREAM_URL, WEIGHTS_URL
from .common import require_drive, sha256, write_json


def audit_source(root):
    root = Path(root)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=root, text=True).strip()
    if actual != UPSTREAM_COMMIT or remote != UPSTREAM_URL:
        raise RuntimeError("YOLOX checkout is not the pinned official source")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip():
        raise RuntimeError("YOLOX source has modified tracked files")
    forbidden = []
    for path in (root / "yolox").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = ([n.name for n in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(m.split(".")[0].lower() in {"ultralytics", "yolov5"} for m in modules):
                forbidden.append(str(path))
    if forbidden:
        raise RuntimeError(f"Prohibited imports: {forbidden}")
    return {"commit": actual, "source": remote, "import_audit": "passed"}


def audit_runtime():
    packages = {d.metadata["Name"].lower().replace("_", "-"): d.version
                for d in importlib.metadata.distributions() if d.metadata["Name"]}
    if "ultralytics" in packages or importlib.util.find_spec("ultralytics") is not None:
        raise RuntimeError("Use a clean runtime without ultralytics installed")
    requirements = Path(__file__).resolve().parents[2] / "requirements" / "colab.txt"
    expected = {"torch": "2.5.1", "torchvision": "0.20.1"}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, version = line.strip().split("==")
            expected[name.lower().replace("_", "-")] = version
    mismatches = {name: {"expected": version, "installed": packages.get(name)}
                  for name, version in expected.items()
                  if packages.get(name, "").split("+")[0] != version}
    if mismatches:
        raise RuntimeError(f"Pinned dependency mismatch; rerun setup and restart session: {mismatches}")
    import yolox
    report = audit_source(Path(yolox.__file__).resolve().parent.parent)
    report["packages"] = packages
    return report


def install(repo, upstream="/content/YOLOX"):
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Pinned environment requires Python 3.10–3.12; choose a compatible Colab runtime")
    if not Path("/content").is_dir():
        raise RuntimeError("This installer is intended for Google Colab")
    if importlib.util.find_spec("ultralytics") is not None:
        raise RuntimeError("Start a clean runtime without ultralytics")
    root = Path(upstream)
    if not root.exists():
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM_URL, str(root)], check=True)
        subprocess.run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=root, check=True)
    audit_source(root)
    pip = [sys.executable, "-m", "pip"]
    subprocess.run(pip + ["install", "torch==2.5.1", "torchvision==0.20.1",
                         "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
    subprocess.run(pip + ["install", "-r", str(Path(repo) / "requirements/colab.txt")], check=True)
    # Import official source directly: avoid its optional ONNX build dependencies.
    # All project subprocesses get this source path through the notebook setup.
    subprocess.run(pip + ["install", "--no-deps", "-e", str(repo)], check=True)
    sys.path.insert(0, str(root))
    report = audit_runtime()
    report["python"] = sys.version
    report["pip_check"] = subprocess.run(pip + ["check"], text=True, capture_output=True).stdout
    print(json.dumps(report, indent=2))
    return report


def download_pretrained(destination):
    """Only called by an explicit notebook cell, writes exclusively to Drive."""
    path = require_drive(destination)
    if path.exists():
        raise FileExistsError("Weights already exist; reuse them or choose a new location")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    urllib.request.urlretrieve(WEIGHTS_URL, temporary)
    digest = sha256(temporary)
    os.replace(temporary, path)
    write_json(str(path) + ".provenance.json", {"url": WEIGHTS_URL, "sha256": digest,
        "note": "Recorded after download; no independently published checksum is asserted."})
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--upstream", default="/content/YOLOX")
    a = p.parse_args()
    install(a.repo, a.upstream)


if __name__ == "__main__":
    main()
