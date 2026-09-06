"""Small shared IO and provenance helpers, independent of the GPU stack."""
import hashlib
import json
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"Expected portable relative path: {relative!r}")
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Path escapes image root: {relative}")
    return path


def require_drive(path):
    """Persistent training artifacts must be under mounted Google Drive."""
    path = Path(path).resolve()
    root = Path("/content/drive/MyDrive")
    if not root.is_dir() or not path.is_relative_to(root):
        raise ValueError(f"Use a path under mounted {root}: {path}")
    return path
