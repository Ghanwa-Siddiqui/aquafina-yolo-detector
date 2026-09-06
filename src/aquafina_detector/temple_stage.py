"""Stage one verified archive on local disk and cache completed TempleRAIL audits.

Drive raw files are read-only. The only Drive writes here are small audit-cache
JSON files, never images or annotations. Conversion remains a separate opt-in.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

from .common import contained, read_json, write_json
from .temple import EXPECTED, TempleAudit, audit_temple, snapshot_raw

ARCHIVE = Path("/content/drive/MyDrive/aquafina-yolo/raw/temple/detection_dataset.tar.gz")
ARCHIVE_MD5 = "fca7260d4785af1dec18aa320fa9fc4a"
STAGE_DIR = Path("/content/temple_stage")
CACHE_DIR = Path("/content/drive/MyDrive/aquafina-yolo/cache/temple_audit")
CACHE_SCHEMA = 1


@dataclass
class VerifiedStage:
    root: Path
    archive: Path
    archive_md5: str
    archive_sha256: str
    snapshot: dict


def _md5(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{32}", value) is None:
        raise ValueError("Expected MD5 must contain exactly 32 hexadecimal characters")
    return value.lower()


def _say(progress, message):
    if progress:
        progress(message)


def _copy_archive(source, destination, progress):
    """One sequential source read, hashing both algorithms while copying."""
    md5, sha = hashlib.md5(), hashlib.sha256()
    count, checkpoint = 0, 0
    with source.open("rb") as reader, destination.open("wb") as writer:
        for block in iter(lambda: reader.read(4 * 1024 * 1024), b""):
            writer.write(block)
            md5.update(block)
            sha.update(block)
            count += len(block)
            if count - checkpoint >= 256 * 1024 * 1024:
                _say(progress, f"Copied and hashed {count // (1024 * 1024):,} MiB of archive")
                checkpoint = count
    return md5.hexdigest(), sha.hexdigest()


def _extract_verified(archive, destination, progress):
    """Extract regular files/directories only, hashing each file as it is written.

    Never use extractall: traversal, links, device files and duplicate file names
    are rejected. Ignore a harmless leading './' directory entry only.
    """
    snapshot, filenames = {}, set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            name = PurePosixPath(member.name)
            if member.isdir() and name.as_posix() == ".":
                continue
            if (name.is_absolute() or ".." in name.parts or "\\" in member.name or ":" in member.name
                    or not name.parts or name.parts[0] != "detection_dataset"):
                raise ValueError(f"Archive path outside detection_dataset: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive links/special files are forbidden: {member.name}")
            target = contained(destination, name.as_posix())
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            relative = name.relative_to("detection_dataset").as_posix()
            if relative == "." or relative.casefold() in filenames:
                raise ValueError(f"Duplicate/invalid archive file: {member.name}")
            filenames.add(relative.casefold())
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, written = hashlib.sha256(), 0
            reader = bundle.extractfile(member)
            if reader is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            with reader, target.open("xb") as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(block)
                    digest.update(block)
                    written += len(block)
            if written != member.size:
                raise ValueError(f"Truncated archive member: {member.name}")
            snapshot[relative] = digest.hexdigest()
            if len(snapshot) % 1000 == 0:
                _say(progress, f"Extracted {len(snapshot):,} files onto temporary disk")
    if not snapshot:
        raise ValueError("Archive has no dataset files")
    return snapshot


def stage_archive(archive=ARCHIVE, expected_md5=ARCHIVE_MD5, stage_dir=STAGE_DIR, *, progress=print):
    """Verify Drive archive and extract locally. Never overwrite a dataset tree.

    Each invocation reads the single source archive once. A completed local stage
    can be reused only after its hashes match; incomplete stages are not adopted.
    On errors, leave temporary files for inspection rather than deleting anything.
    """
    expected_md5 = _md5(expected_md5)
    archive, stage_dir = Path(archive).resolve(), Path(stage_dir).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if (stage_dir.is_relative_to(archive.parent) or archive.is_relative_to(stage_dir)
            or stage_dir.is_relative_to(Path("/content/drive").resolve())):
        raise ValueError("Stage must be on temporary disk, separate from the source archive directory and Drive")
    stage_dir.mkdir(parents=True, exist_ok=True)
    _say(progress, "Reading one archive from Drive; copying and verifying MD5 on temporary disk...")
    descriptor, temporary = tempfile.mkstemp(prefix="archive-", suffix=".part", dir=stage_dir)
    os.close(descriptor)
    temporary = Path(temporary)
    actual_md5, archive_sha = _copy_archive(archive, temporary, progress)
    if actual_md5 != expected_md5:
        raise ValueError(f"Archive MD5 mismatch: expected {expected_md5}, got {actual_md5}; nothing extracted")
    _say(progress, f"Archive MD5 verified: {actual_md5}")
    local_archive = stage_dir / (actual_md5 + ".tar.gz")
    if local_archive.is_symlink():
        raise ValueError("Local archive path must not be a symlink")
    os.replace(temporary, local_archive)
    root, marker = stage_dir / "detection_dataset", stage_dir / "stage.json"
    if root.is_symlink() or marker.is_symlink():
        raise ValueError("Staged dataset and marker must not be symlinks")
    if root.exists():
        if not marker.is_file():
            raise ValueError("Existing stage has no completion marker; use a fresh temporary directory or restart runtime")
        metadata = read_json(marker)
        if metadata.get("archive_md5") != actual_md5 or metadata.get("archive_sha256") != archive_sha:
            raise ValueError("Existing stage belongs to a different archive; use a fresh temporary directory")
        _say(progress, "Checking existing staged files on local disk...")
        snapshot = snapshot_raw(root)
        if snapshot != metadata.get("snapshot"):
            raise ValueError("Staged files changed; use a fresh temporary directory or restart runtime")
        _say(progress, "Reusing verified local extraction")
    else:
        extraction = Path(tempfile.mkdtemp(prefix="extract-", dir=stage_dir))
        _say(progress, "Extracting archive onto temporary disk...")
        snapshot = _extract_verified(local_archive, extraction, progress)
        os.replace(extraction / "detection_dataset", root)
        write_json(marker, {"archive_md5": actual_md5, "archive_sha256": archive_sha, "snapshot": snapshot})
        _say(progress, f"Staged {len(snapshot):,} files at {root}")
    return VerifiedStage(root, archive, actual_md5, archive_sha, snapshot)


def audit_signature(expected):
    """Invalidate cached decisions if audit/validation/staging code or policy changes."""
    digest = hashlib.sha256()
    for name in ("temple.py", "temple_stage.py", "data.py", "common.py", "__init__.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    digest.update(json.dumps(expected, sort_keys=True).encode())
    return digest.hexdigest()


def _payload_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def audit_staged(stage, cache_dir=CACHE_DIR, *, reuse=True, expected=None, progress=print):
    """Reuse a verified audit, otherwise audit locally and atomically cache results.

    Cache is one JSON keyed by archive MD5. A partial, modified, stale, blocked or
    differently configured report is never considered a verified reusable audit.
    """
    expected = EXPECTED if expected is None else expected
    cache_dir = Path(cache_dir).resolve()
    if (cache_dir.is_relative_to(stage.archive.parent) or cache_dir.is_relative_to(stage.root)
            or stage.root.is_relative_to(cache_dir)):
        raise ValueError("Audit cache must be separate from the original archive directory and staged dataset")
    # All per-file reads here are from temporary disk, never extracted Drive data.
    _say(progress, "Checking local stage hashes before audit/cache reuse...")
    if snapshot_raw(stage.root) != stage.snapshot:
        raise ValueError("Staged files changed since archive verification; restage before auditing")
    signature = audit_signature(expected)
    cache = cache_dir / (_md5(stage.archive_md5) + ".json")
    if reuse and cache.is_file():
        try:
            envelope = read_json(cache)
            payload = envelope["payload"]
            if (envelope.get("schema") == CACHE_SCHEMA and envelope.get("status") == "complete"
                    and envelope.get("archive_md5") == stage.archive_md5
                    and envelope.get("archive_sha256") == stage.archive_sha256
                    and envelope.get("audit_signature") == signature
                    and envelope.get("payload_sha256") == _payload_digest(payload)
                    and payload["snapshot"] == stage.snapshot
                    and payload["audit"]["blocking_errors"] == 0
                    and payload["audit"]["status"] == "ready_for_preview"):
                payload["root"] = stage.root
                payload["audit"]["raw_root"] = str(stage.root)
                payload["audit"]["original_archive"] = str(stage.archive)
                result = TempleAudit(**payload)
                _say(progress, f"Reused verified audit cache: {cache}; image/XML/label audit skipped")
                return result
            _say(progress, "Cache is stale, incompatible or blocked; running a fresh local audit")
        except (OSError, ValueError, TypeError, KeyError):
            _say(progress, "Cache is incomplete or unreadable; running a fresh local audit")
    result = audit_temple(stage.root, expected, progress=progress, _snapshot=stage.snapshot)
    if snapshot_raw(stage.root) != stage.snapshot:
        raise ValueError("Staged files changed during audit; result was not cached")
    result.audit["archive_md5"] = stage.archive_md5
    result.audit["archive_sha256"] = stage.archive_sha256
    result.audit["original_archive"] = str(stage.archive)
    payload = asdict(result)
    payload["root"] = str(result.root)
    write_json(cache, {"schema": CACHE_SCHEMA, "status": "complete", "archive_md5": stage.archive_md5,
               "archive_sha256": stage.archive_sha256, "audit_signature": signature,
               "payload_sha256": _payload_digest(payload), "payload": payload})
    _say(progress, f"Completed audit cached at {cache}")
    if result.audit["blocking_errors"]:
        _say(progress, "Audit has blocking errors; cached for inspection but not eligible for verified reuse")
    return result
