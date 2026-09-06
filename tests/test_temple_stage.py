"""Archive and cache tests use local fixture archives, never the Drive originals."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from test_temple import temple  # Shared pytest fixture, not a real dataset.
from aquafina_detector.common import read_json, sha256
from aquafina_detector.temple import convert_temple, snapshot_raw, preview_temple
from aquafina_detector import temple_stage


@pytest.fixture
def archive_fixture(temple, tmp_path):
    root, expected = temple
    folder = tmp_path / "archives"
    folder.mkdir()
    archive = folder / "detection_dataset.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(root, arcname="detection_dataset")
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    return archive, md5, root, expected


def test_single_archive_read_local_extraction_and_originals_unchanged(archive_fixture, tmp_path, monkeypatch):
    archive, md5, root, _ = archive_fixture
    before, archive_before = snapshot_raw(root), sha256(archive)
    original_open = Path.open
    source_reads = []

    def track(path, mode="r", *args, **kwargs):
        if path == archive and mode == "rb":
            source_reads.append(path)
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(Path, "open", track)
        stage = temple_stage.stage_archive(archive, md5, tmp_path / "stage", progress=None)
    assert len(source_reads) == 1
    assert stage.root == tmp_path / "stage/detection_dataset"
    assert stage.archive_md5 == md5
    assert stage.snapshot == before == snapshot_raw(stage.root)
    assert snapshot_raw(root) == before
    assert sha256(archive) == archive_before
    # Reusing extraction still verifies the source archive and local content.
    messages = []
    again = temple_stage.stage_archive(archive, md5, tmp_path / "stage", progress=messages.append)
    assert again.snapshot == stage.snapshot
    assert any("Reusing verified local extraction" in message for message in messages)


def test_md5_mismatch_stops_before_extraction(archive_fixture, tmp_path):
    archive, _, root, _ = archive_fixture
    before = snapshot_raw(root)
    archive_before = sha256(archive)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        temple_stage.stage_archive(archive, "0" * 32, tmp_path / "stage", progress=None)
    assert not (tmp_path / "stage/detection_dataset").exists()
    assert not (tmp_path / "stage/stage.json").exists()
    assert before == snapshot_raw(root) and archive_before == sha256(archive)


@pytest.mark.parametrize("name,kind", [
    ("../escape.txt", "file"), ("/absolute.txt", "file"),
    ("detection_dataset/../../escape.txt", "file"),
    ("detection_dataset/link", "symlink"), ("detection_dataset/hard", "hardlink"),
])
def test_unsafe_archive_rejected(tmp_path, name, kind):
    folder = tmp_path / "archives"
    folder.mkdir()
    archive = folder / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(name)
        if kind == "file":
            member.size = 1
            bundle.addfile(member, io.BytesIO(b"x"))
        else:
            member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            member.linkname = "../../outside"
            bundle.addfile(member)
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Archive"):
        temple_stage.stage_archive(archive, md5, tmp_path / "stage", progress=None)
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "stage/detection_dataset").exists()


def test_verified_cache_reused_without_decoding_and_can_convert(archive_fixture, tmp_path, monkeypatch):
    archive, md5, root, expected = archive_fixture
    original_snapshot = snapshot_raw(root)
    stage = temple_stage.stage_archive(archive, md5, tmp_path / "stage1", progress=None)
    cache_dir = tmp_path / "cache"
    first = temple_stage.audit_staged(stage, cache_dir, expected=expected, progress=None)
    assert (cache_dir / f"{md5}.json").is_file()
    assert first.audit["blocking_errors"] == 0
    # Simulate a new Colab runtime with fresh extraction at another temporary path.
    second_stage = temple_stage.stage_archive(archive, md5, tmp_path / "stage2", progress=None)

    def no_audit(*args, **kwargs):
        pytest.fail("Cache reuse must skip full decoding/XML/label audit")

    monkeypatch.setattr(temple_stage, "audit_temple", no_audit)
    messages = []
    cached = temple_stage.audit_staged(second_stage, cache_dir, expected=expected, progress=messages.append)
    assert cached.root == second_stage.root
    assert cached.records == first.records
    assert cached.train_ids == first.train_ids and cached.val_ids == first.val_ids
    assert any("Reused verified audit cache" in m for m in messages)
    assert preview_temple(cached, limit=1)
    with pytest.raises(ValueError, match="original Drive raw"):
        convert_temple(cached, archive.parent / "extracted/detection_dataset/processed", confirm=True)
    converted = convert_temple(cached, tmp_path / "processed", confirm=True)
    assert converted["raw_unchanged"]
    assert original_snapshot == snapshot_raw(root)


@pytest.mark.parametrize("problem", ["truncated", "signature", "digest", "blocked", "sha256", "force", "old_schema"])
def test_stale_or_unverified_cache_does_not_skip_audit(archive_fixture, tmp_path, monkeypatch, problem):
    archive, md5, _, expected = archive_fixture
    stage = temple_stage.stage_archive(archive, md5, tmp_path / "stage", progress=None)
    cache_dir = tmp_path / "cache"
    temple_stage.audit_staged(stage, cache_dir, expected=expected, progress=None)
    cache = cache_dir / f"{md5}.json"
    envelope = read_json(cache)
    if problem == "old_schema":
        envelope["schema"] = 1
    if problem == "truncated":
        cache.write_text('{"unfinished":')
    else:
        if problem == "signature":
            envelope["audit_signature"] = "old code"
        elif problem == "digest":
            envelope["payload"]["train_ids"] = []
        elif problem == "blocked":
            envelope["payload"]["audit"]["blocking_errors"] = 1
            envelope["payload_sha256"] = temple_stage._payload_digest(envelope["payload"])
        elif problem == "sha256":
            envelope["archive_sha256"] = "wrong archive"
        cache.write_text(json.dumps(envelope))
    original = temple_stage.audit_temple
    calls = []

    def track(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(temple_stage, "audit_temple", track)
    temple_stage.audit_staged(stage, cache_dir, expected=expected, reuse=problem != "force", progress=None)
    assert calls == [True]


def test_interrupted_audit_is_not_cached(archive_fixture, tmp_path, monkeypatch):
    archive, md5, _, expected = archive_fixture
    stage = temple_stage.stage_archive(archive, md5, tmp_path / "stage", progress=None)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(temple_stage, "audit_temple", interrupt)
    with pytest.raises(KeyboardInterrupt):
        temple_stage.audit_staged(stage, tmp_path / "cache", expected=expected, progress=None)
    assert not (tmp_path / "cache" / f"{md5}.json").exists()


def test_changed_stage_and_source_destinations_rejected(archive_fixture, tmp_path):
    archive, md5, root, expected = archive_fixture
    with pytest.raises(ValueError, match="separate"):
        temple_stage.stage_archive(archive, md5, archive.parent / "stage", progress=None)
    stage = temple_stage.stage_archive(archive, md5, tmp_path / "stage", progress=None)
    with pytest.raises(ValueError, match="separate"):
        temple_stage.audit_staged(stage, archive.parent / "cache", expected=expected, progress=None)
    (stage.root / "Readme.txt").write_text("modified temporary stage")
    with pytest.raises(ValueError, match="changed"):
        temple_stage.audit_staged(stage, tmp_path / "cache", expected=expected, progress=None)
    with pytest.raises(ValueError, match="changed"):
        temple_stage.stage_archive(archive, md5, tmp_path / "stage", progress=None)
    assert (root / "Readme.txt").read_text() == "Original source; never edit"


def test_progress_interval_and_completion(temple, monkeypatch):
    from aquafina_detector import temple as module
    root, expected = temple
    assert module.PROGRESS_EVERY == 250
    monkeypatch.setattr(module, "PROGRESS_EVERY", 2)
    messages = []
    module.audit_temple(root, expected, progress=messages.append)
    assert [m.split(" images")[0] for m in messages if m.startswith("Audited ")] == [
        "Audited 2/6", "Audited 4/6", "Audited 6/6"]
