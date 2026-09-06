"""Validation-priority duplicate policy; synthetic files only."""
import copy
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from aquafina_detector.common import read_json
from aquafina_detector.data import verify_prepared
from aquafina_detector.temple import audit_temple, convert_temple, snapshot_raw, resolve_duplicates
from test_temple import temple


def test_nine_groups_conversion_and_raw_immutability(temple, tmp_path):
    root, expected = temple
    for n in range(9):
        train, val = f"train_dup{n}", f"val_dup{n}"
        Image.new("RGB", (100, 80), (10 + n * 20, 150, 5)).save(root / "JPEGImages" / f"{val}.jpg")
        (root / "JPEGImages" / f"{train}.jpg").write_bytes((root / "JPEGImages" / f"{val}.jpg").read_bytes())
        for key in (train, val):
            tree = ET.parse(root / "Annotations/img0.xml")
            tree.getroot().find("filename").text = key + ".jpg"
            tree.write(root / "Annotations" / f"{key}.xml")
            (root / "Labels" / f"{key}.txt").write_bytes((root / "Labels/img0.txt").read_bytes())
    val_file = root / "ImageSets/Main/val.txt"
    val_file.write_text(val_file.read_text() + "\n" + "\n".join(f"val_dup{n}" for n in range(9)))
    expected.update(images=24, xml=24, txt=27, train=13, val=11,
                    processed_train=4, cross_split_groups=9, excluded_train=9)
    expected["darknet_instances"]["Aquafina"] += 18
    before = snapshot_raw(root)
    audit = audit_temple(root, expected)
    assert audit.audit["blocking_errors"] == 0
    assert len(audit.train_ids) == 4 and len(audit.val_ids) == 11
    assert audit.audit["splits"]["cross_split_duplicate_content"] == 0
    assert not set(audit.train_ids) & set(audit.val_ids)
    output = tmp_path / "processed"
    convert_temple(audit, output, confirm=True)
    exclusions = read_json(output / "anomaly_report.json")["excluded_training_images"]
    assert len(exclusions) == 9
    for n, item in enumerate(exclusions):
        assert item["image_id"] == f"train_dup{n}"
        assert item["retained_validation_id"] == f"val_dup{n}"
        assert item["sha256"] == audit.records[item["retained_validation_id"]]["sha256"]
        assert item["reason"]
        assert not (output / "train2017" / (item["image_id"] + ".jpg")).exists()
        assert (output / "val2017" / (item["retained_validation_id"] + ".jpg")).exists()
    assert snapshot_raw(root) == before
    assert read_json(output / "conversion.json")["excluded_training_images"] == 9
    verify_prepared(output)
    marker = output / "conversion.json"
    marker.write_text(marker.read_text().replace('"train_images": 4', '"train_images": 13'))
    with pytest.raises(ValueError, match="completion checks"):
        verify_prepared(output)


def test_unexpected_duplicate_counts_block(temple):
    root, expected = temple
    (root / "JPEGImages/img5.jpg").write_bytes((root / "JPEGImages/img0.jpg").read_bytes())
    audit = audit_temple(root, expected)
    assert audit.audit["blocking_errors"] > 0
    assert any(i["kind"] == "duplicate_policy_count_mismatch" for i in audit.anomalies["issues"])


def test_policy_change_invalidates_compatibility_key(monkeypatch):
    from aquafina_detector import temple_stage
    before = temple_stage.audit_signature({})
    monkeypatch.setattr(temple_stage, "DUPLICATE_POLICY_VERSION", "future-policy")
    assert temple_stage.audit_signature({}) != before


@pytest.mark.parametrize("conflict", ["both_classes", "darknet_geometry", "dog", "invalid_pair", "multiple_val"])
def test_conflicting_or_ambiguous_duplicate_annotations_block(temple, tmp_path, conflict):
    root, expected = temple
    (root / "JPEGImages/img5.jpg").write_bytes((root / "JPEGImages/img0.jpg").read_bytes())
    expected.update(processed_train=3, cross_split_groups=1, excluded_train=1)
    if conflict == "both_classes":
        xml = root / "Annotations/img0.xml"
        xml.write_text(xml.read_text().replace("Aquafina", "Deer"))
        (root / "Labels/img0.txt").write_text("1 0.2 0.375 0.2 0.5")
        expected["darknet_instances"].update(Aquafina=2, Deer=3)
    elif conflict == "darknet_geometry":
        # Still within the per-image XML/Darknet tolerance; not equivalent copies.
        (root / "Labels/img0.txt").write_text("0 0.201 0.375 0.2 0.5")
    elif conflict == "dog":
        tree = ET.parse(root / "Annotations/img0.xml")
        dog = ET.parse(root / "Annotations/img4.xml").getroot().findall("object")[-1]
        tree.getroot().append(dog)
        tree.write(root / "Annotations/img0.xml")
    elif conflict == "invalid_pair":
        (root / "Labels/img0.txt").write_text("1 0.2 0.375 0.2 0.5")
    else:
        (root / "JPEGImages/img4.jpg").write_bytes((root / "JPEGImages/img0.jpg").read_bytes())
    audit = audit_temple(root, expected)
    assert audit.audit["blocking_errors"] > 0
    assert not audit.anomalies["excluded_training_images"]
    assert "img0" in audit.train_ids
    with pytest.raises(ValueError, match="blocking"):
        convert_temple(audit, tmp_path / "blocked", confirm=True)
    assert not (tmp_path / "blocked").exists()


def test_production_counts_preserve_all_validation_ids():
    # Full ID cardinality without generating thousands of fixture images.
    record = {"width": 100, "height": 80, "paired_annotations_verified": True,
              "xml_objects": [], "darknet_labels": []}
    records = {str(n): dict(copy.deepcopy(record), sha256=str(n)) for n in range(4870)}
    train, val = [str(n) for n in range(4000)], [str(n) for n in range(4000, 4870)]
    for n in range(9):
        records[str(n)]["sha256"] = records[str(4000 + n)]["sha256"]
    issues = []
    remaining, exclusions, crossing, overlap = resolve_duplicates(
        records, train, val, lambda *args, **kwargs: issues.append((args, kwargs)))
    assert (len(remaining), len(val), len(exclusions), crossing, overlap) == (3991, 870, 9, 9, 0)
    assert not set(remaining) & set(val)
