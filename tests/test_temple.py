"""TempleRAIL fixtures exercise source anomalies without accessing Google Drive."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from aquafina_detector.common import read_json
from aquafina_detector.data import verify_prepared
from aquafina_detector.temple import (
    CLASSES, audit_temple, convert_temple, preview_temple, reconstruct_splits, snapshot_raw,
)


@pytest.fixture
def temple(tmp_path):
    root = tmp_path / "raw" / "detection_dataset"
    for folder in ("JPEGImages", "Annotations", "Labels", "ImageSets/Main"):
        (root / folder).mkdir(parents=True)
    (root / "Readme.txt").write_text("Original source; never edit", encoding="utf-8")
    classes = [[0], [1], [2], [3], [0, 1], [0]]
    for index, class_ids in enumerate(classes):
        key = f"img{index}"
        Image.new("RGB", (100, 80), (index * 30, 40, 90)).save(root / "JPEGImages" / f"{key}.jpg")
        tree = ET.Element("annotation")
        ET.SubElement(tree, "filename").text = key + ".jpg"
        size = ET.SubElement(tree, "size")
        ET.SubElement(size, "width").text = "100"
        ET.SubElement(size, "height").text = "80"
        lines = []
        for j, class_id in enumerate(class_ids):
            x1, y1, x2, y2 = 10 + 30 * j, 10, 30 + 30 * j, 50
            obj = ET.SubElement(tree, "object")
            ET.SubElement(obj, "name").text = CLASSES[class_id]
            box = ET.SubElement(obj, "bndbox")
            for name, value in zip(("xmin", "ymin", "xmax", "ymax"), (x1, y1, x2, y2)):
                ET.SubElement(box, name).text = str(value)
            lines.append(f"{class_id} {(x1+x2)/200} {(y1+y2)/160} {(x2-x1)/100} {(y2-y1)/80}")
        if index == 4:
            obj = ET.SubElement(tree, "object")
            ET.SubElement(obj, "name").text = "dog"
            box = ET.SubElement(obj, "bndbox")
            for name, value in zip(("xmin", "ymin", "xmax", "ymax"), (70, 10, 90, 50)):
                ET.SubElement(box, name).text = str(value)
        ET.ElementTree(tree).write(root / "Annotations" / f"{key}.xml", encoding="utf-8")
        (root / "Labels" / f"{key}.txt").write_text("\n".join(lines), encoding="utf-8")
    # Corrupt training list contains every image repeatedly, including validation.
    (root / "ImageSets/Main/train.txt").write_text("\n".join([f"img{i}" for i in range(6)] * 3))
    (root / "ImageSets/Main/val.txt").write_text("img4\nimg5\nimg4\n")
    expected = {"images": 6, "xml": 6, "txt": 9, "train": 4, "val": 2,
                "darknet_instances": {"Aquafina": 3, "Deer": 2, "Kirkland": 1, "Nestle": 1}}
    return root, expected


def test_corrupt_train_reconstructed_and_validation_preserved(temple):
    root, expected = temple
    audit = audit_temple(root, expected)
    assert audit.audit["blocking_errors"] == 0
    assert audit.train_ids == ["img0", "img1", "img2", "img3"]
    assert audit.val_ids == ["img4", "img5"]
    assert not set(audit.train_ids) & set(audit.val_ids)
    split = audit.audit["splits"]
    assert split["original_train_entries"] == 18
    assert split["original_train_unique"] == 6
    assert split["original_train_duplicated_ids"] == 6
    assert split["original_train_val_overlap"] == 2
    assert split["reconstructed_overlap"] == 0
    # Membership is unchanged even when the corrupt train list is unrelated.
    (root / "ImageSets/Main/train.txt").write_text("not_a_dataset_id\n")
    again = audit_temple(root, expected)
    assert again.train_ids == audit.train_ids
    assert again.val_ids == audit.val_ids


def test_exact_production_split_counts():
    ids = [f"image{i:05}" for i in range(4870)]
    train, val = reconstruct_splits(ids, ids[:870] + ids[:100])
    assert len(train) == 4000 and len(val) == 870
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(ids)
    with pytest.raises(ValueError, match="Unknown validation"):
        reconstruct_splits(ids, ["missing"])
    with pytest.raises(ValueError, match="Expected"):
        reconstruct_splits(ids, ids[:869])


def test_audit_and_preview_read_only_dog_flagged(temple, tmp_path):
    root, expected = temple
    before = snapshot_raw(root)
    before_paths = set(tmp_path.rglob("*"))
    audit = audit_temple(root, expected)
    previews = preview_temple(audit)
    assert previews and "img4" in previews[0][0]
    assert audit.audit["dog_objects"] == 1
    assert any(i["kind"] == "dog_annotation" for i in audit.anomalies["issues"])
    assert audit.class_counts["darknet_instances"] == expected["darknet_instances"]
    assert audit.class_counts["xml_instances_by_original_name"]["dog"] == 1
    assert snapshot_raw(root) == before
    assert set(tmp_path.rglob("*")) == before_paths


def test_conversion_mapping_negatives_reports_and_immutability(temple, tmp_path):
    root, expected = temple
    before = snapshot_raw(root)
    audit = audit_temple(root, expected)
    output = tmp_path / "processed/temple"
    result = convert_temple(audit, output, confirm=True)
    assert result["raw_unchanged"] is True
    assert snapshot_raw(root) == before
    assert (root / "Annotations/img4.xml").read_text().count("dog") == 1
    assert not (output / "annotations/test.json").exists()
    assert set(verify_prepared(output)) == {"train", "val"}
    for name in ("audit", "anomaly_report", "class_counts", "train_manifest", "val_manifest"):
        assert (output / f"{name}.json").is_file()
    data = read_json(output / "annotations/train.json")
    assert len(data["images"]) == 4
    assert data["categories"] == [{"id": 1, "name": "aquafina_bottle"}]
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["category_id"] == 1  # YOLOX maps the sole category to model index 0.
    assert data["annotations"][0]["bbox"] == pytest.approx([10, 10, 20, 40])
    negatives = [im for im in data["images"] if im["subset"] == "competitor"]
    assert len(negatives) == 3
    assert all(not any(a["image_id"] == im["id"] for a in data["annotations"]) for im in negatives)
    val = read_json(output / "annotations/val.json")
    assert len(val["annotations"]) == 2  # no Deer or dog boxes
    assert val["images"][0]["subset"] == "mixed_brand"
    assert read_json(output / "conversion.json")["raw_sha256_before_after_equal"]
    with pytest.raises(FileExistsError):
        convert_temple(audit, output, confirm=True)
    (output / "conversion.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        verify_prepared(output)


def test_dog_cannot_become_aquafina_if_darknet_has_box(temple, tmp_path):
    root, expected = temple
    # Known XML dog corresponds to an unmatched class-0 label: flag and exclude.
    path = root / "Labels/img4.txt"
    path.write_text(path.read_text() + "\n0 0.8 0.375 0.2 0.5\n")
    expected["darknet_instances"]["Aquafina"] += 1
    audit = audit_temple(root, expected)
    assert audit.audit["blocking_errors"] == 0
    assert audit.class_counts["darknet_instances"]["Aquafina"] == 4
    assert audit.class_counts["aquafina_instances_to_convert"] == 3
    assert any(i["kind"] == "dog_darknet_exclusion" for i in audit.anomalies["issues"])
    output = tmp_path / "processed"
    convert_temple(audit, output, confirm=True)
    assert len(read_json(output / "annotations/val.json")["annotations"]) == 2


@pytest.mark.parametrize("problem,kind", [
    ("class", "xml_darknet_mismatch"), ("bbox", "darknet_invalid_line"),
    ("dimensions", "xml_image_dimensions"), ("missing", "missing_pair"),
    ("xml", "xml_parse_error"), ("xml_bbox", "xml_bbox"), ("decode", "image_decode"),
])
def test_invalid_source_blocks_conversion_without_writes(temple, tmp_path, problem, kind):
    root, expected = temple
    if problem == "class":
        (root / "Labels/img0.txt").write_text("1 0.2 0.375 0.2 0.5")
    elif problem == "bbox":
        (root / "Labels/img0.txt").write_text("0 0.99 0.375 0.5 0.5")
    elif problem in {"dimensions", "xml_bbox"}:
        path = root / "Annotations/img0.xml"
        text = path.read_text()
        text = text.replace("<width>100</width>", "<width>101</width>") if problem == "dimensions" else text.replace("<xmax>30</xmax>", "<xmax>130</xmax>")
        path.write_text(text)
    elif problem == "missing":
        (root / "Labels/img0.txt").unlink()
    elif problem == "xml":
        (root / "Annotations/img0.xml").write_text("<invalid")
    elif problem == "decode":
        (root / "JPEGImages/img0.jpg").write_bytes(b"broken")
    audit = audit_temple(root, expected)
    assert audit.audit["blocking_errors"] > 0
    assert any(i["kind"] == kind for i in audit.anomalies["issues"])
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match="blocking"):
        convert_temple(audit, output, confirm=True)
    assert not output.exists()


def test_confirmation_raw_destination_and_stale_audit_guards(temple, tmp_path):
    root, expected = temple
    audit = audit_temple(root, expected)
    output = tmp_path / "processed"
    with pytest.raises(PermissionError):
        convert_temple(audit, output)
    assert not output.exists()
    with pytest.raises(ValueError, match="disjoint"):
        convert_temple(audit, root / "processed", confirm=True)
    (root / "Readme.txt").write_text("Changed externally after audit")
    with pytest.raises(ValueError, match="changed after audit"):
        convert_temple(audit, output, confirm=True)
    assert not output.exists()


def test_identical_image_content_across_original_splits_blocks(temple):
    root, expected = temple
    (root / "JPEGImages/img5.jpg").write_bytes((root / "JPEGImages/img0.jpg").read_bytes())
    audit = audit_temple(root, expected)
    assert audit.audit["splits"]["reconstructed_overlap"] == 0
    assert any(i["kind"] == "cross_split_duplicate_content" for i in audit.anomalies["issues"])


def test_notebook_has_readonly_order_and_exact_paths():
    repo = Path(__file__).resolve().parents[1]
    doc = json.loads((repo / "notebooks/01_prepare_data.ipynb").read_text())
    code = ["".join(cell["source"]) for cell in doc["cells"] if cell["cell_type"] == "code"]
    assert "raw/temple/detection_dataset.tar.gz" in code[0]
    assert "/content/temple_stage/detection_dataset" in code[0]
    assert "fca7260d4785af1dec18aa320fa9fc4a" in code[0]
    assert "/content/drive/MyDrive/aquafina-yolo/processed/temple" in code[0]
    assert "sys.dont_write_bytecode = True" in code[0]
    assert "stage_archive(ARCHIVE" in code[1]
    assert "audit_staged(STAGE" in code[2]
    assert "REUSE_AUDIT_CACHE = True" in code[2]
    assert "preview_temple(AUDIT" in code[3]
    assert "CONFIRM_CONVERSION = False" in code[4]
    assert "convert_temple(AUDIT" in code[4]
    before_conversion = "\n".join(code[:4])
    assert all(token not in before_conversion for token in ("write_json(", ".mkdir(", "pip", "annotations.json"))
