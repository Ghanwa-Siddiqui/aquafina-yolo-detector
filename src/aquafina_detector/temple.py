"""Read-only TempleRAIL audit and explicitly confirmed canonical conversion.

No training dependencies. Audit and preview return in-memory values only.
Raw files are never opened for writing; conversion copies images to a disjoint root.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from . import CLASS_NAME
from .common import contained, sha256, write_json
from .data import validate, verify_prepared

TEMPLE_ROOT = Path("/content/drive/MyDrive/aquafina-yolo/raw/temple/extracted/detection_dataset")
PROCESSED_ROOT = Path("/content/drive/MyDrive/aquafina-yolo/processed/temple")
CLASSES = {0: "Aquafina", 1: "Deer", 2: "Kirkland", 3: "Nestle"}
EXPECTED = {"images": 4870, "xml": 4870, "txt": 4873, "train": 4000, "val": 870,
            "darknet_instances": {"Aquafina": 5227, "Deer": 4326, "Kirkland": 3552, "Nestle": 3735}}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
PIXEL_TOLERANCE = 2.0  # VOC integer coordinates vs rounded normalized labels.
PROGRESS_EVERY = 250
DUPLICATE_POLICY_VERSION = "validation-priority-discard-ambiguous-training-v2"


def _annotation_signature(record, source):
    # Compare complete annotations, including competitors and dog, ignoring order.
    # Across identical images require exact parsed geometry; the 2-pixel VOC
    # tolerance applies only to XML/Darknet pairing within an individual image.
    return sorted((str(obj.get("name", obj["class_id"])).casefold(),
                   tuple(obj["bbox_xyxy"])) for obj in record[source])


def resolve_duplicates(records, train_ids, val_ids, issue):
    """Preserve unique validation copies; discard conflicting training-only groups."""
    train, val = set(train_ids), set(val_ids)
    groups = defaultdict(list)
    for key in sorted(train | val):
        if key in records:
            groups[records[key]["sha256"]].append(key)
    excluded, crossing = [], 0
    for digest, ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        training, validation = sorted(train.intersection(ids)), sorted(val.intersection(ids))
        crossing += bool(training and validation)
        details = {"sha256": digest, "image_ids": ids}
        if len(validation) > 1:
            issue("multiple_validation_duplicates", "error", **details)
            continue
        if train & val or any(not records[key].get("paired_annotations_verified") for key in ids):
            issue("ambiguous_duplicate_group", "error", **details)
            continue
        first = records[ids[0]]
        equivalent = all(
            (records[key]["width"], records[key]["height"]) == (first["width"], first["height"])
            and all(_annotation_signature(records[key], source) == _annotation_signature(first, source)
                    for source in ("xml_objects", "darknet_labels")) for key in ids[1:])
        if not equivalent:
            issue("conflicting_duplicate_annotations", "warning", **details,
                  action="Exclude training members; preserve unique validation representative if present")
        if training and (validation or not equivalent):
            counterpart = validation[0] if validation else None
            reason = ("Exact-content duplicate; preserve validation annotations and exclude training copy"
                      if validation else "Train-only conflicting annotations; exclude every group member")
            for key in training:
                excluded.append({"image_id": key, "retained_validation_id": counterpart,
                                 "sha256": digest, "reason": reason,
                                 "scope": "cross_split" if validation else "train_only",
                                 "annotations_conflict": not equivalent})
            issue("cross_split_duplicate_resolved" if validation else "train_only_conflict_excluded",
                  "warning", **details, retained_validation_id=counterpart, excluded_training_ids=training)
    excluded.sort(key=lambda item: item["image_id"])
    remaining = sorted(train - {item["image_id"] for item in excluded})
    overlap = {records[k]["sha256"] for k in remaining if k in records} & {
        records[k]["sha256"] for k in val if k in records}
    return remaining, excluded, crossing, len(overlap)


@dataclass
class TempleAudit:
    root: Path
    records: dict
    train_ids: list
    val_ids: list
    snapshot: dict
    audit: dict
    anomalies: dict
    class_counts: dict


def snapshot_raw(root):
    """Content hashes of every raw file, including XML, TXT and Readme.txt."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Raw symlinks/external paths are unsupported: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def read_split_ids(path):
    """Accept IDs or image paths and optional VOC membership columns."""
    ids = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            token = line.split()[0].replace("\\", "/")
            item = Path(token)
            ids.append(item.stem if item.suffix.lower() in IMAGE_SUFFIXES else item.name)
    return ids


def reconstruct_splits(all_image_ids, validation_ids, expected_train=4000, expected_val=870):
    """train.txt deliberately is NOT an argument to split reconstruction."""
    all_ids, val = set(all_image_ids), set(validation_ids)
    if not val.issubset(all_ids):
        raise ValueError(f"Unknown validation IDs: {sorted(val - all_ids)}")
    train = all_ids - val
    if train & val:
        raise ValueError("Train/validation overlap must be zero")
    if len(train) != expected_train or len(val) != expected_val:
        raise ValueError(f"Expected {expected_train}/{expected_val} train/val, got {len(train)}/{len(val)}")
    return sorted(train), sorted(val)


def _index(directory, suffixes, issue):
    result, folded = {}, set()
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        key = path.stem
        if key.casefold() in folded:
            issue("duplicate_file_id", "error", path=str(path), image_id=key)
        folded.add(key.casefold())
        result[key] = path
    return result


def _xml(path, width, height, issue, key):
    try:
        tree = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        issue("xml_parse_error", "error", image_id=key, detail=str(exc))
        return []
    filename = (tree.findtext("filename") or "").strip().replace("\\", "/")
    if filename and Path(filename).stem != key:
        issue("xml_filename_pairing", "error", image_id=key, xml_filename=filename)
    try:
        dims = (int(tree.findtext("size/width")), int(tree.findtext("size/height")))
        if dims != (width, height):
            issue("xml_image_dimensions", "error", image_id=key, xml=list(dims), decoded=[width, height])
    except (ValueError, TypeError):
        issue("xml_image_dimensions", "error", image_id=key, detail="Missing/invalid size")
    objects = []
    names = {name.casefold(): class_id for class_id, name in CLASSES.items()}
    for index, obj in enumerate(tree.findall("object")):
        name = (obj.findtext("name") or "").strip()
        lowered = name.casefold()
        if lowered == "dog":
            issue("dog_annotation", "warning", image_id=key, object_index=index,
                  action="Exclude this XML object and any otherwise-unmatched Darknet box at the same location")
        elif lowered not in names:
            issue("unknown_xml_class", "error", image_id=key, object_index=index, name=name)
        try:
            box = [float(obj.findtext("bndbox/" + k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            x1, y1, x2, y2 = box
            if (not all(math.isfinite(v) for v in box) or x1 < 0 or y1 < 0
                    or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height):
                raise ValueError("Out-of-range or empty XML bbox")
        except (ValueError, TypeError) as exc:
            issue("xml_bbox", "error", image_id=key, object_index=index, detail=str(exc))
            continue
        objects.append({"name": name, "class_id": names.get(lowered), "dog": lowered == "dog",
                        "bbox_xyxy": box, "object_index": index})
    return objects


def _darknet(path, width, height, issue, key):
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError("Expected class_id cx cy w h")
            class_id = int(parts[0])
            if class_id not in CLASSES:
                raise ValueError("Unknown Darknet class ID")
            cx, cy, bw, bh = map(float, parts[1:])
            if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
                raise ValueError("Non-finite coordinate")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError("Invalid normalized center/size")
            corners = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
            if any(v < -1e-6 or v > 1 + 1e-6 for v in corners):
                raise ValueError("BBox extends outside image")
            if any(v < 0 or v > 1 for v in corners):
                issue("darknet_boundary_rounding", "warning", image_id=key, line=line_number,
                      action="Clamp at most 1e-6 normalized rounding error in processed coordinates only")
            corners = [min(1, max(0, v)) for v in corners]
            box = [corners[0] * width, corners[1] * height, corners[2] * width, corners[3] * height]
            labels.append({"class_id": class_id, "bbox_xyxy": box, "line": line_number})
        except ValueError as exc:
            issue("darknet_invalid_line", "error", image_id=key, line=line_number, detail=str(exc))
    return labels


def _close(a, b):
    return max(abs(x - y) for x, y in zip(a, b)) <= PIXEL_TOLERANCE + 1e-9


def _crosscheck(objects, labels, issue, key):
    """One-to-one same-class geometry match; report any unexplained mismatch."""
    known = [obj for obj in objects if obj["class_id"] is not None]
    # Bipartite matching avoids false mismatches in overlapping/dense bottles.
    edges = {i: [j for j, label in enumerate(labels)
                 if obj["class_id"] == label["class_id"] and _close(obj["bbox_xyxy"], label["bbox_xyxy"])]
             for i, obj in enumerate(known)}
    matched = {}

    def assign(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in matched or assign(matched[j], visited):
                matched[j] = i
                return True
        return False

    for i, obj in enumerate(known):
        if not assign(i, set()):
            issue("xml_darknet_mismatch", "error", image_id=key, xml_object=obj,
                  detail="No one-to-one same-class Darknet box within 2 pixels")
    dogs = [obj for obj in objects if obj["dog"]]
    excluded = set()
    for j, label in enumerate(labels):
        if j in matched:
            continue
        dog = next((obj for obj in dogs if _close(obj["bbox_xyxy"], label["bbox_xyxy"])), None)
        if dog is not None:
            excluded.add(j)
            dogs.remove(dog)
            issue("dog_darknet_exclusion", "warning", image_id=key, darknet_line=label["line"],
                  darknet_class=CLASSES[label["class_id"]], action="Excluded from conversion")
        else:
            issue("xml_darknet_mismatch", "error", image_id=key, darknet_label=label,
                  detail="Darknet box has no matching known XML object or dog exception")
    return [label for j, label in enumerate(labels) if j not in excluded]


def audit_temple(root=TEMPLE_ROOT, expected=None, *, progress=None, _snapshot=None):
    """Inspect files read-only. `expected` overrides verified counts for fixtures."""
    from PIL import Image
    root = Path(root).resolve()
    expected = EXPECTED if expected is None else expected
    snapshot = snapshot_raw(root) if _snapshot is None else dict(_snapshot)
    issues = []

    def issue(kind, severity, **details):
        issues.append({"kind": kind, "severity": severity, **details})

    for directory in ("JPEGImages", "Annotations", "Labels", "ImageSets/Main"):
        if not (root / directory).is_dir():
            issue("missing_directory", "error", path=directory)
    if not (root / "Readme.txt").is_file():
        issue("missing_readme", "warning")
    images = _index(root / "JPEGImages", IMAGE_SUFFIXES, issue)
    xml = _index(root / "Annotations", {".xml"}, issue)
    labels = _index(root / "Labels", {".txt"}, issue)
    for name, mapping in (("xml", xml), ("darknet", labels)):
        for key in sorted(images.keys() - mapping.keys()):
            issue("missing_pair", "error", image_id=key, missing=name)
        for key in sorted(mapping.keys() - images.keys()):
            issue("orphan_annotation", "error", image_id=key, source=name)
    counts = {"images": len(images), "xml": len(xml), "darknet_labels": len(labels),
              "txt": sum(Path(name).suffix.lower() == ".txt" for name in snapshot)}
    for name in ("images", "xml", "txt"):
        if counts[name] != expected[name]:
            issue("count_mismatch", "error", field=name, expected=expected[name], actual=counts[name])
    records, dn_counts, xml_counts, kept_counts = {}, Counter(), Counter(), Counter()
    if progress:
        progress(f"Auditing {len(images):,} images (local decoding, XML and Darknet checks)...")
    for position, (key, image_path) in enumerate(images.items(), 1):
        if progress and position > 1 and (position - 1) % PROGRESS_EVERY == 0:
            progress(f"Audited {position - 1:,}/{len(images):,} images")
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                if image.getexif().get(274, 1) != 1:
                    issue("exif_orientation", "error", image_id=key, detail="Normalize orientation in a separate reviewed workflow")
        except (OSError, ValueError) as exc:
            issue("image_decode", "error", image_id=key, detail=str(exc))
            continue
        if key not in xml or key not in labels:
            continue
        objects = _xml(xml[key], width, height, issue, key)
        try:
            darknet = _darknet(labels[key], width, height, issue, key)
        except (UnicodeError, OSError) as exc:
            issue("darknet_read_error", "error", image_id=key, detail=str(exc))
            darknet = []
        xml_counts.update(obj["name"] for obj in objects)
        dn_counts.update(CLASSES[label["class_id"]] for label in darknet)
        kept = _crosscheck(objects, darknet, issue, key)
        kept_counts.update(CLASSES[label["class_id"]] for label in kept)
        aquafina = [label for label in kept if label["class_id"] == 0]
        competitor = any(label["class_id"] != 0 for label in kept)
        if len(aquafina) > 50:
            issue("positive_capacity", "error", image_id=key, count=len(aquafina))
        subset = ("mixed_brand" if competitor else "visible_brand") if aquafina else (
            "competitor" if competitor else "empty")
        records[key] = {"source_id": key, "file_name": image_path.relative_to(root / "JPEGImages").as_posix(),
                        "width": width, "height": height, "labels": kept, "xml_objects": objects,
                        "darknet_labels": darknet,
                        "paired_annotations_verified": not any(i["severity"] == "error" and i.get("image_id") == key for i in issues),
                        "subset": subset, "sha256": snapshot[image_path.relative_to(root).as_posix()]}
    if progress:
        progress(f"Audited {len(images):,}/{len(images):,} images; checking classes and splits...")
    for name, count in expected.get("darknet_instances", {}).items():
        if dn_counts[name] != count:
            issue("class_count_mismatch", "error", name=name, expected=count, actual=dn_counts[name])
    original_train, original_val = [], []
    for name in ("train", "val"):
        try:
            ids = read_split_ids(root / "ImageSets/Main" / f"{name}.txt")
            if name == "train":
                original_train = ids
            else:
                original_val = ids
        except (OSError, UnicodeError) as exc:
            issue("split_read_error", "error" if name == "val" else "warning", split=name, detail=str(exc))
    train_counter = Counter(original_train)
    split_report = {"original_train_entries": len(original_train), "original_train_unique": len(train_counter),
                    "original_train_duplicate_entries": len(original_train) - len(train_counter),
                    "original_train_duplicated_ids": sum(n > 1 for n in train_counter.values()),
                    "original_train_val_overlap": len(set(original_train) & set(original_val)),
                    "original_validation_entries": len(original_val), "original_validation_unique": len(set(original_val)),
                    "original_train_unknown_ids": sorted(set(original_train) - set(images)),
                    "policy": "Ignore train.txt membership; train = JPEGImages IDs minus unique val.txt IDs"}
    issue("untrusted_train_txt", "warning", **split_report)
    try:
        train_ids, val_ids = reconstruct_splits(images, original_val, expected["train"], expected["val"])
    except ValueError as exc:
        issue("split_reconstruction", "error", detail=str(exc))
        train_ids, val_ids = [], []
    split_report.update({"train_count": len(train_ids), "val_count": len(val_ids),
                         "reconstructed_overlap": len(set(train_ids) & set(val_ids)), "test_count": 0})
    train_ids, exclusions, crossing, content_overlap = resolve_duplicates(records, train_ids, val_ids, issue)
    split_report.update({"policy": split_report["policy"] + "; preserve unique validation copies, exclude their training duplicates, discard all train-only conflicting members",
                         "duplicate_policy_version": DUPLICATE_POLICY_VERSION,
                         "original_validation_ids": sorted(set(original_val)),
                         "processed_train_count": len(train_ids), "processed_val_count": len(val_ids),
                         "cross_split_groups_before_policy": crossing,
                         "excluded_training_count": len(exclusions),
                         "cross_split_duplicate_content": content_overlap,
                         "expected_processed_train": split_report["train_count"] - len(exclusions),
                         "expected_processed_val": len(set(original_val))})
    if content_overlap:
        issue("cross_split_duplicate_content", "error", groups=content_overlap)
    processed_records = [records[k] for k in train_ids + val_ids if k in records]
    processed_counts = Counter(CLASSES[label["class_id"]] for record in processed_records for label in record["labels"])
    summary = {"source": "TempleRAIL", "raw_root": str(root), "counts": counts,
               "pairing": {"missing_or_orphan": sum(i["kind"] in {"missing_pair", "orphan_annotation"} for i in issues)},
               "splits": split_report,
               "xml_parsing_errors": sum(i["kind"] == "xml_parse_error" for i in issues),
               "darknet_invalid_lines": sum(i["kind"] == "darknet_invalid_line" for i in issues),
               "image_decode_errors": sum(i["kind"] == "image_decode" for i in issues),
               "dimension_errors": sum(i["kind"] == "xml_image_dimensions" for i in issues),
               "bounding_box_errors": sum(i["kind"] in {"xml_bbox", "darknet_invalid_line"} for i in issues),
               "dog_objects": sum(i["kind"] == "dog_annotation" for i in issues),
               "xml_darknet_mismatches": sum(i["kind"] == "xml_darknet_mismatch" for i in issues),
               "blocking_errors": sum(i["severity"] == "error" for i in issues),
               "warnings": sum(i["severity"] == "warning" for i in issues),
               "status": "blocked" if any(i["severity"] == "error" for i in issues) else "ready_for_preview",
               "raw_files_hashed": len(snapshot), "writes_performed": False}
    class_counts = {"darknet_class_mapping": CLASSES, "darknet_instances": dict(dn_counts),
                    "xml_instances_by_original_name": dict(xml_counts),
                    "after_dog_exclusion": dict(kept_counts),
                    "after_duplicate_exclusion": dict(processed_counts),
                    "aquafina_instances_to_convert": processed_counts["Aquafina"],
                    "canonical_mapping": {"source_darknet": 0, "model_class": 0, "coco_category": 1},
                    "image_subsets": dict(Counter(r["subset"] for r in processed_records))}
    return TempleAudit(root, records, train_ids, val_ids, snapshot, summary,
                       {"issues": issues, "raw_xml_modified": False,
                        "duplicate_policy_version": DUPLICATE_POLICY_VERSION,
                        "excluded_training_images": exclusions}, class_counts)


def preview_temple(result, limit=8):
    """Return (caption, PIL image) pairs in memory; never save thumbnails."""
    from PIL import Image, ImageDraw
    if limit < 1:
        raise ValueError("Preview limit must be positive")
    selected = []
    selectors = [lambda r: any(obj["dog"] for obj in r["xml_objects"])]
    selectors += [lambda r, s=s: r["subset"] == s for s in ("mixed_brand", "visible_brand", "competitor", "empty")]
    for select in selectors:
        key = next((k for k, r in result.records.items() if select(r) and k not in selected), None)
        if key is not None:
            selected.append(key)
    selected += [k for k in result.records if k not in selected]
    previews = []
    for key in selected[:limit]:
        record = result.records[key]
        with Image.open(contained(result.root / "JPEGImages", record["file_name"])) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for label in record["labels"]:
            color = "lime" if label["class_id"] == 0 else "orange"
            draw.rectangle(label["bbox_xyxy"], outline=color, width=3)
            draw.text(label["bbox_xyxy"][:2], CLASSES[label["class_id"]], fill=color)
        for obj in record["xml_objects"]:
            if obj["dog"]:
                draw.rectangle(obj["bbox_xyxy"], outline="red", width=3)
                draw.text(obj["bbox_xyxy"][:2], "dog: EXCLUDED", fill="red")
        image.thumbnail((800, 600))
        previews.append((f"{key} | {record['subset']} | green=Aquafina orange=background red=excluded", image))
    return previews


def convert_temple(result, output=PROCESSED_ROOT, *, confirm=False):
    """Write processed files only after explicit confirmation and a clean audit."""
    if confirm is not True:
        raise PermissionError("Conversion requires confirm=True after audit and preview review")
    output = Path(output).resolve()
    if output == result.root or output.is_relative_to(result.root) or result.root.is_relative_to(output):
        raise ValueError("Processed output must be disjoint from the entire raw dataset directory")
    if result.audit.get("original_archive"):
        original_raw = Path(result.audit["original_archive"]).resolve().parent
        if output.is_relative_to(original_raw) or original_raw.is_relative_to(output):
            raise ValueError("Processed output must not overlap the original Drive raw directory")
    if output.exists():
        raise FileExistsError("Processed output exists; use a new version directory, never overwrite")
    if result.audit["blocking_errors"]:
        raise ValueError("Audit has blocking errors; inspect the in-memory anomaly report before converting")
    if snapshot_raw(result.root) != result.snapshot:
        raise ValueError("Raw data changed after audit; rerun audit and preview")
    splits = result.audit["splits"]
    excluded = result.anomalies["excluded_training_images"]
    content_overlap = {result.records[k]["sha256"] for k in result.train_ids} & {
        result.records[k]["sha256"] for k in result.val_ids}
    if (splits["duplicate_policy_version"] != DUPLICATE_POLICY_VERSION
            or len(result.train_ids) != splits["expected_processed_train"]
            or len(result.val_ids) != splits["expected_processed_val"]
            or sorted(result.val_ids) != splits["original_validation_ids"]
            or set(result.train_ids) & set(result.val_ids) or content_overlap
            or {item["image_id"] for item in excluded} & set(result.train_ids + result.val_ids)):
        raise ValueError("Duplicate policy completion checks failed before conversion")
    numeric_ids = {key: i + 1 for i, key in enumerate(sorted(result.records))}
    datasets, manifests = {}, {}
    for split, ids in (("train", result.train_ids), ("val", result.val_ids)):
        data = {"info": {"source": "TempleRAIL", "split": split, "split_policy": result.audit["splits"]["policy"]},
                "categories": [{"id": 1, "name": CLASS_NAME}], "images": [], "annotations": []}
        for key in ids:
            r = result.records[key]
            im = {"id": numeric_ids[key], "source_id": key, "file_name": r["file_name"],
                  "width": r["width"], "height": r["height"], "subset": r["subset"],
                  "group_id": "temple-sha256:" + r["sha256"], "scene": "temple_unspecified",
                  "visibility_review": "source_labels_not_independently_verified"}
            data["images"].append(im)
            for label in r["labels"]:
                if label["class_id"] != 0:
                    continue
                x1, y1, x2, y2 = label["bbox_xyxy"]
                data["annotations"].append({"id": len(data["annotations"]) + 1, "image_id": im["id"],
                    "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        summary = validate(data)
        datasets[split] = data
        manifests[split] = {"split": split, "source_ids": ids, "summary": summary,
            "images": [{"id": im["id"], "source_id": im["source_id"], "file_name": im["file_name"],
                        "group_id": im["group_id"], "sha256": result.records[im["source_id"]]["sha256"]}
                       for im in data["images"]]}
    # All validation above precedes the first write. No symlinks into raw data.
    output.mkdir(parents=True, exist_ok=False)
    for split, data in datasets.items():
        for im in data["images"]:
            source = contained(result.root / "JPEGImages", im["file_name"])
            destination = contained(output / (split + "2017"), im["file_name"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        write_json(output / "annotations" / f"{split}.json", data)
        write_json(output / f"{split}_manifest.json", manifests[split])
    write_json(output / "split_manifest.json", {"source": "TempleRAIL", "splits": manifests,
               "test_split": None, "split_policy": result.audit["splits"]["policy"]})
    write_json(output / "audit.json", result.audit)
    write_json(output / "anomaly_report.json", result.anomalies)
    write_json(output / "class_counts.json", result.class_counts)
    verify_prepared(output, require_complete=False)
    if snapshot_raw(result.root) != result.snapshot:
        raise RuntimeError("Raw files changed during conversion; output is not verified")
    write_json(output / "conversion.json", {"status": "complete", "raw_sha256_before_after_equal": True,
               "train_images": len(result.train_ids), "val_images": len(result.val_ids),
               "source_file_sha256": result.snapshot,
               "duplicate_policy_version": DUPLICATE_POLICY_VERSION,
               "excluded_training_images": len(excluded),
               "validation_ids_preserved": True, "cross_split_duplicate_content": 0})
    return {"output": str(output), "train": manifests["train"]["summary"],
            "val": manifests["val"]["summary"], "raw_unchanged": True, "test_split": None}
