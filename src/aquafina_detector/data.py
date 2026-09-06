"""Validate a one-class COCO export and prepare leakage-safe grouped splits."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import shutil

from . import CLASS_NAME
from .common import contained, read_json, sha256, write_json

SUBSETS = {"visible_brand", "mixed_brand", "competitor", "ambiguous", "empty"}
SPLITS = ("train", "val", "test")


def validate(data, image_root=None):
    if data.get("categories") != [{"id": 1, "name": CLASS_NAME}]:
        raise ValueError("categories must be exactly [{id: 1, name: aquafina_bottle}]")
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    if not images:
        raise ValueError("Dataset contains no images")
    by_id, filenames = {}, set()
    for im in images:
        key = im.get("id")
        if type(key) is not int or key in by_id:
            raise ValueError("Image IDs must be unique integers")
        if any(type(im.get(k)) is not int or im[k] <= 0 for k in ("width", "height")):
            raise ValueError(f"Invalid dimensions for image {key}")
        if not isinstance(im.get("group_id"), str) or not im["group_id"].strip():
            raise ValueError(f"Image {key} needs a capture/source group_id")
        if im.get("subset") not in SUBSETS:
            raise ValueError(f"Image {key} needs subset in {sorted(SUBSETS)}")
        if not isinstance(im.get("scene"), str) or not im["scene"].strip():
            raise ValueError(f"Image {key} needs scene (e.g. shelf, desk, outdoor)")
        filename = im.get("file_name")
        path = contained(image_root or Path.cwd(), filename)
        if filename.casefold() in filenames:
            raise ValueError(f"Repeated filename: {filename}")
        filenames.add(filename.casefold())
        if image_root is not None:
            from PIL import Image
            with Image.open(path) as decoded:
                decoded.load()
                if decoded.size != (im["width"], im["height"]):
                    raise ValueError(f"Recorded dimensions differ from image: {filename}")
        by_id[key] = im
    annotation_ids, counts = set(), Counter()
    for ann in annotations:
        key = ann.get("id")
        if type(key) is not int or key <= 0 or key in annotation_ids:
            raise ValueError("Annotation IDs must be unique positive integers (COCO uses 0 for unmatched)")
        annotation_ids.add(key)
        if ann.get("image_id") not in by_id or ann.get("category_id") != 1:
            raise ValueError(f"Unknown image/category in annotation {key}")
        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in bbox):
            raise ValueError(f"Invalid bbox in annotation {key}")
        x, y, w, h = bbox
        im = by_id[ann["image_id"]]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > im["width"] or y + h > im["height"]:
            raise ValueError(f"Out-of-bounds/empty bbox in annotation {key}")
        if ann.get("iscrowd", 0) != 0:
            raise ValueError("Crowd annotations unsupported; label individual visible bottles")
        if "area" in ann and not math.isclose(ann["area"], w * h, rel_tol=1e-5):
            raise ValueError("Detection annotation area must equal bbox width * height")
        counts[ann["image_id"]] += 1
    for key, im in by_id.items():
        positive = im["subset"] in {"visible_brand", "mixed_brand"}
        if positive != (counts[key] > 0):
            raise ValueError(f"Image {key}: subset conflicts with positive annotation count")
        if counts[key] > 50:
            raise ValueError("More than 50 bottles in one image exceeds training label capacity")
    return {"images": len(images), "annotations": len(annotations),
            "negative_images": sum(counts[k] == 0 for k in by_id),
            "subsets": dict(Counter(i["subset"] for i in images))}


def grouped_split(data, hashes=None, seed=42):
    """Join source groups sharing exact file hashes, then stratify whole components.

    Try deterministic randomized allocations; fail instead of silently emitting a
    split without positives or competitor negatives. Rare scenes are reported.
    """
    validate(data)
    hashes = hashes or {}
    parent = {im["group_id"]: im["group_id"] for im in data["images"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for im in data["images"]:
        digest = hashes.get(im["id"])
        if digest is not None:
            if digest in seen:
                parent[find(im["group_id"])] = find(seen[digest])
            seen[digest] = im["group_id"]
    groups = defaultdict(list)
    for im in data["images"]:
        groups[find(im["group_id"] )].append(im)
    if len(groups) < 3:
        raise ValueError("Need at least three independent groups after duplicate merging")
    rng, best = random.Random(seed), None
    ratios = (0.70, 0.15, 0.15)
    total = Counter()
    features = {}
    for key, ims in groups.items():
        c = Counter({"total": len(ims)})
        for im in ims:
            c["subset:" + im["subset"]] += 1
            c["scene:" + im["scene"]] += 1
            c["positive" if im["subset"] in {"visible_brand", "mixed_brand"} else "negative"] += 1
        features[key] = c
        total.update(c)
    for _ in range(2000):
        allocation = {name: [] for name in SPLITS}
        for key in groups:
            allocation[rng.choices(SPLITS, ratios)[0]].append(key)
        counts = {s: sum((features[k] for k in keys), Counter()) for s, keys in allocation.items()}
        if any(c["positive"] == 0 or c["subset:competitor"] == 0 for c in counts.values()):
            continue
        score = sum(((counts[s][f] - ratio * n) / max(n, 1)) ** 2
                    for s, ratio in zip(SPLITS, ratios) for f, n in total.items())
        if best is None or score < best[0]:
            best = score, allocation
    if best is None:
        raise ValueError("Cannot produce splits with positives and competitor images in each; collect more independent groups")
    output = {}
    for split, keys in best[1].items():
        ims = [im for k in keys for im in groups[k]]
        ids = {im["id"] for im in ims}
        output[split] = {"info": {"split": split, "seed": seed},
                         "categories": data["categories"], "images": ims,
                         "annotations": [dict(a, area=a["bbox"][2] * a["bbox"][3], iscrowd=0)
                                         for a in data["annotations"] if a["image_id"] in ids]}
    return output


def prepare(annotations, image_root, output, seed=42):
    data = read_json(annotations)
    validate(data, image_root)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new version directory; prepared data is immutable")
    hashes = {im["id"]: sha256(contained(image_root, im["file_name"])) for im in data["images"]}
    splits = grouped_split(data, hashes, seed)
    manifest = {"seed": seed, "source_sha256": sha256(annotations), "splits": {}}
    for name, part in splits.items():
        for im in part["images"]:
            dst = contained(output / (name + "2017"), im["file_name"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(contained(image_root, im["file_name"]), dst)
        write_json(output / "annotations" / f"{name}.json", part)
        manifest["splits"][name] = {"summary": validate(part),
            "scenes": dict(Counter(im["scene"] for im in part["images"])),
            "images": [{"id": im["id"], "group_id": im["group_id"], "sha256": hashes[im["id"]]}
                       for im in part["images"]]}
    write_json(output / "split_manifest.json", manifest)
    return manifest


def verify_prepared(root, *, require_complete=True):
    """Check immutable files and cross-split leakage before any GPU work."""
    root = Path(root)
    manifest = read_json(root / "split_manifest.json")
    if manifest.get("source") == "TempleRAIL" and require_complete:
        marker = root / "conversion.json"
        if not marker.is_file() or read_json(marker).get("status") != "complete":
            raise ValueError("TempleRAIL conversion is incomplete: missing complete conversion.json")
    group_owner, hash_owner, image_owner = {}, {}, {}
    annotation_hashes = {}
    available = set(manifest["splits"])
    if available not in ({"train", "val"}, {"train", "val", "test"}):
        raise ValueError("Prepared data must contain train/val with an optional independent test split")
    for name in (s for s in SPLITS if s in available):
        path = root / "annotations" / f"{name}.json"
        part = read_json(path)
        validate(part, root / (name + "2017"))
        if part.get("info", {}).get("split") != name:
            raise ValueError(f"Wrong split metadata in {path}")
        entries = manifest["splits"][name]["images"]
        expected = {entry["id"]: entry for entry in entries}
        if len(expected) != len(entries) or set(expected) != {im["id"] for im in part["images"]}:
            raise ValueError("Manifest/image membership differs")
        for im in part["images"]:
            group = im["group_id"]
            digest = sha256(contained(root / (name + "2017"), im["file_name"]))
            if expected[im["id"]]["sha256"] != digest or expected[im["id"]]["group_id"] != group:
                raise ValueError("Prepared image/group changed since manifest creation")
            for key, owners in ((group, group_owner), (digest, hash_owner), (im["id"], image_owner)):
                if owners.setdefault(key, name) != name:
                    raise ValueError("Cross-split group, duplicate, or image-ID leakage")
        annotation_hashes[name] = sha256(path)
    return annotation_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", help="Omit for validation only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare(args.annotations, args.images, args.output, args.seed) if args.output
          else validate(read_json(args.annotations), args.images))


if __name__ == "__main__":
    main()
