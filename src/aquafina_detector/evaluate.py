"""Evaluate COCO predictions, select on validation, then freeze for test."""
import argparse
from collections import defaultdict
import math
from itertools import groupby
from pathlib import Path

from .common import read_json, sha256, write_json
from .data import validate


def iou(a, b):
    x = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    intersection = x * y
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union else 0.0


def validate_predictions(data, predictions):
    ids = {im["id"] for im in data["images"]}
    for p in predictions:
        if p.get("image_id") not in ids or p.get("category_id") != 1:
            raise ValueError("Prediction contains an unknown image or category")
        score = p.get("score")
        box = p.get("bbox", [])
        if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid prediction confidence")
        if len(box) != 4 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in box):
            raise ValueError("Invalid prediction box")
        if box[2] <= 0 or box[3] <= 0:
            raise ValueError("Prediction box must have positive extent")


def operating_metrics(data, predictions, threshold, match_iou=0.5):
    """Greedy score-ordered, one-to-one matching at IoU 0.5."""
    truth, detected = defaultdict(list), defaultdict(list)
    for a in data["annotations"]:
        truth[a["image_id"]].append(a["bbox"])
    for p in predictions:
        if p["score"] >= threshold:
            detected[p["image_id"]].append(p)
    rows = []
    for im in data["images"]:
        gt = truth[im["id"]]
        used, tp, fp = set(), 0, 0
        for p in sorted(detected[im["id"]], key=lambda x: -x["score"]):
            candidates = [(iou(p["bbox"], box), j) for j, box in enumerate(gt) if j not in used]
            overlap, j = max(candidates, default=(0, -1))
            if overlap >= match_iou:
                used.add(j)
                tp += 1
            else:
                fp += 1
        rows.append({"image_id": im["id"], "file_name": im["file_name"], "subset": im["subset"],
                     "tp": tp, "fp": fp, "fn": len(gt) - tp, "negative": not gt})

    def summarize(items):
        tp = sum(r["tp"] for r in items)
        fp = sum(r["fp"] for r in items)
        fn = sum(r["fn"] for r in items)
        negative = [r for r in items if r["negative"]]
        competitors = [r for r in items if r["subset"] == "competitor"]
        return {"images": len(items), "positive_instances": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "negative_images": len(negative), "competitor_images": len(competitors),
                "fp_per_negative_image": sum(r["fp"] for r in negative) / len(negative) if negative else None,
                "negative_image_fpr": sum(r["fp"] > 0 for r in negative) / len(negative) if negative else None,
                "competitor_image_fpr": sum(r["fp"] > 0 for r in competitors) / len(competitors) if competitors else None}
    return {"threshold": threshold, "iou": match_iou, "overall": summarize(rows),
            "subsets": {s: summarize([r for r in rows if r["subset"] == s])
                        for s in sorted({im["subset"] for im in data["images"]})},
            "errors": [r for r in rows if r["fp"] or r["fn"]]}


def select_threshold(data, predictions, minimum_precision=0.95, maximum_fpr=0.05):
    if data.get("info", {}).get("split") != "val":
        raise ValueError("Threshold selection is allowed only on the validation split")
    validate(data)
    validate_predictions(data, predictions)
    # Sweep once in descending score order. Adding lower-score detections cannot
    # change earlier greedy matches. This avoids a full evaluation per score.
    truth, used = defaultdict(list), defaultdict(set)
    for ann in data["annotations"]:
        truth[ann["image_id"]].append(ann["bbox"])
    competitors = {im["id"] for im in data["images"] if im["subset"] == "competitor"}
    competitor_hits, tp, fp, best = set(), 0, 0, None
    total = len(data["annotations"])
    for threshold, batch in groupby(sorted(predictions, key=lambda p: -p["score"]), key=lambda p: p["score"]):
        for p in batch:
            image_id = p["image_id"]
            candidates = [(iou(p["bbox"], box), j) for j, box in enumerate(truth[image_id]) if j not in used[image_id]]
            overlap, j = max(candidates, default=(0, -1))
            if overlap >= 0.5:
                used[image_id].add(j)
                tp += 1
            else:
                fp += 1
            if image_id in competitors:
                competitor_hits.add(image_id)
        precision = tp / (tp + fp)
        recall = tp / total if total else None
        fpr = len(competitor_hits) / len(competitors) if competitors else None
        if recall is not None and fpr is not None and precision >= minimum_precision and fpr <= maximum_fpr:
            key = (recall, precision, -fpr, threshold)
            if best is None or key > best[0]:
                best = key, threshold
    return {"status": "qualified" if best else "no_qualifying_threshold",
            "threshold": best[1] if best else None,
            "targets": {"minimum_precision": minimum_precision, "maximum_competitor_image_fpr": maximum_fpr},
            "operating_report": operating_metrics(data, predictions, best[1]) if best else None,
            "diagnostic_report": operating_metrics(data, predictions, 0.5) if best is None else None}


def coco_ap(data, predictions):
    """Use the official COCO evaluator over all images, including negatives."""
    if not data["annotations"]:
        return {"ap50": None, "ap50_95": None}
    if not predictions:
        return {"ap50": 0.0, "ap50_95": 0.0}
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = dict(data, info=data.get("info", {}))
    gt.createIndex()
    dt = gt.loadRes(predictions)
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.imgIds = [im["id"] for im in data["images"]]
    evaluator.params.catIds = [1]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"ap50": float(evaluator.stats[1]), "ap50_95": float(evaluator.stats[0])}


def evaluate_files(annotations, predictions_path, output, operating_point=None):
    data = read_json(annotations)
    predictions = read_json(predictions_path)
    validate(data)
    validate_predictions(data, predictions)
    meta = read_json(str(predictions_path) + ".meta.json")
    if meta["annotations_sha256"] != sha256(annotations):
        raise ValueError("Predictions do not match these annotations")
    split = data.get("info", {}).get("split")
    if operating_point is None:
        result = select_threshold(data, predictions)
        result["provenance"] = meta
    else:
        if split not in {"val", "test"}:
            raise ValueError("Frozen operating point can only evaluate val/test")
        point = read_json(operating_point)
        if point["status"] != "qualified" or point["threshold"] is None:
            raise ValueError("No qualified operating point; improve training first")
        for key in ("checkpoint_sha256", "input_size", "nms", "score_floor", "upstream_commit"):
            if point["provenance"][key] != meta[key]:
                raise ValueError(f"Frozen operating point differs in {key}")
        report = operating_metrics(data, predictions, point["threshold"])
        m, targets = report["overall"], point["targets"]
        passed = (m["precision"] is not None and m["competitor_image_fpr"] is not None
                  and m["precision"] >= targets["minimum_precision"]
                  and m["competitor_image_fpr"] <= targets["maximum_competitor_image_fpr"])
        result = {"status": "targets_met" if passed else "targets_not_met", "split": split,
                  "operating_report": report, "provenance": meta,
                  "operating_point_sha256": sha256(operating_point)}
    result["ap"] = coco_ap(data, predictions)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new report path; do not overwrite evaluation evidence")
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--operating-point", help="Required for test; omit to select on val")
    a = parser.parse_args()
    print(evaluate_files(a.annotations, a.predictions, a.output, a.operating_point))


if __name__ == "__main__":
    main()
